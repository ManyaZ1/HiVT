import torch
import torch.nn as nn
import math

class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal PE for 2D points, projects to model dim D."""
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
    
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: (K, 2)  — intention point coordinates
        returns: (K, D)
        """
        K = xy.shape[0]
        D_half = self.d_model // 2          # D/2 dims per axis
        
        div_term = torch.exp(
            torch.arange(0, D_half, device=xy.device).float()
            * -(math.log(10000.0) / D_half)
        )                                   # (D/2,)
        
        x = xy[:, 0:1] * div_term           # (K, D/2)
        y = xy[:, 1:2] * div_term           # (K, D/2)
        
        pe = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(y), torch.cos(y)
        ], dim=-1)                          # (K, 2D) → needs projection
        
        return pe                           # (K, D_half*4) — project below


class StaticIntentionQuery(nn.Module):
    """
    MTR-style static intention queries for HiVT.
    
    Replaces HiVT's fixed K anchor queries with intention-point-
    conditioned learnable queries, one per motion mode.
    """
    def __init__(
        self,
        intention_points: np.ndarray,   # (K, 2) precomputed from k-means
        d_model: int = 64,              # HiVT hidden dim
        mlp_hidden: int = 256
    ):
        super().__init__()
        K = intention_points.shape[0]
        self.K = K
        self.d_model = d_model
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer(
            "intention_points",
            torch.tensor(intention_points, dtype=torch.float32)  # (K, 2)
        )
        
        # PE projects 2D → d_model
        self.pe = SinusoidalPositionEncoding(d_model)
        
        # MLP: PE(I) → Q_I  (Eq. 4 in MTR paper)
        pe_dim = d_model * 2            # sin/cos for x and y each
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, d_model)
        )
    
    def forward(self) -> torch.Tensor:
        """
        Returns intention queries Q_I: (K, D)
        Call once per forward pass; broadcast over batch in attention.
        """
        pe = self.pe(self.intention_points)     # (K, 2D)
        Q_I = self.mlp(pe)                      # (K, D)
        return Q_I
class HiVTDecoderWithIntentionQueries(nn.Module):
    """
    Drop-in replacement for HiVT's prediction head.
    Uses MTR-style intention queries instead of mode embeddings.
    """
    def __init__(
        self,
        intention_points: np.ndarray,
        d_model: int = 64,
        future_steps: int = 30,
        num_heads: int = 8,
    ):
        super().__init__()
        self.K = intention_points.shape[0]
        self.future_steps = future_steps
        
        # Intention query generator (static, shared across batch)
        self.intention_query = StaticIntentionQuery(
            intention_points, d_model
        )
        
        # Cross-attention: queries=intentions, keys/values=agent context
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Trajectory regression head: (D) → (T*2)
        self.traj_head = nn.Linear(d_model, future_steps * 2)
        
        # Confidence head
        self.conf_head = nn.Linear(d_model, 1)
    
    def forward(self, agent_context: torch.Tensor) -> tuple:
        """
        agent_context: (B, D) — per-agent encoding from HiVT encoder
        
        Returns:
            trajectories: (B, K, T, 2)
            confidences:  (B, K)
        """
        B, D = agent_context.shape
        
        # Q_I: (K, D) → (B, K, D)  broadcast over batch
        Q_I = self.intention_query()                        # (K, D)
        queries = Q_I.unsqueeze(0).expand(B, -1, -1)       # (B, K, D)
        
        # Context as keys/values: (B, 1, D) → attend over K modes
        context = agent_context.unsqueeze(1)                # (B, 1, D)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(queries, context, context)
        queries = self.norm(queries + attn_out)             # (B, K, D)
        queries = self.norm2(queries + self.ffn(queries))   # (B, K, D)
        
        # Decode trajectories and confidences
        traj = self.traj_head(queries)                      # (B, K, T*2)
        traj = traj.view(B, self.K, self.future_steps, 2)  # (B, K, T, 2)
        
        conf = self.conf_head(queries).squeeze(-1)          # (B, K)
        
        return traj, conf