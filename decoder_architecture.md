# MLPDecoder Architecture & Forward Pass

## Overview

The `MLPDecoder` in HiVT takes two embeddings per agent and outputs:
1. **Trajectory predictions** (mean location + uncertainty) for F modes
2. **Mode probabilities** (logits) indicating which mode is most likely

---

## Input Tensors

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `local_embed` | `[N, D_local]` | Per-agent local context (from LocalEncoder) |
| `global_embed` | `[F, N, D_global]` | Per-mode, per-agent global context (from GlobalInteractor) |

Where:
- `N` = number of agents
- `F` = num_modes (default 6)
- `D_local` = D_global = embed_dim (typically 64)

---

## Architecture Diagram

```
                    local_embed [N, D]           global_embed [F, N, D]
                         |                              |
                         |                              |
                    expand to [F, N, D]          reshape to [F*N, D]
                         |                              |
                         +---------- cat along dim=-1 ----------+
                                        |
                                [F, N, 2D]
                                        |
                         +---- split into two paths ----+
                         |                              |
                    (Path 1)                       (Path 2)
                         |                              |
        ╔════════════════════════╗      ╔════════════════════════╗
        ║    MODE PROBABILITY    ║      ║   TRAJECTORY DECODER   ║
        ║   (pi Block)           ║      ║                        ║
        ════════════════════════╝      ╚════════════════════════
        │                              │
        │ Linear(2D -> D)              │ **aggr_embed block:**
        │ LayerNorm                    │  ├─ Linear(2D -> D)
        │ ReLU                         │  ├─ LayerNorm
        │ Linear(D -> D)               │  └─ ReLU
        │ LayerNorm                    │  → out [F, N, D]
        │ ReLU                         │
        │ Linear(D -> 1)               ├─ **loc block:**
        │                              │  ├─ Linear(D -> D)
        │ → logits [F, N, 1]           │  ├─ LayerNorm
        │                              │  ├─ ReLU
        ├─ squeeze(-1) → [F, N]        │  └─ Linear(D -> H*2)
        │                              │  → loc [F, N, H*2]
        └─ transpose → [N, F]          │
             pi [N, F]                 ├─ scale block (if uncertain):
                                       │  ├─ Linear(D -> D)
                                       │  ├─ LayerNorm
                                       │  ├─ ReLU
                                       │  └─ Linear(D -> H*2)
                                       │  → scale [F, N, H*2]
                                       │
                                       ├─ ELU activation + shift
                                       │
                                       ├─ reshape & concatenate
                                       │
                                       └─ output [F, N, H, 4] or [F, N, H, 2]
```

---

## Step-by-Step Forward Pass

### Step 1: Input Concatenation

```python
# Concatenate local + global embeddings for mode probability prediction
cat_input = torch.cat((
    local_embed.expand(self.num_modes, *local_embed.shape),  # [F, N, D]
    global_embed                                              # [F, N, D]
), dim=-1)  # → [F, N, 2D]
```

Same concatenation is used for trajectory prediction, but passed through different blocks.

### Step 2A: Mode Probability Branch (pi block)

```python
# Input: [F, N, 2D]
# Network layers:
#   - Linear(2D + D) → D
#   - LayerNorm(D)
#   - ReLU
#   - Linear(D) → D
#   - LayerNorm(D)
#   - ReLU
#   - Linear(D) → 1

pi = self.pi(torch.cat((
    local_embed.expand(self.num_modes, *local_embed.shape),
    global_embed
), dim=-1))  # [F, N, 1]

pi = pi.squeeze(-1).t()  # [N, F] — one logit per (agent, mode)
```

**Meaning:** For each agent, scores each of the F modes on how likely it is.

### Step 2B: Trajectory Decoder Branch (aggr_embed + loc/scale blocks)

```python
# Step 1: Aggregate embeddings
out = self.aggr_embed(torch.cat((
    global_embed,                                             # [F, N, D]
    local_embed.expand(self.num_modes, *local_embed.shape)  # [F, N, D]
), dim=-1))  # [F, N, 2D] → [F, N, D]
```

**aggr_embed network:**
- Linear(2D → D)
- LayerNorm
- ReLU

This fuses mode-specific and agent-specific information into a joint representation.

---

### Step 3: Trajectory Mean (loc block)

```python
# Input: out [F, N, D]
# Network:
#   - Linear(D) → D
#   - LayerNorm
#   - ReLU
#   - Linear(D) → H*2  (H = future_steps)

loc = self.loc(out)  # [F, N, H*2]

# Reshape for clarity
loc = loc.view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
```

**Interpretation:**
- `loc[f, n, t, :]` = predicted (x, y) coordinates for mode f, agent n, at timestep t (0 ≤ t < H)

---

### Step 4: Uncertainty Estimation (if self.uncertain)

```python
# Input: out [F, N, D]
# Network:
#   - Linear(D) → D
#   - LayerNorm
#   - ReLU
#   - Linear(D) → H*2

scale = self.scale(out)  # [F, N, H*2]

# Apply ELU activation: ensures smooth output, then shift to ensure positivity
scale = F.elu_(scale, alpha=1.0)  # [F, N, H*2]
scale = scale.view(self.num_modes, -1, self.future_steps, 2) + 1.0  # [F, N, H, 2]

# Add minimum floor to prevent numerical issues
scale = scale + self.min_scale  # [F, N, H, 2]  (min_scale = 1e-3)
```

**Why these operations?**

1. **ELU activation:** Smooth, handles both negative and positive outputs smoothly.
2. **+1.0 shift:** Moves outputs into positive range (since scale ≥ 0 required).
3. **+min_scale:** Prevents scale from becoming too small (avoids divide-by-zero or numerical instability in loss).

**Interpretation:**
- `scale[f, n, t, :]` = predicted standard deviation for (x, y) at mode f, agent n, timestep t
- Represents **aleatoric uncertainty** (model's epistemic uncertainty about future outcomes)

---

### Step 5: Output Assembly

#### With Uncertainty (self.uncertain = True):

```python
# Concatenate mean and variance along feature dimension
output = torch.cat((loc, scale), dim=-1)  # [F, N, H, 4]

# Final output format:
# output[f, n, t, :] = [x_mean, y_mean, σ_x, σ_y]

return output, pi  # ([F, N, H, 4], [N, F])
```

#### Without Uncertainty (self.uncertain = False):

```python
return loc, pi  # ([F, N, H, 2], [N, F])
```

---

## Output Tensors

| Output | Shape | Meaning |
|--------|-------|---------|
| **y_hat** | `[F, N, H, 4]` (uncertain) or `[F, N, H, 2]` (certain) | Predicted trajectories (± uncertainty) |
| **pi** | `[N, F]` | Mode logits (before softmax) |

Where:
- `F` = num_modes (6)
- `N` = num_agents (32)
- `H` = future_steps (30)
- `y_hat[f, n, t, :]` = (x, y, σ_x, σ_y) for mode f, agent n, timestep t
- `pi[n, f]` = logit for agent n choosing mode f

---

## Loss Computation (Training Context)

During training, these outputs are compared to ground truth:

```python
# Compute L2 distance for each mode
l2_norm = torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1)  # [F, N]

# Find best mode (lowest error)
best_mode = l2_norm.argmin(dim=0)  # [N]

# Trajectory loss: only on best mode
reg_loss = LaplaceNLLLoss(y_hat_best, data.y)

# Probability loss: train pi to predict best mode
soft_target = F.softmax(-l2_norm / valid_steps, dim=0).t()
cls_loss = SoftTargetCrossEntropyLoss(pi, soft_target)

# Total loss
loss = reg_loss + cls_loss
```

---

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| **Two separate MLPs (pi vs. loc/scale)** | pi uses concatenated input; loc/scale use aggregated | Separates mode scoring from trajectory generation |
| **ELU + shift for scale** | Ensures σ > 0 and smooth gradients | Standard practice for predicting variances |
| **min_scale floor** | Prevents σ → 0 (numerical stability) | Typical in probabilistic models |
| **Soft targets for pi** | Use per-mode errors as soft labels | Encourages confident predictions on good modes |

---

## Example: Concrete Shapes

Suppose:
- N = 32 agents
- F = 6 modes
- H = 30 timesteps
- D = 64 (embed_dim)

**Forward pass shapes:**

```
Input:
  local_embed:  [32, 64]
  global_embed: [6, 32, 64]

After expand & cat:
  concat_in: [6, 32, 128]

pi block:
  input:  [6, 32, 128]
  output: [6, 32, 1]
  squeeze+transpose: [32, 6]  ← pi

Trajectory branch:
  aggr_embed_in:  [6, 32, 128]
  aggr_embed_out: [6, 32, 64]
  
  loc_out:   [6, 32, 60]  (30*2)
  loc_view:  [6, 32, 30, 2]  ← loc
  
  scale_out:   [6, 32, 60]
  scale_view:  [6, 32, 30, 2]  ← scale (after ELU+shift+floor)
  
  cat(loc, scale):  [6, 32, 30, 4]  ← y_hat

Output:
  y_hat: [6, 32, 30, 4]  (6 modes, 32 agents, 30 timesteps, (x,y,σ_x,σ_y))
  pi:    [32, 6]         (32 agents, 6 mode logits)
```


# DECODER NOTES

## Decoder

The decoder is implemented as `MLPDecoder` in [models/decoder.py](models/decoder.py#L87). It takes the global embeddings (which are mode-specific) and produces:
- `y_hat`: predicted future trajectories of shape [F, N, T, D]
- `pi`: predicted mode probabilities of shape [N, F]

### input to decoder

**local_emb**: [N, D] — per AGENT (N) local embedding from the local encoder. 
expand for modes: shape -> [F, N, D] 

**global_emb**: [F, N, D] — per-mode global embedding from the global interactor.

### code breakdown for decoder input processing
#### forward pass
**returns** loc,pi
shapes: [F, N, H, 4], [N, F] or  [F, N, H, 2], [N, F]

**Step 1**:
```python 
pi = self.pi(torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)).squeeze(-1).t()
```
**self.pi** is a **neural network** that outputs **one logit per mode**. **pi** represents mode probabilities—the predicted likelihood of each trajectory mode.


1. expand local embed
2. concatenate along feature dimension: [F, N, 2D]
```python
torch.cat((local_embed.expand(self.num_modes, *local_embed.shape),
                                global_embed), dim=-1)
```
(example If local_embed is [N=32, D=64] and global_embed is [6, 32, 64]: After expand: local becomes [6, 32, 64], After cat along dim=-1: result is [6, 32, 128] [MODES, AGENTS, 2*FEATURES])

3. Sequential layer self.pi is defined in [models/decoder.py](models/decoder.py#L121) as:

**final layer** is nn.Linear(..., 1)        
**Results**
Feeding a tensor of shape [F, N, feat] produces output shape [F, N, 1] (one scalar logit per (mode, agent)).
- `.squeeze(-1).t():` removes the trailing size-1 dimension → [F, N]
- .t() transposes the last two dims → [N, F]

Final pi is shape [N, F]: one logit per agent per mode (agents as rows, modes as columns).

---
**Step 2**:
```python
out = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
```

1. For each mode-agent pair, you get one combined feature vector.
2. pass through [self.aggr_embed](models/decoder.py#L106-109) **sequential MLP** to produce predicted trajectories for each mode:
Output out shape is [F, N, hidden_size]

*Question* 2 seperate MLPs find trajectories and probabilities. how is th trajectory related to the probability then?
they are coupled through training (regression loss, classification loss) but not explicitly in the architecture. The global embedding influences both, so it can learn to produce features that lead to accurate trajectories and correct mode probabilities.

***Step 3**:
```python
loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
```
1. Pass the combined features through another MLP (`self.loc`) to loc
2. self.loc is defined in [models/decoder.py](models/decoder.py#L110) as a sequential MLP that outputs `future_steps * 2` values per mode-agent pair (x and y for each future step).
2. Reshape the output to [F, N, H, 2], where H is the number of trajectory heads (e.g., 6) and 2 corresponds to (x, y) coordinates. `.view(self.num_modes, -1, self.future_steps, 2)`
 
 loc is the predicted trajectory mean/location for each mode-agent-timestep.

Shape [F, N, H, 2] means: for each of the 6 modes, for each of 32 agents, for each of 30 future timesteps, you get an (x, y) coordinate.

So if F=6, N=32, H=30, then [6, 32, 30, 2] represents 6 different 30-step trajectories per agent.
 
  ---
