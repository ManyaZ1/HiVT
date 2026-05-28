"""
kd_loss.py

Knowledge distillation loss for HiVT output distributions.
Operates on Laplace mixture outputs from the MLPDecoder.

All tensors use the HiVT decoder convention:
    loc   : [B, F, H, 2]    Laplace means
    scale : [B, F, H, 2]    Laplace scales  (already >= min_scale, no softplus needed)
    pi    : [B, F]          raw logits  (pre-softmax)

The loss also accepts unbatched tensors shaped like [F, H, 2] and [F].

where F=num_modes, N=num_agents, H=future_steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class HiVTKDLoss(nn.Module):
    """
    Two-term KD loss:

    1. Weighted reverse KL between per-mode Laplace distributions.
       KL( Lap(μ_T, b_T) || Lap(μ_S, b_S) )  weighted by π_T.

       Reverse KL (teacher inside) makes the student mode-seeking:
       it commits sharply rather than spreading mass, which is the
       right inductive bias for minADE optimisation.

    2. Cross-entropy on mode probabilities:
       CE( softmax(π_S), softmax(π_T) )

    Args:
        lambda_kl   : weight for the Laplace KL term
        lambda_pi   : weight for the mode CE term
    """

    def __init__(self, lambda_kl: float = 0.5, lambda_pi: float = 0.5):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_pi = lambda_pi

    def forward(
        self,
        # student outputs
        loc_s:   torch.Tensor,   # [F, N, H, 2]
        scale_s: torch.Tensor,   # [F, N, H, 2]
        pi_s:    torch.Tensor,   # [N, F]
        # teacher outputs  (detached — no grad flows to teacher)
        loc_t:   torch.Tensor,   # [F, N, H, 2]
        scale_t: torch.Tensor,   # [F, N, H, 2]
        pi_t:    torch.Tensor,   # [N, F]
    ):
        if loc_s.dim() == 3:
            loc_s = loc_s.unsqueeze(0)
            scale_s = scale_s.unsqueeze(0)
        if loc_t.dim() == 3:
            loc_t = loc_t.unsqueeze(0)
            scale_t = scale_t.unsqueeze(0)
        if pi_s.dim() == 1:
            pi_s = pi_s.unsqueeze(0)
        if pi_t.dim() == 1:
            pi_t = pi_t.unsqueeze(0)

        # ------------------------------------------------------------------ #
        # 1. Per-mode Laplace KL
        #    torch.distributions computes the closed-form KL:
        #    KL(Lap(μ_T,b_T) || Lap(μ_S,b_S)) =
        #        log(b_S/b_T) + (b_T/b_S)*exp(-|μ_T-μ_S|/b_T) + |μ_T-μ_S|/b_S - 1
        # ------------------------------------------------------------------ #
        teacher_dist = D.Laplace(loc_t, scale_t)          # [B, F, H, 2]
        student_dist = D.Laplace(loc_s, scale_s)          # [B, F, H, 2]

        # kl shape: [B, F, H, 2]  — sum/mean over H and xy
        kl = D.kl_divergence(teacher_dist, student_dist)  # [F, N, H, 2]
        kl_per_mode = kl.mean(dim=(-1, -2))               # [B, F]

        # Weight each mode by teacher's soft mode probability
        pi_t_soft = torch.softmax(pi_t, dim=-1)           # [B, F]
        kl_loss = (pi_t_soft * kl_per_mode).sum(dim=-1).mean()   # scalar

        # ------------------------------------------------------------------ #
        # 2. Mode probability cross-entropy
        #    CE( π_S , π_T_soft )  — student log-probs vs teacher soft targets
        # ------------------------------------------------------------------ #
        log_pi_s = F.log_softmax(pi_s, dim=-1)            # [B, F]
        pi_ce_loss = -(pi_t_soft * log_pi_s).sum(dim=-1).mean()   # scalar

        total = self.lambda_kl * kl_loss + self.lambda_pi * pi_ce_loss

        return total, {
            "kd/kl_laplace": kl_loss.detach(),
            "kd/pi_ce":      pi_ce_loss.detach(),
            "kd/total":      total.detach(),
        }