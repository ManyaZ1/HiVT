"""
kd_loss.py

Permutation-invariant knowledge-distillation loss for HiVT Laplace-mixture
outputs.

--------------------------------------------------------------------------
WHY THE PREVIOUS VERSION WAS WRONG
--------------------------------------------------------------------------
HiVT trains its F mixture modes with a winner-takes-all (WTA) regression
objective: on each scene only the single mode closest to the ground truth
receives a regression gradient. The WTA loss is *symmetric under relabelling*
of the modes, so which slot specialises to a given behaviour ("turn left",
"keep straight", ...) is arbitrary per training run. Two independently trained
models -- a HiVT-128 teacher and a HiVT-32 student -- therefore have NO reason
to share the same slot ordering.

The old loss compared student-mode-k against teacher-mode-k (index aligned)
via a per-mode Laplace KL plus an index-aligned probability cross-entropy.
Across two independent models those indices do not correspond, so both terms
were comparing unrelated trajectories -- injecting structured noise that
fought the task loss. This is a leading reason KD "did nothing" in earlier
runs, independent of student capacity.

--------------------------------------------------------------------------
METHOD IMPLEMENTED HERE
--------------------------------------------------------------------------
Treat the teacher's F mode-mean trajectories as a SET of targets and score
them under the student's ENTIRE K-mode Laplace mixture, weighting each teacher
target by the teacher's softmax confidence:

    L_KD = - (1 / (H * D)) * sum_f  pi_T_f * log p_student( mu_T_f )

    log p_student(y) = logsumexp_k [ log pi_S_k
                                     + sum_{t,d} log Laplace(y_{t,d}; mu_S_{k,t,d}, b_S_{k,t,d}) ]

Because a mixture density is symmetric in its components, this loss is
invariant to the ordering of either model's modes -- the permutation problem
disappears and NO Hungarian matching is required. It is the Bishop (1994)
mixture-density log-likelihood with the teacher mode-means as observations and
the teacher mode-weights as sample weights; a continuous-output analogue of
Hinton et al. (2015) soft-target distillation.

The leading 1/(H*D) factor is a constant that rescales the summed-over-horizon
log-likelihood to a *per-element* quantity, so its magnitude is comparable to
HiVT's per-element regression loss. It does not change the optimum; it only
makes the lambda weight live on a sane scale. You may still want to sweep
lambda.

--------------------------------------------------------------------------
TENSOR CONVENTIONS (HiVT decoder)
--------------------------------------------------------------------------
    loc_s   : [K, B, H, 2]   student Laplace means       (K = student modes)
    scale_s : [K, B, H, 2]   student Laplace scales      (> 0, post-transform)
    pi_s    : [B, K]         student mode logits (pre-softmax)

    loc_t   : [F, B, H, 2]   teacher Laplace means        (F = teacher modes)
    scale_t : [F, B, H, 2]   teacher scales -- ACCEPTED BUT UNUSED in v1
    pi_t    : [B, F]         teacher mode logits (pre-softmax)

K (student modes) and F (teacher modes) need NOT be equal: the loss scores F
teacher means under the student's K-mode mixture. Only B, H and the coord dim
must agree.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiVTKDLoss(nn.Module):
    """
    Permutation-invariant mixture-NLL distillation loss.

    Args:
        lambda_kl : weight on the mixture-NLL distillation term.
                    (kept under the old name for CLI backward-compatibility)
        lambda_pi : DEPRECATED. The index-aligned mode cross-entropy is not
                    well defined across independently trained models, so it is
                    ignored here. The teacher's confidence already enters via
                    the per-target weighting. A warning is emitted if nonzero.
        min_scale : floor applied to the student scale before taking logs, to
                    guard against division by zero / log(0).
    """

    def __init__(self, lambda_kl: float = 1.0, lambda_pi: float = 0.0,
                 min_scale: float = 1e-3):
        super().__init__()
        self.lambda_kd = float(lambda_kl)
        self.min_scale = float(min_scale)
        if lambda_pi not in (0.0, None):
            warnings.warn(
                "HiVTKDLoss: lambda_pi is ignored by the permutation-invariant "
                "mixture-NLL loss (index-aligned mode CE is not meaningful "
                "across independently trained models). Pass --lambda_pi 0 to "
                "silence this warning.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Student mixture log-likelihood of a set of target trajectories.
    # ------------------------------------------------------------------ #
    def _student_mixture_logprob(self, targets, loc_s, scale_s, log_pi_s):
        """
        targets  : [F, B, H, D]   trajectories to score (teacher means)
        loc_s    : [K, B, H, D]
        scale_s  : [K, B, H, D]
        log_pi_s : [B, K]         student log mixture weights
        returns  : [F, B]         log p_student(target_f)  for each f, b
        """
        b_s = scale_s.clamp_min(self.min_scale)        # [K,B,H,D]

        mu = loc_s.unsqueeze(1)                         # [K,1,B,H,D]
        bb = b_s.unsqueeze(1)                           # [K,1,B,H,D]
        tg = targets.unsqueeze(0)                       # [1,F,B,H,D]

        # Per-element Laplace log-density, then sum over horizon + coords.
        log_dens = -torch.log(2.0 * bb) - (tg - mu).abs() / bb   # [K,F,B,H,D]
        log_dens = log_dens.sum(dim=(-1, -2))                    # [K,F,B]

        # Add student log mixture weights:  [B,K] -> [K,1,B]
        log_w = log_pi_s.permute(1, 0).unsqueeze(1)              # [K,1,B]
        log_joint = log_w + log_dens                            # [K,F,B]

        # Mixture log-likelihood = logsumexp over the student's K modes.
        return torch.logsumexp(log_joint, dim=0)                # [F,B]

    # ------------------------------------------------------------------ #
    def forward(self, loc_s, scale_s, pi_s, loc_t, scale_t=None, pi_t=None):
        # B, H and coord dims must match; K and F may differ.
        assert loc_s.shape[1:] == loc_t.shape[1:], \
            f"B/H/coord mismatch: student {tuple(loc_s.shape)} vs teacher {tuple(loc_t.shape)}"
        assert pi_s.shape[0] == loc_s.shape[1], \
            f"student pi batch {tuple(pi_s.shape)} vs loc batch {tuple(loc_s.shape)}"
        assert pi_t is not None and pi_t.shape[0] == loc_t.shape[1], \
            f"teacher pi batch {tuple(pi_t.shape) if pi_t is not None else None} vs loc batch {tuple(loc_t.shape)}"

        H = loc_s.shape[2]
        D = loc_s.shape[3]
        norm = float(H * D)            # per-element rescale for lambda sanity

        log_pi_s  = F.log_softmax(pi_s, dim=-1)        # [B,K]
        pi_t_soft = torch.softmax(pi_t, dim=-1)        # [B,F]

        # log p_student(teacher mean_f)  ->  [F,B]
        log_p = self._student_mixture_logprob(loc_t, loc_s, scale_s, log_pi_s)

        # Weight each teacher target by teacher confidence; sum over teacher
        # modes, mean over batch, rescale to per-element.
        w   = pi_t_soft.permute(1, 0)                  # [F,B]
        nll = -(w * log_p).sum(dim=0)                  # [B]
        kd_loss = nll.mean() / norm                    # scalar

        total = self.lambda_kd * kd_loss
        return total, {
            "kd/mix_nll": kd_loss.detach(),
            "kd/total":   total.detach(),
        }
