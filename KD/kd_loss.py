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
        K=student modes, F=teacher modes, B=batch, H=horizon(time), D=coords
        """
        b_s = scale_s.clamp_min(self.min_scale)        # [K,B,H,D]

        mu = loc_s.unsqueeze(1)                         # [K,1,B,H,D]
        bb = b_s.unsqueeze(1)                           # [K,1,B,H,D]
        tg = targets.unsqueeze(0)                       # [1,F,B,H,D]

        # broadcasting mu and tg
        # axis:      0    1    2    3    4
        # mu :     [ K ,  1 ,  B ,  H ,  D ]
        # tg :     [ 1 ,  F ,  B ,  H ,  D ]
        #         ─────────────────────────
        # result:  [ K ,  F ,  B ,  H ,  D ]

        # Per-element Laplace log-density, then sum over horizon + coords.
        # log Lap(target | μ_k, b_k), per element
        #Lap(traj | mode k) = product over all timesteps t and coords d of:
                        # (1 / (2 * b[k,t,d])) * exp( -|y[t,d] - μ[k,t,d]| / b[k,t,d] )
        # P(whole trajectory | mode k) = P(x1)*P(y1)*P(x2)*P(y2)* ... = product of all the bumps

        log_dens = -torch.log(2.0 * bb) - (tg - mu).abs() / bb   # [K,F,B,H,D]
        ## Σ over H steps & D coords → log p(target|mode k)
        log_dens = log_dens.sum(dim=(-1, -2))                    # [K,F,B]

        # Add student log mixture weights:  [B,K] -> [K,1,B]
        # log p_student(traj) = log( sum over k of  exp( log π[k] + log Lap(traj | mode k) ) )
        log_w = log_pi_s.permute(1, 0).unsqueeze(1)              # [K,1,B]
        log_joint = log_w + log_dens                            # [K,F,B] #log_joint[k] already means log(π_k · p_k). log_w

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
        # Σ_f π_T[f] · log p_student(teacher-f)   =   E_{f ~ teacher}[ log p_student(teacher-f) ]

        total = self.lambda_kd * kd_loss
        return total, {
            "kd/mix_nll": kd_loss.detach(),
            "kd/total":   total.detach(),
        }

#NOTES
#Mean-target mixture distillation improves geometric accuracy and the official ranking metric
# (brier-minFDE) but degrades full-distribution calibration, because it transfers the teacher's
# mode locations and weights while discarding its predictive variance, yielding overconfident student
# mixtures.


class HiVTKDLossDist(HiVTKDLoss):
    """
    Distribution-matching distillation (v2).

    --------------------------------------------------------------------------
    WHY v2 EXISTS
    --------------------------------------------------------------------------
    The v1 loss (HiVTKDLoss) scores the teacher mode-MEANS as point targets and
    discards the teacher scales entirely. Maximising log p_student(mu_T) is
    achieved by shrinking the student's Laplace scales to pile density on those
    points, so the student becomes systematically OVER-CONFIDENT: best-of-K
    geometry (minADE/FDE) improves but the full-mixture calibration measured
    against ground truth (mixNLL) degrades. See docs/kd_emb32_kl_comparison.md.

    v2 distils the teacher's full predictive DISTRIBUTION instead of its means.
    It minimises the Monte-Carlo cross-entropy between the teacher mixture and
    the student mixture,

        L_KD = - (1/(H*D)) * sum_f pi_T_f * E_{y ~ Laplace(mu_T_f, b_T_f)}[ log p_student(y) ]
             ~ - (1/(H*D)) * sum_f pi_T_f * (1/S) sum_s log p_student(y_{f,s})

    with S reparameterised Laplace samples per teacher component. Because the
    student must now place density over the teacher's SPREAD (not just its mean
    points), collapsing the scales no longer minimises the loss -- the teacher's
    uncertainty is transferred, restoring calibration.

    Relationship to v1: in the zero-variance limit (sampling exactly at the
    teacher mean) v2 reduces to v1. v2 is therefore the strict generalisation
    that adds the discarded teacher-scale information back in.

    Permutation invariance is preserved: like v1 it scores teacher targets under
    the whole student mixture (logsumexp over student modes), so no Hungarian
    matching is required and K need not equal F.

    Extra arg:
        n_samples : Monte-Carlo samples drawn per teacher mode (default 8).
                    Larger = lower-variance gradient, linearly more compute.
    """

    def __init__(self, lambda_kl: float = 1.0, lambda_pi: float = 0.0,
                 min_scale: float = 1e-3, n_samples: int = 8):
        super().__init__(lambda_kl=lambda_kl, lambda_pi=lambda_pi,
                         min_scale=min_scale)
        self.n_samples = int(n_samples)

    def _sample_teacher(self, loc_t, scale_t):
        """Reparameterised Laplace samples. -> [S, F, B, H, D]."""
        b = scale_t.clamp_min(self.min_scale)
        shape = (self.n_samples,) + loc_t.shape          # [S,F,B,H,D]
        # u in (-0.5, 0.5); inverse CDF: x = mu - b * sign(u) * log(1 - 2|u|)
        u = torch.rand(shape, device=loc_t.device, dtype=loc_t.dtype) - 0.5
        arg = (1.0 - 2.0 * u.abs()).clamp_min(1e-6)
        return loc_t.unsqueeze(0) - b.unsqueeze(0) * torch.sign(u) * torch.log(arg)

    def forward(self, loc_s, scale_s, pi_s, loc_t, scale_t=None, pi_t=None):
        assert scale_t is not None, \
            "HiVTKDLossDist requires teacher scales; pass scale_t (it is cached)."
        assert loc_s.shape[1:] == loc_t.shape[1:], \
            f"B/H/coord mismatch: student {tuple(loc_s.shape)} vs teacher {tuple(loc_t.shape)}"
        assert pi_t is not None and pi_t.shape[0] == loc_t.shape[1], \
            f"teacher pi batch {tuple(pi_t.shape) if pi_t is not None else None} vs loc batch {tuple(loc_t.shape)}"

        S = self.n_samples
        Fm, B, H, D = loc_t.shape
        norm = float(H * D)

        log_pi_s  = F.log_softmax(pi_s, dim=-1)          # [B,K]
        pi_t_soft = torch.softmax(pi_t, dim=-1)          # [B,F]

        # Draw teacher samples and fold S into the "target" axis: [S*F, B, H, D]
#  For each teacher mode f, instead of taking just its center μ_T,f, we draw S random points (default S=8) scattered around that center, with the scatter width equal to the teacher's own b_T,f.
# Teacher very confident at some timestep (small b_T) → the 8 samples cluster tightly near the mean.
# Teacher unsure (large b_T) → the 8 samples spread out wide.
        samples = self._sample_teacher(loc_t, scale_t)   # [S,F,B,H,D]
        samples = samples.reshape(S * Fm, B, H, D)

        # log p_student over the K-mode mixture for every sample -> [S*F, B]
        # score every sample under the student mixture
        log_p = self._student_mixture_logprob(samples, loc_s, scale_s, log_pi_s)
        log_p = log_p.reshape(S, Fm, B).mean(dim=0)      # MC average -> [F,B]
#the average of log p_student over the cloud
        # Weight by teacher confidence, sum over teacher modes, per-element norm.
        w   = pi_t_soft.permute(1, 0)                    # [F,B]
        nll = -(w * log_p).sum(dim=0)                    # [B]
        kd_loss = nll.mean() / norm

        total = self.lambda_kd * kd_loss
        return total, {
            "kd/mix_nll": kd_loss.detach(),
            "kd/total":   total.detach(),
        }


def make_kd_loss(kd_mode: str = "mean", *, lambda_kl: float = 1.0,
                 lambda_pi: float = 0.0, min_scale: float = 1e-3,
                 n_samples: int = 8) -> HiVTKDLoss:
    """Factory selecting the KD loss variant.

    kd_mode="mean" -> HiVTKDLoss      (v1, mean-target; default, original behaviour)
    kd_mode="dist" -> HiVTKDLossDist  (v2, distribution-matching MC cross-entropy)
    """
    if kd_mode == "mean":
        return HiVTKDLoss(lambda_kl=lambda_kl, lambda_pi=lambda_pi,
                          min_scale=min_scale)
    if kd_mode == "dist":
        return HiVTKDLossDist(lambda_kl=lambda_kl, lambda_pi=lambda_pi,
                              min_scale=min_scale, n_samples=n_samples)
    raise ValueError(f"unknown kd_mode {kd_mode!r}; expected 'mean' or 'dist'")

# To run v2: add --kd_mode dist to your existing KD command (optionally --kd_n_samples 16 for a lower-variance gradient).

# One caveat worth a sanity check before a full run: v2's loss magnitude differs from v1 at the same λ (1.37 vs 1.14 in my random test), so the effective KD strength shifts — worth a quick triage λ for dist rather than assuming the mean λ transfers. 