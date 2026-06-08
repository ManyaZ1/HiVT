# Permutation-Invariant KD for HiVT — Goals & Changes

## Goal

Distil a large HiVT teacher (e.g. HiVT-128) into a small-capacity student
(HiVT-32) so the student's *best-of-K* accuracy (minADE/minFDE/MR) improves
beyond what it reaches from ground-truth supervision alone. The intended
mechanism: under HiVT's winner-takes-all (WTA) loss, only the single
GT-closest mode is supervised per scene, so a small student learns most of its
modes from a sparse, fragmented signal. The teacher holds an opinion about
*all* modes on *every* scene; KD transfers that dense signal toward the same
best-of-K objective.

## The bug this fixes

The previous `kd_loss.py` compared **student mode-k against teacher mode-k**
(index aligned) via a per-mode Laplace KL plus an index-aligned probability
cross-entropy.

HiVT's WTA objective is **symmetric under relabelling of the modes** — which
slot specialises to which behaviour is arbitrary per training run. The teacher
and student are independent runs (different init, width, epoch budget), so
their slot orderings do not correspond. The old loss therefore compared
*unrelated* trajectories and injected structured noise that fought the task
loss. This is a leading reason KD previously "did nothing," separate from the
student-capacity issue.

## The method now implemented

Treat the teacher's `F` mode-mean trajectories as a **set of targets** and
score them under the student's **entire `K`-mode Laplace mixture**, weighting
each teacher target by the teacher's softmax confidence:

```
L_KD = - (1 / (H*D)) * sum_f  pi_T_f * log p_student( mu_T_f )

log p_student(y) = logsumexp_k [ log pi_S_k
                                 + sum_{t,d} logLaplace(y; mu_S_k, b_S_k) ]
```

Scoring against the *mixture* (not slot-by-slot) makes the loss invariant to
the ordering of either model's modes — the permutation problem disappears and
**no Hungarian matching is needed**. `K` and `F` need not be equal.

Theoretical lineage: Bishop (1994) mixture-density likelihood + Hinton et al.
(2015) soft-target distillation, applied to a continuous Laplace mixture; the
"supervise against a set, pick the best fit" idea is the Multiple-Choice-
Learning / WTA family that HiVT itself uses for GT.

## Files changed

| File | Status | What changed |
|------|--------|--------------|
| `kd_loss.py` | **rewritten** | Index-aligned Laplace-KL + mode-CE replaced by permutation-invariant mixture-NLL. New log keys `kd/mix_nll`, `kd/total`. `lambda_pi` now ignored (warns if nonzero). Handles `K != F`. |
| `hivt_kd.py` | **edited** | KD call site no longer asserts `student.shape == teacher.shape` (only B/H/coords must match). `_last_metrics` now logs `train/kd/mix_nll`. `--lambda_pi` default 0 and documented as deprecated. |
| `kd_dataset.py` | unchanged | Still loads teacher `loc`/`scale`/`pi`. `scale` is loaded but **unused in v1** (means-only) — harmless. |
| `kd_datamodule.py` | unchanged | — |
| `train_student_kd.py` | unchanged | Existing launch command still works; `--lambda_pi 0.5` will now warn — set it to `0`. |
| `kd_mode_alignment_diagnostic.py` | **new** | Measures whether teacher/student mode indices accidentally align; produces the justification figure. Untested against your repo — adjust the marked blocks. |

## Loss-weight note

`L_KD` is rescaled by `1/(H*D)` so its magnitude sits near HiVT's per-element
regression loss; this keeps `--lambda_kl` on a sane scale but does **not**
remove the need to sweep it. Suggested starting point: `--lambda_task 1.0
--lambda_kl 0.5 --lambda_pi 0`. Watch `train/kd/mix_nll` and `train/loss_task`
in the same range; if KD dominates, lower `--lambda_kl`.

## Open items to verify before a long run

1. **Student scale parametrisation.** The mixture NLL uses the student scale
   `pred_s[..., 2:]`. Confirm in `models/hivt.py` that this is the
   **post-positivity-transform** scale (strictly positive). The loss clamps to
   `min_scale=1e-3` as a guard, but if the decoder emits raw pre-transform
   values the likelihood is degraded. (v1 does *not* use the teacher scale, so
   only the student side matters here.)
2. **Teacher cache key mismatch.** `kd_dataset._load_teacher` looks up
   `f"tensor({seq_id})"`, while `kd_datamodule._validate_teacher_cache` checks
   `store.exists(seq_id)` with the plain stem. If these key formats differ,
   validation can pass while every load returns `None`, silently disabling KD
   (`has_teacher=False`). Make both use the **same** key convention. (Send
   `kd_teacher_store_fixed.py` and `save_teacher_outputs.py` to confirm.)
3. **Single-batch sanity check.** Overfit one batch ~200 steps with KD on;
   `train/kd/mix_nll` should be finite and decreasing, and `total_loss`
   should fall. If `kd/mix_nll` never logs, `has_teacher` is False (see #2).

## Suggested experiment protocol (recap)

* Confirm the misalignment with `kd_mode_alignment_diagnostic.py` (one figure).
* Vanilla-32 vs KD-32, **both 64 epochs, same data subset, >=3 seeds each**,
  identical eval protocol. Teacher is precomputed (zero teacher cost/epoch).
* Optional ablation: permutation-invariant mixture-NLL (this) vs
  Hungarian-matched per-mode KL — shows the problem understood two ways.
* Report params, FLOPs, and inference latency on the target device alongside
  accuracy, since model compression is the thesis motivation.
