# KD on HiVT-32 — kl=0.0 vs kl=0.5 (Phase C, full data)

Student: **HiVT-32** · Teacher: **HiVT-128** (pretrained) · Recipe: `bs=128`, `lr=3e-3`, full Argoverse, ~64 epochs.
Both runs use the **same** KD harness/recipe; the only difference is `lambda_kl` (0.0 = no distillation, just the student trained inside the KD framework).

Checkpoints (best by `val_minFDE`):
- `kl=0.0`: `emb32-bs128-lkl0.0/best/HiVTKD-epoch=46-val_minFDE=1.23.ckpt`
- `kl=0.5`: `emb32-bs128-lkl0.5/best/HiVTKD-epoch=62-val_minFDE=1.12.ckpt`

## Head-to-head (the clean A/B — same recipe, only λ_kl changes)

| Metric | kl=0.0 | kl=0.5 | Δ (abs) | Δ (%) | Better? |
|---|---:|---:|---:|---:|:--:|
| **val_minADE** | 0.7670 | 0.7264 | −0.0406 | **−5.30%** | ✅ |
| **val_minFDE** | 1.2293 | 1.1211 | −0.1082 | **−8.80%** | ✅ |
| **val_minMR** | 0.12895 | 0.11788 | −0.01107 | **−8.59%** | ✅ |
| val_brier_minADE | 1.4255 | 1.3977 | −0.0279 | −1.95% | ✅ |
| val_brier_minFDE | 1.8878 | 1.7924 | −0.0954 | −5.05% | ✅ |
| val_p_minFDE | 2.9227 | 2.8522 | −0.0705 | −2.41% | ✅ |
| val_p_minADE | 2.4604 | 2.4575 | −0.0030 | −0.12% | ≈ |
| val_p_MR | 0.83154 | 0.83671 | +0.00517 | +0.62% | ❌ |
| **val_mixNLL** | 28.739 | 34.658 | +5.919 | **+20.60%** | ❌ |
| val_reg_loss | −0.1991 | −0.1461 | +0.0530 | (less neg.) | ❌ |

Lower is better for every row except `reg_loss` (more negative = better) — kl=0.5 is *less* negative there, i.e. worse.

### Reading it
- **Displacement / geometry metrics improve substantially and consistently.** minFDE −8.8%, minMR −8.6%, minADE −5.3%, plus both brier variants. This is the headline KD effect and it is **not** razor-thin.
- **Probabilistic / calibration metrics get worse.** mixNLL +20.6%, reg_loss less negative, p_MR slightly up. The KD soft target pulls the predicted trajectories toward the teacher's modes (great for best-of-K geometry) but the **mixture probability assignment and Laplace-scale calibration degrade.**
- The W&B curves confirm this is a genuine trade-off, not noise: `val_mixNLL` for kl=0.5 bottoms around ~30k steps then **rises** to ~38, while kl=0.0 keeps descending to ~28. Meanwhile kl=0.5's minADE/minFDE stay below kl=0.0 for the entire run.

## Sanity check vs the real baselines

| Model | Setup | minADE | minFDE | minMR |
|---|---|---:|---:|---:|
| HiVT-32 (original) | bs32, 64 ep, no KD | 0.7446 | 1.1782 | 0.12505 |
| HiVT-32 KD **kl=0.0** | bs128, KD harness | 0.7670 | 1.2293 | 0.12895 |
| **HiVT-32 KD kl=0.5** | bs128, KD harness | **0.7264** | **1.1211** | **0.11788** |
| HiVT-64 | author ckpt | 0.6869 | 1.0301 | 0.10263 |
| HiVT-128 (teacher) | author ckpt | 0.6611 | 0.9692 | 0.09204 |

Two important nuances:
1. **The kl=0.0 run is actually *worse* than the original HiVT-32** (minFDE +4.3%, minADE +3.0%). So the kl=0.0 arm is a slightly weakened in-harness baseline (bs128 vs bs32, best ckpt at ep46 then plateau). Part of the kl=0.5-vs-kl=0.0 gap reflects that.
2. **But kl=0.5 still beats the strong original HiVT-32** — minFDE −4.84%, minMR −5.73%, minADE −2.44%. So KD is a real improvement over the best honest same-size baseline, not just over a crippled one.

**Gap closed toward HiVT-64** (on minFDE): standalone HiVT-32 is 0.1481 above HiVT-64; KD kl=0.5 narrows that to 0.0911 → **~38% of the HiVT-32→HiVT-64 gap recovered**, at zero inference cost. Still ~+15.7% above the HiVT-128 teacher.

## Answers to the three questions

**Is KD meaningful?** — **Yes, clearly, on displacement.** kl=0.5 improves every geometry metric by 5–9% over the matched kl=0.0 baseline and beats the original HiVT-32 too. The catch: it *costs* probabilistic calibration (mixNLL +20.6%). If minFDE/minADE/MR is the deliverable, KD wins outright. If calibrated mixture likelihood matters, that regression is a real lever to address.

**Are epochs / batch size optimal?** — **Close enough; not the bottleneck.**
- The minADE/minFDE curves flatten by ~60–80k steps and are nearly converged at 64 epochs; kl=0.5's best ckpt at ep62 is still edging down slightly, so a longer run might shave a hair more, but diminishing returns. 64 epochs is reasonable.
- bs128 + lr3e-3 curves are smooth with no early instability → healthy optimization. No reason to change the batch size.
- The real issue isn't epochs/bs, it's the **loss balance**: mixNLL overfits/rises mid-training while displacement keeps improving, and we select on minFDE, so we deliberately keep the geometrically-best-but-calibration-degraded model. That's a λ_kl / loss-weighting question, not an epochs question.

**Smaller student next, or is capacity too small / try other methods?** — **Go smaller (Phase D, emb16).**
- emb32 **cleanly benefited** from KD → capacity is *not* too small, and there's no evidence we need to pivot to other methods.
- The compelling research question is whether the KD benefit **grows** as the student shrinks (KD typically helps more for smaller students, up to a floor). emb16 directly tests that.
- Carry the recipe over but **re-triage LR with 1–2 short runs** (a smaller model can shift the optimum), and keep both a `kl=0.0` and `kl=0.5` arm so the emb16 A/B is as clean as this one.

---

## Mechanism — why mixNLL gets *worse* (the central finding)

The intuition "KD matches output distributions, so calibration should improve" only holds for a *distribution-matching* loss. The v1 loss in [`KD/kd_loss.py`](../KD/kd_loss.py) does **not** match distributions. Its objective is

```
L_KD = − Σ_f  π_T(f) · log p_student( μ_T(f) )
```

It scores the teacher's mode-**means** `loc_t` as **point observations** under the student mixture, and the teacher scales `scale_t` are **accepted but unused** (see the loss docstring + `forward` signature). So the only teacher information transferred is *where the modes are* and *how the modes are weighted* — the teacher's **predictive variance is discarded**.

What does that gradient reward? Maximizing `log p_student(μ_T)` is achieved by **shrinking the student's Laplace scales** to pile density exactly on the teacher's mean points. The student is pushed to be **sharper / more confident than the teacher ever was**. This is mean-/mode-seeking distillation — the continuous analogue of distilling *hard* predictions, not soft ones.

That single mechanism explains every metric movement:

| KD (v1) rewards… | measured by | direction | observed |
|---|---|---|---|
| best mode anchored to a good teacher mean | minADE/FDE/MR (oracle best-of-6) | better | −5…−9% ✅ |
| high probability on the (oracle-)best mode | brier-minFDE, p_minFDE | better | −2…−5% ✅ |
| sharp density at teacher *means* | mixNLL, reg_loss (at **ground truth**) | worse | +20.6% / less-neg ❌ |

`mixNLL`/`reg_loss` are evaluated at the **ground truth**, while the KD loss sharpens density at the **teacher mean**. Teacher mean ≠ ground truth on average, so sharpening pulls density away from the truth and thins the tails → the log penalty blows up. **Overconfidence is the predicted consequence of distilling means and discarding variance**, not a bug.

> **Thesis statement:** *Mean-target mixture distillation improves geometric accuracy and the official Argoverse ranking metric (brier-minFDE) but degrades full-distribution calibration (mixNLL), because it transfers the teacher's mode locations and weights while discarding its predictive variance, producing overconfident student mixtures.*

Note `brier-minFDE`/`p_minFDE` improve while `mixNLL` worsens because the former only inspect the *best* mode's probability, whereas `mixNLL` integrates the *whole* predictive density at the truth. The two "calibration" families measuring different things is itself a reportable observation.

`val_p_MR` (+0.62%) is **not** a counter-result: it is a 2.0 m-thresholded count (hits add `1−p_best`, misses add `1.0`), quantized and within single-run noise. Lean on the continuous `p_minFDE`/`brier_minFDE`, not on `p_MR`.

## v2 — distribution-matching loss (implemented, opt-in)

The fix is to distil the teacher's full predictive *distribution* instead of its means. `HiVTKDLossDist` in [`KD/kd_loss.py`](../KD/kd_loss.py) minimizes the Monte-Carlo cross-entropy between the teacher and student mixtures:

```
L_KD = − (1/HD) · Σ_f π_T(f) · E_{y ~ Laplace(μ_T_f, b_T_f)}[ log p_student(y) ]
```

Because the student must now place density over the teacher's **spread**, collapsing its scales no longer minimizes the loss — the discarded variance is transferred back. It is permutation-invariant exactly like v1, and in the zero-variance limit (`b_T→0`) it **reduces exactly to v1** (verified numerically). The teacher scales are already in the cache, so no re-caching is needed.

- **v1 (default, unchanged):** `--kd_mode mean`
- **v2 (new):** `--kd_mode dist [--kd_n_samples 8]`

Clean thesis arc: *v1 trades calibration for geometry → diagnose via sharpness/coverage → v2 recovers calibration.*

## Next experiments (ordered)

1. **λ dose-response sweep** — `kd_mode=mean`, λ ∈ {0, 0.25, 0.5, 1.0}, on the **cheap triage setting** (25% data, 15 epochs), not full runs. Hypothesis: as λ↑, minFDE/brier improve monotonically while mixNLL degrades monotonically. Two clean crossing curves *prove* the mechanism — the strongest single figure for the thesis. Then confirm the chosen λ on full data.
2. **v1 vs v2 at fixed λ** — same triage setting, `kd_mode ∈ {mean, dist}`. Expect v2 to recover most of the mixNLL loss (and ideally improve it past the no-KD baseline) while keeping most of the geometric gain.
3. **Phase D — emb16 transfer** — re-triage LR (1–2 short runs), then full runs for both kl arms. Tests whether the KD benefit *grows* as the student shrinks.

## Metrics to add (the mixNLL story demands measuring the mechanism)

- **Sharpness** — mean predicted Laplace scale `b`. The smoking gun: v1 should visibly shrink it; v2 should not. Cheap to log.
- **Reliability / coverage calibration** — fraction of ground-truth endpoints inside the predicted X% region vs X. A calibration curve *visually proves* over-confidence; the figure reviewers will want.
- **Mode diversity / collapse** — effective number of modes used, mean pairwise mode separation, entropy of `π`. Mean-seeking KD risks collapse.
- **Efficiency axis (essential — it's the point of a small student)** — params, FLOPs, measured inference latency/throughput for HiVT-32/64/128. The narrative "recover X% of accuracy at Y% of cost" needs Y.
- **Multiple seeds** — several deltas here (brier_minADE −1.95%, p_MR +0.62%) are within plausible single-run variance. **3 seeds per condition** with mean±std makes every claim defensible.