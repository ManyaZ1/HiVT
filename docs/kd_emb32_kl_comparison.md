# KD on HiVT-32 — kl=0.0 vs kl=0.5 (Phase C, full data)

Student: **HiVT-32** · Teacher: **HiVT-128** (pretrained) · Recipe: `bs=128`, `lr=3e-3`, full Argoverse, ~64 epochs.
Both runs use the **same** KD harness/recipe; the only difference is `lambda_kl` (0.0 = no distillation, just the student trained inside the KD framework).

> **CORRECTION (2026-06-22):** earlier versions of this doc used the *older* `emb32-bs128-lkl0.x`
> checkpoints (June 12/16) as canonical. Those are superseded by the June-18 `triage-…-full` runs trained
> with the fixed teacher cache (`train_fix.h5`), which are better and are the runs in wandb group
> `full-emb32-lr3e-3`. All full-data numbers below are now from those correct runs. Net effect: KD's geometry
> gain is *larger* than previously recorded, v1's calibration damage is *worse*, and **v2 ties v1 on geometry**
> (the previously-claimed v2 geometry win was a stale-baseline artifact). The mechanism and the v2 calibration
> fix are unchanged (in fact sharper).

Checkpoints (best by `val_minFDE`, correct `train_fix.h5` full runs):
- `kl=0.0`: `triage-emb32-lr3e-3-kl0.0-full/best/HiVTKD-epoch=54-val_minFDE=1.16.ckpt`
- `kl=0.5`: `triage-emb32-lr3e-3-kl0.5-full/best/HiVTKD-epoch=58-val_minFDE=1.05.ckpt`
- `kl=0.5 v2 (dist)`: `emb32-bs128-lkl0.5-distv2-full/best/HiVTKD-epoch=63-val_minFDE=1.05.ckpt`

## Head-to-head (the clean A/B — same recipe, only λ_kl changes)

| Metric | kl=0.0 | kl=0.5 | Δ (abs) | Δ (%) | Better? |
|---|---:|---:|---:|---:|:--:|
| **val_minADE** | 0.7365 | 0.6958 | −0.0407 | **−5.53%** | ✅ |
| **val_minFDE** | 1.1574 | 1.0509 | −0.1065 | **−9.20%** | ✅ |
| **val_minMR** | 0.12252 | 0.10557 | −0.01695 | **−13.83%** | ✅ |
| val_brier_minADE | 1.3989 | 1.3684 | −0.0305 | −2.18% | ✅ |
| val_brier_minFDE | 1.8197 | 1.7235 | −0.0962 | −5.29% | ✅ |
| val_p_minFDE | 2.8635 | 2.7854 | −0.0781 | −2.73% | ✅ |
| val_p_minADE | 2.4426 | 2.4303 | −0.0123 | −0.50% | ≈ |
| val_p_MR | 0.83222 | 0.83549 | +0.00327 | +0.39% | ❌ |
| **val_mixNLL** | 26.695 | 37.624 | +10.929 | **+40.94%** | ❌ |
| **val_calib_err** | 0.0362 | 0.1827 | +0.1465 | **+405%** (5.0×) | ❌ |
| val_b_scale | 0.4129 | 0.2678 | −0.1451 | −35.1% (shrink) | ❌ |
| val_reg_loss | −0.2459 | −0.1876 | +0.0583 | (less neg.) | ❌ |

Lower is better for every row except `reg_loss` (more negative = better) and `b_scale` (smaller = sharper, which here means over-confident). kl=0.5 is worse on `reg_loss`/`b_scale`/`calib_err`/`mixNLL`.

### Reading it
- **Displacement / geometry metrics improve substantially and consistently.** minFDE −9.2%, minMR −13.8%, minADE −5.5%, plus both brier variants. This is the headline KD effect and it is **not** razor-thin.
- **Probabilistic / calibration metrics get much worse.** mixNLL +40.9%, calib_err 5×, b_scale −35% (over-confident scale shrinkage), reg_loss less negative. The KD soft target pulls trajectories toward the teacher's modes (great for best-of-K geometry) but the **Laplace-scale calibration collapses.**
- The W&B curves confirm a genuine trade-off, not noise: `val_mixNLL` for kl=0.5 bottoms then **rises**, while kl=0.0 keeps descending. Meanwhile kl=0.5's minADE/minFDE stay below kl=0.0 throughout.

## Sanity check vs the real baselines

| Model | Setup | minADE | minFDE | minMR |
|---|---|---:|---:|---:|
| HiVT-32 (original) | bs32, 64 ep, no KD | 0.7446 | 1.1782 | 0.12505 |
| HiVT-32 KD **kl=0.0** | bs128, KD harness | 0.7365 | 1.1574 | 0.12252 |
| **HiVT-32 KD kl=0.5 (v1)** | bs128, mean-target | **0.6958** | **1.0509** | **0.10557** |
| **HiVT-32 KD kl=0.5 (v2 dist)** | bs128, dist-match | 0.6982 | 1.0505 | 0.10630 |
| HiVT-64 | author ckpt | 0.6869 | 1.0301 | 0.10263 |
| HiVT-128 (teacher) | author ckpt | 0.6611 | 0.9692 | 0.09204 |

Notes:
1. The kl=0.0 in-harness baseline (1.1574) is now slightly **better** than the original HiVT-32 (1.1782), so the KD gain is measured against an honest, not weakened, baseline.
2. **KD kl=0.5 beats the original HiVT-32 by −10.8% minFDE** and lands essentially **on top of HiVT-64** (1.0301) — a model with 2× the width.
3. **v1 and v2 are tied on geometry** (1.0509 vs 1.0505); they differ only in calibration (see the v2 section).

**Gap closed toward HiVT-64** (on minFDE): original HiVT-32 is 0.1481 above HiVT-64; KD kl=0.5 (both variants) narrows that to ~0.02 → **~86% of the HiVT-32→HiVT-64 gap recovered**, at zero inference cost. Still ~+8.4% above the HiVT-128 teacher.

## Answers to the three questions

**Is KD meaningful?** — **Yes, clearly, on displacement.** kl=0.5 improves every geometry metric by 5–14% over the matched kl=0.0 baseline and beats the original HiVT-32 too. The catch: it *costs* probabilistic calibration (mixNLL +40.9%). If minFDE/minADE/MR is the deliverable, KD wins outright. If calibrated mixture likelihood matters, that regression is a real lever to address.

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
| sharp density at teacher *means* | mixNLL, reg_loss (at **ground truth**) | worse | +40.9% / less-neg ❌ |

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

1. ✅ **DONE — λ dose-response sweep** (see "λ dose-response" section below). Result refined the hypothesis:
   calibration (`b_scale`/`calib_err`/coverage) degrades *monotonically* with λ, but *geometry peaks at
   λ≈0.25* (not monotone), and mixNLL is U-shaped (min at 0.25 on triage). Leading follow-up: confirm
   **λ=0.25 on full data** — it dominated λ=0.5 on every triage metric.
2. **v1 vs v2 at fixed λ** — same triage setting, `kd_mode ∈ {mean, dist}`. Expect v2 to recover most of the mixNLL loss (and ideally improve it past the no-KD baseline) while keeping most of the geometric gain.
3. **Phase D — emb16 transfer** — re-triage LR (1–2 short runs), then full runs for both kl arms. Tests whether the KD benefit *grows* as the student shrinks.

## Diagnostic metrics — sharpness / coverage / diversity (Task #2)

Re-evaluated both checkpoints with `eval.py` after adding the diagnostics (sharpness, Laplace
coverage, π-entropy) to `HiVT.validation_step`. All other rows are unchanged from above (and
reproduced exactly, confirming the metrics were added without disturbing the load/validation path).

### Sharpness, calibration error, mode-weight entropy

| Metric | kl=0.0 | kl=0.5 | Direction | Confirms over-confidence? |
|---|---:|---:|---|:--:|
| **val_b_scale** (chosen-mode `b`) | 0.4129 | 0.2678 | smaller `b` ⇒ sharper | ✅ −35.1% |
| val_b_scale_all (all-mode `b`) | 0.4698 | 0.2995 | smaller `b` ⇒ sharper | ✅ −36.2% |
| **val_calib_err** (mean_p \|emp−p\|) | 0.0362 | 0.1827 | higher ⇒ worse calibration | ✅ 5.0× worse |
| val_pi_entropy (mode-weight entropy) | 1.7423 | 1.7595 | lower ⇒ mode collapse | ❌ ~equal (+1.0%) |

### Reliability curve — empirical coverage vs nominal `p`

| nominal `p` | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kl=0.0** empirical | 0.073 | 0.154 | 0.242 | 0.339 | 0.445 | 0.557 | 0.674 | 0.791 | 0.900 |
| **kl=0.5** empirical | 0.043 | 0.091 | 0.145 | 0.206 | 0.276 | 0.358 | 0.454 | 0.569 | 0.713 |

Both models are nominally *slightly* under-covered, but **kl=0.5 is dramatically more under-covered at
every level** — e.g. its 90% predicted interval captures only 71.3% of ground-truth points vs 90.0% for
kl=0.0. Empirical < nominal everywhere ⇒ predicted Laplace intervals are too narrow ⇒ over-confident.

### Falsifiable verdict

The prediction was that the more-distilled student (kl=0.5) should show **smaller `b`**, **worse coverage /
higher calib error**, and **lower π-entropy** than kl=0.0. Result: **2 of 3 confirmed.**

- ✅ **Sharpness:** kl=0.5 mean `b` is ~31% smaller — the smoking gun. v1 mean-target KD visibly shrinks
  the Laplace scales, exactly as the mechanism predicts.
- ✅ **Coverage / calibration:** kl=0.5 is under-covered at every nominal level and its scalar calibration
  error is 5× larger (0.183 vs 0.036). The reliability curve *visually proves* the over-confidence and
  tracks the +40.9% mixNLL regression.
- ❌ **Mode-weight entropy:** essentially unchanged (1.758 vs 1.740). **The over-confidence lives in the
  Laplace *scales*, not in mixture-weight collapse** — consistent with the mechanism (the v1 loss sharpens
  density at teacher *means*; it does not by itself collapse the mode weights). The π-entropy proxy is
  therefore a *negative* control here, not a failure of the metric.

Bottom line: the new sharpness + coverage metrics successfully capture the predicted over-confidence and
make the central thesis claim falsifiable and confirmed. Mode collapse is not the mechanism; scale shrinkage is.

## λ dose-response (calibration across distillation strength)

Re-evaluated the four `-cal` **triage** checkpoints (25% data, 15 epochs, lr=3e-3 — apples-to-apples),
each judged on the **full** val set. This completes the λ axis for the diagnostic metrics.

| λ | minFDE | minADE | minMR | mixNLL | b_scale | b_scale_all | calib_err | π-entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.683 | 0.925 | 0.207 | 37.93 | 0.5420 | 0.5970 | 0.0245 | 1.7382 |
| **0.25** | **1.369** | **0.823** | 0.153 | **33.86** | 0.4055 | 0.4515 | 0.0925 | 1.7520 |
| 0.5 | 1.393 | 0.835 | 0.153 | 35.74 | 0.3919 | 0.4139 | 0.1204 | 1.7554 |
| 1.0 | 1.416 | 0.844 | 0.156 | 40.59 | 0.3662 | 0.3940 | 0.1547 | 1.7573 |

Reliability — empirical coverage vs nominal `p` (monotone under-coverage that deepens with λ):

| nominal `p` | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| λ=0.0 | .078 | .163 | .256 | .358 | .467 | .582 | .700 | .812 | .911 |
| λ=0.25 | .061 | .127 | .202 | .285 | .376 | .478 | .590 | .711 | .838 |
| λ=0.5 | .055 | .116 | .184 | .259 | .344 | .440 | .548 | .668 | .803 |
| λ=1.0 | .049 | .102 | .162 | .230 | .306 | .394 | .496 | .614 | .754 |

### What the dose-response shows

- **Scale shrinkage is monotone in λ.** `b_scale` 0.542 → 0.405 → 0.392 → 0.366 (and `b_scale_all` likewise).
  Stronger distillation ⇒ tighter modes, at every dose.
- **Calibration degrades monotonically.** `calib_err` 0.024 → 0.093 → 0.120 → 0.155 (6.3× from λ=0 to λ=1),
  and the reliability curve sits progressively further below the diagonal at *every* nominal level. λ=0 is
  well-calibrated; KD breaks it in proportion to λ.
- **No mode collapse at any dose.** π-entropy is flat — in fact *rises* slightly (1.738 → 1.757). The
  over-confidence is purely within-mode scale, confirmed across the whole λ axis, not just at λ=0.5.
- **Geometry is NOT monotone — it peaks at λ≈0.25.** minFDE 1.683 → **1.369** → 1.393 → 1.416; minADE and
  brier-minFDE agree. A little KD captures essentially all the displacement gain; more only over-distills.

### The one subtle result: mixNLL is U-shaped (and fidelity-dependent)

mixNLL is minimized at **λ=0.25** (33.86, *better* than the λ=0 baseline's 37.93), then climbs and at λ=1.0
(40.59) is *worse* than no KD. Why U-shaped while `calib_err` rises monotonically? Because mixNLL conflates
**two** effects: (a) how close the modes sit to the truth (geometry) and (b) how sharp they are (calibration).
At λ=0.25 the modes move much closer to the truth, and that density gain outweighs the over-confidence cost.
Past 0.25 geometry saturates but scales keep shrinking, so the calibration cost dominates and mixNLL rises.

⚠️ **This U-shape does not transfer to full data.** On the full-data runs the sign flips: λ=0.5 mixNLL was
*worse* than λ=0 (+40.9%). The reason is fidelity — on triage the λ=0 baseline is geometrically weak
(undertrained, minFDE 1.68), so KD's geometry gain is large and dominates mixNLL; on full data the λ=0
baseline is well-converged (minFDE 1.16, well-calibrated), so the geometry gain is small and the
over-confidence cost dominates. **Trust the decomposed metrics** (`b_scale`, `calib_err`, coverage) — those
are geometry-agnostic and degrade monotonically with λ regardless of fidelity; mixNLL's λ-shape is
training-dependent.

### Implication for the operating point

On triage, **λ=0.25 dominates λ=0.5 on every axis** — better minFDE, minADE, brier-minFDE, mixNLL, *and*
lower calibration error — consistent across 5 metrics, so not single-run noise. The full-data run used
λ=0.5, which is past this optimum. **Strong candidate next step: a full-data run at λ=0.25**, which should
give both better geometry and less calibration damage than the existing λ=0.5 full run. (Caveat: 15-epoch/25%
triage optima don't always transfer exactly to 64-epoch/full; treat 0.25 as the leading hypothesis, confirm
on full data.)

### Offline mode diversity (pairwise final-displacement spread, full val set)

`measure_mode_diversity.py` (patched to load KD checkpoints via the eval.py prefix-strip loader), run over
all 39,472 val scenes — pairwise Euclidean distance between the 6 modes' final endpoints, per agent:

| stat (m) | kl=0.0 | kl=0.5 |
|---|---:|---:|
| mean | 8.548 | 10.686 |
| std | 9.381 | 13.523 |
| median | 5.213 | 5.233 |
| 10th pct | 0.657 | 0.691 |
| 90th pct | 21.694 | 29.511 |

This **independently corroborates the π-entropy result**: kl=0.5's modes are *more* spread apart (mean
+25%, 90th pct +36%), not collapsed. So neither the mixture weights nor the mode geometry collapse under
KD — the modes fan out toward the teacher's distinct mode locations. The overconfidence is **entirely**
in the per-mode Laplace scales, confirming the "distil means, discard variance" mechanism precisely.

## Teacher gold-standard (calibration target for v2)

`eval.py` on the HiVT-128 teacher (full val), to anchor what "correctly calibrated" looks like:

| (full data) | b_scale | b_scale_all | calib_err | cov@p90 | mixNLL |
|---|---:|---:|---:|---:|---:|
| student kl=0.0 (well-cal baseline) | 0.4129 | 0.4698 | 0.0362 | 0.900 | 26.69 |
| student kl=0.5 (KD) | **0.2678** | 0.2995 | 0.1827 | 0.713 | 37.62 |
| **teacher HiVT-128** | 0.358 | 0.412 | 0.047 | 0.894 | 21.05 |

Two findings that reframe the v2 target:

1. **The teacher is *sharper* than the well-calibrated kl=0 student (b 0.358 < 0.413) yet stays
   well-calibrated** (calib_err 0.047, p90 coverage 0.894 ≈ nominal). Possible only because its modes are
   *accurate* — small residuals justify small scales. Sharpness is over-confidence only when the modes
   aren't accurate enough to earn it.
2. **The kl=0.5 student over-shot *below* the teacher** (b 0.268 < 0.358): KD made the student *more*
   confident than the model it distils from, despite being far less accurate. Pure "distil means, discard
   variance" — nothing holds the scales at a calibrated width.

**Consequence — "match the teacher's b" is NOT the right v2 target.** The teacher is calibrated at b=0.358
*because it is accurate*; the less-accurate student needs a *larger* b than the teacher to be calibrated (the
kl=0 student is calibrated at 0.413). So distilling teacher scales (v2) should move the student's b **up**
from 0.268 toward ~0.413 and cut calib_err — but likely only **partial** recovery, since teacher-level
sharpness is still slightly too tight for the student's own accuracy. **Pre-registered v2 expectation:
b ↑ toward the teacher, calib_err ↓ toward ~0.05 — partial, not necessarily full, recovery.** (mixNLL 21.05
is the teacher's full-distribution ceiling; no emb32 student will reach it.)

## v2 full-data result — the fix works (and the prediction held)

Full-data run: `kd_mode=dist`, λ=0.5, 64 ep (`kd_ckpt/emb32-bs128-lkl0.5-distv2-full/`, ckpt ep63).
Evaluated on full val via `eval.py`:

| Metric | v1 kl=0 (baseline) | v1 kl=0.5 | **v2 kl=0.5 (dist)** | teacher-128 | v2 vs v1-kl0.5 | v2 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| minADE | 0.7365 | 0.6958 | **0.6982** | 0.6611 | +0.3% (tie) | −5.2% |
| minFDE | 1.1574 | 1.0509 | **1.0505** | 0.9692 | ≈tie | −9.2% |
| minMR | 0.1225 | 0.1056 | **0.1063** | 0.0920 | +0.7% (tie) | −13.2% |
| brier_minFDE | 1.8197 | 1.7235 | **1.7240** | — | ≈tie | −5.3% |
| mixNLL | 26.69 | 37.62 | **24.18** | 21.05 | **−35.7%** | **−9.4%** |
| calib_err | 0.0362 | 0.1827 | **0.0275** | 0.047 | **−85.0%** | −24.0% |
| b_scale | 0.4129 | 0.2678 | **0.4126** | 0.358 | +54.1% | −0.1% |
| cov@p90 | 0.900 | 0.713 | **0.909** | 0.894 | — | — |
| reg_loss | −0.246 | −0.188 | **−0.251** | — | better | better |

**v2 matches v1's geometry exactly and eliminates its calibration penalty — KD's full accuracy gain at zero
calibration cost.**

1. **Geometry: v2 ties v1 kl=0.5** (minFDE 1.0505 vs 1.0509; v1 a hair better on minADE/minMR — within noise).
   Both land on HiVT-64 (1.0301), recovering **~86%** of the HiVT-32→HiVT-64 gap at zero inference cost. v2 is
   *not* a geometry win over v1; the gain is from KD generally and v2 preserves it.
2. **Calibration: v2 fully repairs v1's collapse.** calib_err 0.0275 vs v1's 0.1827 (6.6× better), *better than
   the no-KD baseline* (0.0362) and *better than the teacher* (0.047); cov@p90 ≈ nominal; mixNLL 24.18 is
   *below* the baseline (26.69), approaching the teacher (21.05). v1's +40.9% mixNLL regression is erased.
3. **The δ+b_teacher prediction is confirmed.** v2 recovered b_scale to **0.413 — above the teacher's 0.358 and
   essentially equal to the student's own calibrated width** (the no-KD baseline's 0.413), up from v1's
   collapsed 0.268. The student needs a wider scale than the teacher (it's less accurate), and v2 delivers
   exactly that. Mechanism validated at full scale.

> **Updated thesis arc:** v1 mean-target KD buys geometry by spending calibration (overconfident scale
> shrinkage). v2 distribution-matching KD — which distils the teacher's predictive *variance*, not just its
> means — delivers the geometry gain with *no* calibration cost (in fact better calibration than no-KD). The
> diagnosis (scale shrinkage) and the fix (cover the teacher's spread) are matched and both confirmed.

## Metrics to add (the mixNLL story demands measuring the mechanism)

- **Sharpness** — mean predicted Laplace scale `b`. The smoking gun: v1 should visibly shrink it; v2 should not. Cheap to log.
- **Reliability / coverage calibration** — fraction of ground-truth endpoints inside the predicted X% region vs X. A calibration curve *visually proves* over-confidence; the figure reviewers will want.
- **Mode diversity / collapse** — effective number of modes used, mean pairwise mode separation, entropy of `π`. Mean-seeking KD risks collapse.
- **Efficiency axis (essential — it's the point of a small student)** — params, FLOPs, measured inference latency/throughput for HiVT-32/64/128. The narrative "recover X% of accuracy at Y% of cost" needs Y.
- **Multiple seeds** — several deltas here (brier_minADE −1.95%, p_MR +0.62%) are within plausible single-run variance. **3 seeds per condition** with mean±std makes every claim defensible.

## How the numbers were produced

- **Geometry + calibration tables** come from **`eval.py`** (not `plot_lambda_sweep.py`), run per checkpoint:
  `python eval.py --root /home/manya/argoverse --batch_size 128 --ckpt_path <ckpt>`. `eval.py` strips the
  `student.` prefix, loads a plain `HiVT`, and runs `models/hivt.py::validation_step` on the full val set;
  the metrics are read from its stdout (saved to `kd_ckpt/_triage_logs/eval_cal_kl*.log`).
- **`measure_mode_diversity.py`** produced the offline pairwise-spread table.
- **`visualisation_other_tests/plot_lambda_sweep.py`** is the *plotter* for the 5-panel λ figure; it reads a
  `{lambda: {metric: value}}` JSON (assembled from the eval logs) and writes a PNG. Not yet run.
**The λ matrix → loop**: eval.py over the four checkpoints
This recreates the dose-response table (each judged on full val, apples-to-apples triage family):
```bash
declare -A C=(
  [0.0]="kd_ckpt/triage-emb32-lr3e-3-kl0.0-cal/best/HiVTKD-epoch=12-val_minFDE=1.68.ckpt"
  [0.25]="kd_ckpt/triage-emb32-lr3e-3-kl0.25-cal/best/HiVTKD-epoch=14-val_minFDE=1.36.ckpt"
  [0.5]="kd_ckpt/triage-emb32-lr3e-3-kl0.5-cal/best/HiVTKD-epoch=13-val_minFDE=1.38.ckpt"
  [1.0]="kd_ckpt/triage-emb32-lr3e-3-kl1.0-cal/best/HiVTKD-epoch=14-val_minFDE=1.39.ckpt" )
for kl in 0.0 0.25 0.5 1.0; do
  echo "===== λ=$kl ====="
  python eval.py --root /home/manya/argovtaerse --batch_size 128 --ckpt_path "${C[$kl]}" \
    2>&1 | tee kd_ckpt/_triage_logs/eval_cal_kl${kl}.log
done
```
## Metric glossary

All metrics are on the focal agent, full val set. "Oracle" = best of the K=6 modes (lower bound; ignores
which mode the model thought was likely). "p-" / "brier-" = probability-aware (penalise the confidence
assigned to the chosen mode). Lower is better unless stated.

**Geometry (oracle best-of-6)**
- **minADE** — average L2 distance (over the 30 future steps) between the GT and the *closest* of the 6 predicted modes, in metres.
- **minFDE** — L2 distance at the *final* step for the closest mode (endpoint error), in metres.
- **minMR** — miss rate: fraction of agents whose best-mode endpoint is > 2.0 m from GT.

**Probability-aware (depend on the mode weights π)**
- **brier_minADE / brier_minFDE** — the oracle min* plus a Brier penalty `(1 − p_best)²`, where `p_best` is the
  probability the model put on the mode that turned out best. **brier-minFDE is the official Argoverse-1
  ranking metric.** Rewards both good geometry *and* assigning high probability to the good mode.
- **p_minADE / p_minFDE** — min* plus a log penalty `min(−log p_best, −log 0.05)` (the official Argoverse
  probabilistic variant; the −log 0.05 cap bounds the penalty for very low-confidence predictions).
- **p_MR** — miss-rate variant: a hit contributes `(1 − p_best)`, a miss contributes `1.0`. A thresholded,
  quantised metric — noisy; don't over-read small changes.

**Full-distribution likelihood / calibration**
- **mixNLL** — negative log-likelihood of the GT under the *whole* 6-component Laplace mixture
  (`−log Σ_k π_k · Laplace(y | μ_k, b_k)`). Unlike min*/brier (which look only at the best mode), this scores
  the entire predictive distribution, so it punishes both bad geometry *and* mis-sized scales. Conflates the
  two — see the U-shape discussion above.
- **reg_loss** — HiVT's own winner-takes-all Laplace regression NLL (the training objective on the best mode).
  *More negative = better.*

**Diagnostic metrics added in Task #2 (no effect on training; pure measurement)**
- **b_scale** — mean predicted Laplace scale `b` (metres) of the **chosen / min-FDE mode**, averaged over the
  30 steps and 2 coords. The model's stated uncertainty for the trajectory it reports. *Smaller = sharper /
  more confident.* The direct probe of scale shrinkage.
- **b_scale_all** — same, but averaged over **all 6 modes** — the average sharpness of the whole mixture.
- **calib_err** — scalar miscalibration, `mean_p |empirical_coverage(p) − p|` over `p ∈ {0.1,…,0.9}` (an
  ECE-like summary of the reliability table). *Higher = worse calibrated.* 0 = perfectly calibrated.
- **cov_p10 … cov_p90** — empirical coverage: the fraction of GT points falling inside the chosen mode's
  predicted central-`p` Laplace interval (half-width `t = −b·ln(1−p)`). Compared to nominal `p`, this is the
  reliability curve; empirical < nominal ⇒ over-confident (intervals too narrow).
- **pi_entropy** — entropy (nats) of the softmax mode weights `π`, averaged over agents. *Lower ⇒ mode-weight
  collapse* (mass piling on few modes). Max for 6 modes is `ln 6 ≈ 1.792`; the observed ≈1.74–1.76 means the
  weights are already near-uniform, so the slight *rise* under KD = no collapse.


  **brier-minFDE** is the official Argoverse-1 ranking metric (so it's your headline number), and π-entropy's ceiling is ln 6 ≈ 1.792 — your values (~1.74–1.76) sit just under it, which is why "flat/slightly up" means the weights were near-uniform all along and KD doesn't collapse them

# appendix - tables

**parameter/efficiency** (CPU latency: `profile_efficiency.py`, batch_size=1, OMP_NUM_THREADS=4, median of 30 fwd passes; latency is architecture-determined so it needs no trained weights. Peak GPU mem N/A on CPU.)

| model | embed | params | rel. params | CPU ms/scene | speedup |
|-------|------:|-------:|------------:|-------------:|--------:|
| teacher-128 | 128 | 2,559,993 | 100% (1×) | 20.83 | 1.0× |
| student-32 | 32 | 170,073 | 6.6% (15× smaller) | 7.95 | 2.6× |
| student-16 | 16 | 45,929 | 1.8% (55× smaller) | 6.87 | 3.0× |

**Key caveat — latency speedup is far sublinear in param reduction.** student-32 has 15× fewer params but is
only **2.6×** faster; student-16 (3.7× fewer params than student-32) is barely faster (7.95→6.87 ms). At
batch_size=1 on CPU, runtime is dominated by **dim-independent overhead** (graph construction, the fixed
count of temporal/global transformer layers, Python/PyG dispatch), not the embed_dim²-scaling matmuls. So
"55× smaller" is a **memory/param** claim, not a speed claim — do not equate the two in the thesis. The
deployment story is: *student-32 keeps ~86% of the HiVT-32→64 accuracy gap (with v2, fully calibrated) at 6.6%
of the params and ~2.6× faster CPU inference.* (Latency is not batch-size invariant; bs=1 = online/real-time.)

**no KD vs KDv1 (λ=0.5)** — full data, correct `train_fix.h5` ckpts (`triage-…-kl0.0-full` ep54, `…-kl0.5-full` ep58), `eval.py`. Identical to the head-to-head / diagnostic / coverage sections above.

| Metric | λ=0 (No KD) | v1 λ=0.5 (Mean) | Change |
| :--- | :--- | :--- | :--- |
| **minADE** | 0.7365 | 0.6958 | −5.5% ✓ |
| **minFDE** | 1.1574 | 1.0509 | −9.2% ✓ |
| **minMR** | 0.12252 | 0.10557 | −13.8% ✓ |
| **mixNLL** | 26.695 | 37.624 | +40.9% ✗ |
| **calib_err** | 0.0362 | 0.1827 | 5.0× worse ✗ |
| **b_scale** | 0.4129 | 0.2678 | −35.1% (scale shrinkage) |
| **cov_p90** | 0.900 | 0.713 | over-confident |