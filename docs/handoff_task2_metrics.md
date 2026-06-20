# Handoff — Task #2: calibration / diagnostic metrics

**Goal:** add three measurements that quantify the KD calibration mechanism documented in
[docs/kd_emb32_kl_comparison.md](kd_emb32_kl_comparison.md):
1. **Sharpness** — mean predicted Laplace scale `b`.
2. **Reliability / coverage** — empirical vs nominal coverage of the predicted Laplace, + a scalar miscalibration.
3. **Mode diversity** — pairwise mode spread + an in-training entropy proxy.

These gate every later experiment (λ-sweep, v2, seeds): without them the central thesis claim
("v1 mean-target KD makes the student overconfident") is unprovable. Pure engineering, **no GPU training** —
just code + re-eval of two existing checkpoints.

## Context you need

- **Why mixNLL got worse (the claim these metrics must test):** the v1 KD loss
  ([KD/kd_loss.py](../KD/kd_loss.py), `HiVTKDLoss`) scores teacher mode-*means* as point targets and discards
  teacher scales, so it rewards the student for shrinking its Laplace scales (over-confidence). Prediction:
  the kl=0.5 checkpoint should show **smaller mean `b`** and **worse coverage** than kl=0.0. If the new metrics
  show that, they work. (v2 `HiVTKDLossDist` is the planned fix; not part of this task.)
- **Env:** `conda activate hivt_new`. Dataset root: `/home/manya/argoverse`.
- **Two checkpoints to validate against** (emb32, full data):
  - kl=0.0: `/home/manya/HiVT/kd_ckpt/emb32-bs128-lkl0.0/best/HiVTKD-epoch=46-val_minFDE=1.23.ckpt`
  - kl=0.5: `/home/manya/HiVT/kd_ckpt/emb32-bs128-lkl0.5/best/HiVTKD-epoch=62-val_minFDE=1.12.ckpt`

## CRITICAL integration fact — where metrics must go

`eval.py` (eval.py:51-71) **strips the `student.` prefix** from KD checkpoints and loads the weights into a
**plain `HiVT`**, then runs [models/hivt.py](../models/hivt.py) `validation_step`. The KD wrapper's own
`validation_step` only runs during training. Therefore:

- **Register the new metrics in `HiVT.__init__`** (models/hivt.py:91-100) so they exist on the plain model.
- **Add `.update()` + `self.log()` in `models/hivt.py::validation_step`** (models/hivt.py:160-193) — this is
  what `eval.py` exercises on the existing checkpoints.
- **Mirror the same calls in `KD/hivt_kd.py::validation_step`** (KD/hivt_kd.py:205-239) so they log to W&B
  during training. The KD wrapper already delegates metrics to `self.student.<metric>`, so reuse the same
  metric objects registered on `HiVT` — e.g. `self.student.coverage.update(...)`.

Follow the existing torchmetrics pattern: metrics are `torchmetrics.Metric` subclasses (see
[metrics/prob.py](../metrics/prob.py) for the template — `add_state('sum'...)`, `add_state('count'...)`,
`update`, `compute`), exported from [metrics/__init__.py](../metrics/__init__.py), constructed in
`HiVT.__init__`, and logged with `self.log(..., on_epoch=True, batch_size=...)`. Don't compute raw means
inline — use a metric (or `torchmetrics.MeanMetric`) so epoch aggregation across batches is correct.

## Tensor facts

- Decoder output `y_hat` is `[F, N, H, 4]`: channels `0:2` = loc (μ), channels `2:4` = scale (b).
- Scale is already positive: decoder applies `elu(x)+1+min_scale` (models/decoder.py:139-140), `min_scale=1e-3`.
- In `validation_step`, `y_hat_agent = y_hat[:, agent_index, :, :2]` currently **drops the scale channels** —
  for sharpness/coverage you must also grab `y_hat[:, agent_index, :, 2:]` and select the same `best_mode_agent`.
- `best_mode_agent` (models/hivt.py:172) indexes the per-agent min-FDE mode; reuse it to pick the chosen
  mode's μ and b: `b_best = y_hat[:, agent_index, :, 2:][best_mode_agent, torch.arange(num_graphs)]` → `[B,H,2]`.

## The three metrics

### 1. Sharpness `val_b_scale` (~20 min)
Mean of the chosen mode's `b` over (agent, horizon, coord). Optionally also `val_b_scale_all` over all modes.
Use a `MeanMetric` (or a tiny Metric). Expectation: kl=0.5 `b` < kl=0.0 `b`.

### 2. Coverage / reliability `val_cov_*` + `val_calib_err` (~1.5h, the real work)
For a Laplace(μ, b), the central interval with nominal coverage `p` has half-width `t = -b * ln(1-p)`
(from CDF `P(|X-μ| ≤ t) = 1 - exp(-t/b)`). Per coordinate, per timestep, on the **chosen mode**:
- For nominal levels `p ∈ {0.1, 0.2, ..., 0.9}`, empirical coverage = fraction of GT points with `|y-μ| ≤ t(p)`.
- Log each `val_cov_p{XX}` (the reliability curve) and a scalar `val_calib_err = mean_p |empirical(p) - p|` (ECE-like).
Implement as a new metric class, e.g. `metrics/calibration.py::LaplaceCoverage` accumulating per-level hit counts
+ total count. Expectation: kl=0.5 is **under-covered** (empirical < nominal → over-confident).
GT for the agent is `y_agent = data.y[agent_index]` (`[B,H,2]`); reg_mask handles padding.

### 3. Mode diversity (~40 min)
- **In-training proxy:** add `val_pi_entropy` = mean entropy of `softmax(pi[agent_index])` to both validation
  steps (cheap, captures mode collapse). Lower entropy ⇒ collapse.
- **Offline figures:** [visualisation_other_tests/measure_mode_diversity.py](../visualisation_other_tests/measure_mode_diversity.py)
  already computes pairwise final-displacement spread. ⚠️ It calls `HiVT.load_from_checkpoint` directly, which
  **fails on KD checkpoints** (the `student.` prefix). Reuse the prefix-strip loader from eval.py:57-71 so it
  accepts KD ckpts. Then run it on both checkpoints for the diversity histograms.

## Definition of done

1. New metrics registered in `HiVT.__init__`, logged in **both** `validation_step`s.
2. `python eval.py --root /home/manya/argoverse --batch_size 128 --ckpt_path <ckpt>` prints the new metrics for
   **both** checkpoints (commands above).
3. **Falsifiable check:** report whether kl=0.5 shows smaller `b`, lower coverage / higher `val_calib_err`,
   and lower `pi_entropy` than kl=0.0 — i.e. whether the metrics confirm the predicted over-confidence.
   Drop the numbers into a short table in [docs/kd_emb32_kl_comparison.md](kd_emb32_kl_comparison.md).
4. `measure_mode_diversity.py` runs on KD checkpoints; save the diversity stats for both.

## Guardrails
- Do **not** modify the KD loss, the training loop, or any existing metric — only add.
- Keep `eval.py`'s plain-HiVT load path working (don't make HiVT require KD-only args).
- The new metrics must be valid for the non-KD baselines too (HiVT-32/64/128), so don't assume teacher data.