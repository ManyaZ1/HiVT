# HiVT - 32 

## HiVT-32 attention heads 2 vs attention heads 8 (no KD) kl=0.0

Both runs: HiVT-32, no KD (`lambda_kl=0.0`), `bs=128`, `lr=3e-3`, full Argoverse, 64 epochs, same KD harness. The **only** difference is `--num_heads` (2 vs the default 8). Δ = (2 heads) − (8 heads); "Better?" = is the 2-head model better than 8-head on that metric. Both columns from `eval.py` on the best-`val_minFDE` checkpoint (8-head = `triage-emb32-lr3e-3-kl0.0-full`; 2-head = `triage-emb32-lr3e-3-kl0.0-nh2-full` ep63).

| Metric | attention heads 2 | attention heads 8 | Δ (abs) | Δ (%) | Better? |
|---|---:|---:|---:|---:|:--:|
| **val_minADE** | 0.7381 | 0.7365 | +0.0016 | +0.22% | ≈ |
| **val_minFDE** | 1.16219 | 1.1574 | +0.0048 | +0.41% | ≈ |
| **val_minMR** | 0.12424 | 0.12252 | +0.00172 | +1.40% | ≈ |
| val_brier_minADE | 1.3962 | 1.3989 | −0.0027 | −0.19% | ≈ |
| val_brier_minFDE | 1.8202 | 1.8197 | +0.0005 | +0.03% | ≈ |
| val_p_minFDE | 2.8544 | 2.8635 | −0.0091 | −0.32% | ≈ |
| val_p_minADE | 2.4303 | 2.4426 | −0.0123 | −0.50% | ≈ |
| val_p_MR | 0.83001 | 0.83222 | −0.00221 | −0.27% | ≈ |
| **val_mixNLL** | 26.891 | 26.695 | +0.196 | +0.73% | ≈ |
| **val_calib_err** | 0.03156 | 0.0362 | −0.00464 | **−12.8%** | ✅ |
| val_b_scale | 0.4162 | 0.4129 | +0.0033 | +0.80% | ≈ |
| val_reg_loss | −0.23872 | −0.2459 | +0.0072 | (less neg.) | ≈ |

Lower is better for every row except `reg_loss` (more negative = better); for `b_scale`, both sit at the well-calibrated width (~0.41), so closer is a tie, not a win.

### Reading it

- **Geometry is a dead heat.** minADE, minFDE, minMR, and both brier variants all move <1.5% and the differences flip sign — this is single-run noise, not a real gap. Dropping from 8 heads to 2 (head dim 4→16) costs essentially **nothing** on displacement at emb32. With 32 channels there isn't enough width for 8 heads to express more than 2.
- **Calibration is the only metric the user flagged, and it differs.** 2-head `calib_err` is 0.0316 vs 8-head 0.0362 (−12.8%), with `b_scale` essentially equal (0.416 vs 0.413) and `mixNLL` within 0.7%. So the 2-head model is, if anything, *marginally better calibrated* — but both are firmly in the well-calibrated regime (cf. the KD kl=0.5 collapse to calib_err 0.18 / b_scale 0.27). This ~13% gap is plausibly still single-run variance; it is **not** the order-of-magnitude effect KD produces. To claim it as real you'd need multiple seeds.

**Bottom line:** at emb32, 2 attention heads matches 8 on geometry and calibration alike — head count is not a meaningful lever here, so the fair KD baseline can use either without affecting the comparison.

