
# HiVT params - embed dim
| model | embed | params | vs teacher| val_minFDE |val_minADE | val_minMR |
|-------|-------|--------|-----------|------------|------------|-----------|
teacher-128| 128| 2,559,993| 100% (1×)| 0.9692| 0.7365| 0.12252|
model-64| 64| 653,369| 25.5% (4× smaller)| **1.0301**| 0.7381| 0.12424|
model-32| 32| 170,073| 6.6% (15× smaller)| **1.1574**| 0.7670| 0.12895|
student-32 v2| 32| 170,073| 6.6% (15× smaller)|**1.0505**|0.69824 |0.10630 |
model-16 nh2| 16| 45,929| 1.8% (55× smaller)|**1.6052** |0.8858| 0.1988|
student-16| 16| 45,929| 1.8% (55× smaller)| |

student-32 v2 is 26% of model-64 (3.8 times smaller) and 6.6% of teacher-128 (15 times smaller). It loses on 0.1273m of accuracy compared to the HiVT-64. It recovers 84% of the HiVT-32→HiVT-64 gap at zero inference cost, (1.1574-1.0505)/(1.1574-1.0301)= 84% while being 26% of the params.

teacher-128 vs student 32: it recovers (1.0505-1.1574)/(0.9692-1.1574)= 57% of the HiVT-32→HiVT-128 gap at zero inference cost, while being 6.6% of the params.

*note*: ≈86% vs the original author-style bs32 HiVT-32(batch size 32, 64 epochs lr=5e-4)gap.
# efficiency

The hardware motivation is "smaller model → cheaper inference." Two cost axes: **parameters/memory** (linear in model size, the clean win) and **latency** (sublinear, regime-dependent).

## Parameters (architecture-determined, exact — `KD/profile_efficiency.py`)

| model | embed | params | vs teacher |
|---|---:|---:|---:|
| teacher-128 | 128 | 2,559,993 | 100% (1×) |
| model-64 | 64 | 653,369 | 25.5% (4× smaller) |
| **student-32** | 32 | 170,073 | **6.6% (15× smaller)** |
| **student-16** | 16 | 45,929 | **1.8% (55× smaller)** |

## Latency — CPU, the batch-size dependence

Per-scene forward latency (ms), `profile_efficiency.py` on real val batches, `OMP_NUM_THREADS=4`, median of 30 passes. "speedup" = teacher-128 ÷ student.

| batch_size | teacher-128 | student-32 | student-16 | speedup T/S32 | speedup T/S16 |
|---|---:|---:|---:|---:|---:|
| 1 | 20.82 | 6.86 | 6.11 | 3.03× | 3.41× |
| 8 | 16.07 | 2.95 | 2.14 | **5.45×** | 7.53× |
| 32 | 27.05 | 4.68 | 2.35 | 5.78× | **11.52×** |
| 64 | 27.85 | 5.85 | 2.87 | 4.76× | 9.71× |

### Interpretation

- **Latency is far sublinear in param reduction, and batch-size-dependent.** At `bs=1` (online, one scene at a time) student-32 is only **3.0×** faster despite **15× fewer params**, because a fixed ~7 ms floor — graph construction, the fixed count of transformer layers, Python/PyG dispatch, memory-bound norm/scatter ops — is *independent of `embed_dim`* and dominates the tiny `embed_dim²` matmuls. So "55× smaller" is a **memory/param** claim, not a speed claim; do not equate the two.
- **Batching amortises that floor and widens the speedup.** Per-scene `latency ≈ floor/B + compute`; as `B` grows the floor divides away and the part that *does* scale with model width is exposed. The student-32 advantage roughly doubles (3.0×→5.5×) and student-16's roughly triples (3.4×→**11.5×**), because the smallest model was most buried under the floor.
- **On CPU the gain is capped by core count — per-scene latency is U-shaped, not monotone.** It bottoms around `bs=8` (student-32 2.95 ms, student-16 2.14 ms) then *rises* again: with only 4 threads, past `bs≈8` the matmuls saturate CPU parallelism so larger batches contend (cache/memory pressure) instead of amortising. The teacher saturates first (it is compute-heavy), which is why T/S16 keeps climbing to `bs=32` while T/S32 has already turned over. A many-core CPU or a GPU (abundant parallelism, no early saturation) should push the ratio further toward the 15× param ratio.

> **Deployment story:** *student-32 keeps ~86% of the HiVT-32→HiVT-64 accuracy gap (v2, fully calibrated) at 6.6% of the params, ~3× faster CPU inference online (`bs=1`) and ~5.5× under modest batching.* Param/memory savings are linear and unconditional; the latency win depends on batch size and hardware parallelism.

> ⏳ **GPU latency + peak-memory pending** — measured during a contended training run gives unusable numbers; deferred until the GPU is free. The GPU sweep is also what extends the ratio past CPU's 4-thread saturation point. Param counts are the reliable cost numbers for now.


# training comparison
## best hivt32: triage-emb32-lr3e-3-kl0.0-full 
```bash
 python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/kd_ckpt/triage-emb32-lr3e-3-kl0.0-full/best/HiVTKD-epoch=63-val_minFDE=1.16.ckpt
{'val_minADE': 0.7361068725585938,
 'val_minFDE': 1.1568084955215454,
 'val_minMR': 0.121706523001194,
 'val_reg_loss': -0.24731363356113434}
--------------------------------------------------------------------------------
 ```
 *note:* for attention heads 2 minFDE=1.16219, minADE=0.7381, minMR=0.12424, reg_loss=-0.23872
## HIVT-32 STUDENT KD KL=0.5 Loss v2
```bash 
python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/kd_ckpt/emb32-bs128-lkl0.5-distv2-full/best/HiVTKD-epoch=63-val_minFDE=1.05.ckpt
{'val_minADE': 0.6982370018959045,
 'val_minFDE': 1.050521731376648,
 'val_minMR': 0.10630320012569427,
 'val_reg_loss': -0.25004932284355164}
--------------------------------------------------------------------------------
```

## HiVT-64 trained for 64 epochs (pretrained ckpt provided by HiVT authors)
```bash
python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt
--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{'val_minADE': 0.6868845820426941,
 'val_minFDE': 1.0300804376602173,
 'val_minMR': 0.10262971371412277,
 'val_reg_loss': -0.30333220958709717}
--------------------------------------------------------------------------------
```
## HiVT-128 trained for 128 epochs (pretrained ckpt provided by HiVT authors)
```bash
python eval.py --root /home/manya/argoverse --batch_size 32 --ckpt_path /home/manya/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt
--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{'val_minADE': 0.6611008048057556,
 'val_minFDE': 0.9691530466079712,
 'val_minMR': 0.09203992784023285,
 'val_reg_loss': -0.33467531204223633}
--------------------------------------------------------------------------------
```

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

- **Geometry shows no improvement.** minADE, minFDE, minMR, and both brier variants all move <1.5% and the differences flip sign — this is single-run noise, not a real gap. Dropping from 8 heads to 2 (head dim 4→16) costs essentially **nothing** on displacement at emb32. With 32 channels there isn't enough width for 8 heads to express more than 2.
- **Calibration is the only metric the user flagged, and it differs.** 2-head `calib_err` is 0.0316 vs 8-head 0.0362 (−12.8%), with `b_scale` essentially equal (0.416 vs 0.413) and `mixNLL` within 0.7%. So the 2-head model is, if anything, *marginally better calibrated* — but both are firmly in the well-calibrated regime (cf. the KD kl=0.5 collapse to calib_err 0.18 / b_scale 0.27). This ~13% gap is plausibly still single-run variance; it is **not** the order-of-magnitude effect KD produces. To claim it as real you'd need multiple seeds.

**Bottom line:** at emb32, 2 attention heads matches 8 on geometry and calibration alike — head count is not a meaningful lever here, so the fair KD baseline can use either without affecting the comparison.

## triage-lr-emb32
documented in wandb, lr sweep for emb32, 25% of data, 64 ep, no KD, bs128. The best lr=3e-3 (ep63) is used. 

## HiVT-32 KD

## emb32-lr3e-3-kl 1.0 vs 0.5
why was this done...
anyways 0.5 won
later there was 0.25 vs 0.5 vs 0.75 vs 1.0 and 0.25 was best. but 0.5 is used for the main KD runs because it is the most common value in the literature and is a good compromise between geometry and calibration.

### Head-to-head (the clean A/B — same recipe, only λ_kl changes)

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

**Reading it.** v1 KD **trades calibration for accuracy.** Geometry improves substantially and consistently (minFDE −9.2%, minMR −13.8%, minADE −5.5%, both brier variants better), but full-distribution calibration collapses (mixNLL +40.9%, calib_err 5×, b_scale −35%). The mechanism is **Laplace-scale shrinkage, not mode collapse**: maximising density at the teacher's mode *means* (v1 discards the teacher's scales) pushes the student to be sharper than the teacher ever was, so the predicted intervals get too narrow. π-entropy is unchanged and the modes actually fan out *wider*, confirming the over-confidence lives entirely in the per-mode scales.

> **Thesis statement:** *Mean-target mixture distillation improves geometric accuracy and the official Argoverse ranking metric (brier-minFDE) but degrades full-distribution calibration (mixNLL), because it transfers the teacher's mode locations and weights while discarding its predictive variance, producing overconfident student mixtures.*

### v2 fix — distribution-matching distillation (full data)

v2 (`--kd_mode dist`) draws Monte-Carlo Laplace samples from each teacher mode (using the teacher *scales* `b_T`), so the student must cover the teacher's spread — collapsing its scales no longer minimises the loss. It reduces exactly to v1 in the zero-variance limit. Full-data run: `kd_mode=dist`, λ=0.5, 64 ep (ckpt ep63), `eval.py`.

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

**Reading it.** **v2 matches v1's geometry exactly and eliminates its calibration penalty.** Geometry ties v1 (minFDE 1.0505 vs 1.0509); calibration is fully repaired — calib_err 0.0275 (6.6× better than v1, *better than the no-KD baseline and the teacher*), cov@p90 ≈ nominal, mixNLL 24.18 below the baseline. v2 recovered b_scale to 0.413 — the student's own calibrated width — exactly as predicted (a less-accurate student needs a *wider* scale than the teacher to stay calibrated). KD's full accuracy gain at zero calibration cost.

### Sanity check vs the real baselines

| Model | Setup | minADE | minFDE | minMR |
|---|---|---:|---:|---:|
| HiVT-32 (original) | bs32, 64 ep, no KD | 0.7446 | 1.1782 | 0.12505 |
| HiVT-32 KD kl=0.0 | bs128, KD harness | 0.7365 | 1.1574 | 0.12252 |
| **HiVT-32 KD kl=0.5 (v1)** | bs128, mean-target | **0.6958** | **1.0509** | **0.10557** |
| **HiVT-32 KD kl=0.5 (v2 dist)** | bs128, dist-match | 0.6982 | 1.0505 | 0.10630 |
| HiVT-64 | author ckpt | 0.6869 | 1.0301 | 0.10263 |
| HiVT-128 (teacher) | author ckpt | 0.6611 | 0.9692 | 0.09204 |

**Reading it.** KD kl=0.5 beats the original standalone HiVT-32 by −10.8% minFDE and lands essentially **on top of HiVT-64** (1.0301) — a model with 2× the width — recovering **~86% of the HiVT-32→HiVT-64 gap at zero inference cost**, still ~+8.4% above the HiVT-128 teacher. The kl=0.0 in-harness baseline (1.1574) is slightly *better* than the original HiVT-32 (1.1782), so the KD gain is measured against an honest, not weakened, baseline.

# HiVT 16
| Metric | kl=0.0 | kl=0.5 v2 |kl=0.5 v1 | Δ (%) | Better? |
|---|---:|---:|---:|---:|:--:|
|val_minADE|: 
|val_minFDE|: 
|val_minMR|: 

**NOTE** with LR 3e-3 the model doesn't train well, so I use LR 1e-3 (unlike HiVT-32 which needs 3e-3). The LR sweep below shows that 1e-3 is the best LR for HiVT-16, and the results above are from that LR.
```
full run for 3e-3
 'val_cov_p90': 0.93057781457901,
 'val_minADE': 1.6556787490844727,
 'val_minFDE': 3.2196803092956543,
 'val_minMR': 0.45457538962364197,
 'val_mixNLL': 61.78700256347656,
```
 ## HiVT-16 lr sweep (25% OF DATA)
 |Metric	|lr=5e-4	|lr=1e-3 ✅	|lr=2e-3	|lr=3e-3|
 |----------|----------|----------|----------|----------|
|val_minADE	|3.6572	|1.5074	|1.7749	|1.7232|
|val_minFDE	|9.4680	|2.9142	|3.4486	|3.3241|
|val_minMR	|0.7684	|0.3186	|0.4868	|0.4748|
|val_brier_minADE	|4.1865	|2.1428	|2.4201	|2.3846|
|val_brier_minFDE	|9.9974	|3.5496	|4.0938	|3.9855|
|val_calib_err	|0.0860	|0.0284	|0.0729	|0.0717|
|val_b_scale	|1.7039	|0.7845	|0.9594	|0.9300|
|val_mixNLL	|92.76	|53.20	|69.36	|67.55|
|val_reg_loss	|0.4554	|0.1286	|0.2304	|0.2147|
|val_pi_entropy	|1.5510	|1.7078	|1.7271	|1.7429|

**Even with lr=1e-3, the model is still underfitting** (val_minADE 1.69 at epoch 9 while the curve has flattened).

## HiVT-16 attention heads sweep (25% OF DATA)
|Metric	|nh=1	|nh=2 ✅	|nh=4    |nh=8|
|----------|----------|----------|----------|----------|

## HiVT-16 KD kl sweep (25% OF DATA)

Student HiVT-16 (`num_heads=2`, head_dim=8), `lr=1e-3`, teacher = HiVT-128 (`train_fix.h5`), **v2 distribution-matching** (`--kd_mode dist --kd_n_samples 8`). Proxy: 25% data, 15 epochs, bf16. `kl=0.0` is the in-batch no-KD anchor (full-data baseline = 1.605 minFDE). Lower is better for every row except `b_scale` (closer to the calibrated width ~0.73 = better). Best geometry per row in **bold**.

| Metric | kl=0.0 | kl=0.5 | **kl=1.0** ✅ | kl=2.0 |
|---|---:|---:|---:|---:|
| val_minADE | 1.3628 | 1.2701 | **1.1770** | 1.2400 |
| val_minFDE | 2.6471 | 2.4878 | **2.2125** | 2.3655 |
| val_minMR | 0.2995 | 0.3024 | **0.2571** | 0.2632 |
| val_brier_minADE | 2.0065 | 1.9259 | **1.8445** | 1.9160 |
| val_brier_minFDE | 3.2908 | 3.1436 | **2.8800** | 3.0416 |
| val_calib_err | 0.0195 | 0.0259 | 0.0244 | 0.0291 |
| val_b_scale | 0.7334 | 0.6750 | 0.6958 | 0.6820 |
| val_mixNLL | 51.30 | 45.62 | 44.88 | **43.75** |
| val_reg_loss | 0.1254 | 0.0891 | 0.1097 | 0.1551 |

| Δ% vs kl=0.0 | kl=0.5 | **kl=1.0** | kl=2.0 |
|---|---:|---:|---:|
| val_minADE | −6.8% | **−13.6%** | −9.0% |
| val_minFDE | −6.0% | **−16.4%** | −10.6% |
| val_minMR | +1.0% | **−14.1%** | −12.1% |

**Reading it.** KD helps HiVT-16, and the benefit is **monotonic up to kl=1.0** then turns: geometry bottoms at kl=1.0 (minFDE −16.4%, minADE −13.6%, minMR −14.1% vs no-KD), and kl=2.0 regresses on every displacement metric — so **kl=1.0 is the knee.** Crucially, **v2 keeps calibration intact**: mixNLL *improves* with kl (51.3 → 43.8), `b_scale` stays in the healthy ~0.68–0.73 band (no over-confidence collapse), and `calib_err` rises only to 0.024–0.029 (vs emb32-v1's collapse to 0.18). So the geometry win comes essentially free.

**vs emb32.** The relative KD gain is **larger for the smaller student** — emb16 kl=1.0 gives −16.4% minFDE where emb32 kl=0.5 gave −9.2% — consistent with the hypothesis that KD helps more as the student shrinks. (Note: emb16's optimum sits at kl=1.0 vs emb32's 0.5.)

> ⏳ **Proxy ranking only — full-data confirmation of kl=1.0 (64 ep) vs the kl=0.0 baseline (1.605) is the pending deliverable.** Proxy ranks, full data decides (see the lr round, where proxy ordering didn't survive full training). Calibration metrics also move late in training, so treat the proxy calib numbers as optimistic.

## HiVT-16 KD V2