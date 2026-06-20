# Handoff — KD-for-HiVT thesis, project state (as of 2026-06-20)

A fresh conversation can resume from here. The **scientific report of record is
[docs/kd_emb32_kl_comparison.md](kd_emb32_kl_comparison.md)** (full tables, mechanism, glossary, dose-response).
This file is the *operational* state: what's done, what's running, where things live, gotchas, next steps.

## 1. What this project is

Diploma thesis: **knowledge distillation for HiVT trajectory prediction** on Argoverse 1. Student **HiVT-32**
(embed_dim=32, ~170K params) distilled from a pretrained **HiVT-128** teacher. HiVT predicts a 6-component
Laplace mixture over the future trajectory (each mode = location μ, scale b, weight π).

## 2. The central finding (the thesis contribution)

**Mean-target KD improves geometry but degrades calibration via scale shrinkage, not mode collapse.**

- The v1 KD loss scores the teacher's mode-*means* as point targets and **discards the teacher scales**, so it
  rewards the student for shrinking its Laplace scales → over-confidence. Geometry (minADE/FDE/MR, brier-minFDE)
  improves; full-distribution calibration (mixNLL, coverage) degrades.
- **Confirmed and falsified-tested** with diagnostics: kl=0.5 vs kl=0.0 (full data) shows `b_scale` −31%,
  `calib_err` 5× worse, coverage curve below diagonal — but **π-entropy flat** (no mode collapse) and offline
  pairwise mode-spread actually *wider*. So the defect is purely within-mode **scale**.
- **λ dose-response** (triage, 4 points): `b_scale`↓ and `calib_err`↑ **monotonically** with λ; π-entropy flat
  at every dose; **geometry peaks at λ≈0.25** (not monotone); mixNLL is U-shaped (min at 0.25 on triage, but
  fidelity-dependent — see the doc; don't transfer its sign to full data).
- **Proposed fix = v2** (`HiVTKDLossDist`): distribution-matching MC cross-entropy that uses the teacher
  scales, so the student must cover the teacher's spread (can't shrink b). Implemented, not yet trained.

## 3. Code state (all on branch `fix/kd-permutation-invariant-loss`, uncommitted)

| File | State | What |
|---|---|---|
| [KD/kd_loss.py](../KD/kd_loss.py) | modified | `HiVTKDLoss` (v1, mean-target, **unchanged default**) + `HiVTKDLossDist` (v2, distribution-matching) + `make_kd_loss(kd_mode)` factory. v2 verified to reduce to v1 as teacher scale→0. |
| [KD/hivt_kd.py](../KD/hivt_kd.py) | modified | wires `--kd_mode {mean,dist}` (default `mean`) and `--kd_n_samples` into the factory; call site unchanged. |
| [models/hivt.py](../models/hivt.py) | modified | Task-2 diagnostics added to `__init__` (`bScale`,`bScaleAll`,`piEntropy`,`coverage`) + `validation_step` logs `val_b_scale`,`val_b_scale_all`,`val_calib_err`,`val_cov_p10..p90`,`val_pi_entropy`. |
| [metrics/calibration.py](../metrics/calibration.py) | new | `LaplaceCoverage` + `ScalarMean` + `log_laplace_coverage`. Coverage uses half-width `t=−b·ln(1−p)`. |
| [metrics/__init__.py](../metrics/__init__.py) | modified | exports the calibration metrics. |
| [KD/eval_lambda_sweep.py](../KD/eval_lambda_sweep.py) | new | runs eval over `--entry λ=ckpt` pairs, dumps `{lambda:{metric:value}}` JSON. **num_workers defaults to 0** (see gotcha #3). |
| [visualisation_other_tests/plot_lambda_sweep.py](../visualisation_other_tests/plot_lambda_sweep.py) | new | 5-panel dose-response plotter; reads the JSON. |
| [visualisation_other_tests/measure_mode_diversity.py](../visualisation_other_tests/measure_mode_diversity.py) | modified | patched to load KD ckpts (prefix-strip). |
| [eval.py](../eval.py) | unchanged | strips `student.` prefix → plain HiVT → full-val validate. The metric numbers all come from here. |

Nothing is committed yet. `Task #2` (diagnostics) is **done**; its handoff [docs/handoff_task2_metrics.md](handoff_task2_metrics.md) is now historical.

## 4. Environment & key paths

- **Env:** `source /home/manya/miniconda3/etc/profile.d/conda.sh && conda activate hivt_new`
- **Dataset root:** `/home/manya/argoverse`
- **Teacher cache (verified-good):** `teacher_outputs/train_fix.h5` — use this, NOT `train.h5` (see [[teacher-cache-fixed]]). Stores teacher loc+scale+pi.
- **Full-data checkpoints** (the headline numbers):
  - kl=0.0: `kd_ckpt/emb32-bs128-lkl0.0/best/HiVTKD-epoch=46-val_minFDE=1.23.ckpt`
  - kl=0.5: `kd_ckpt/emb32-bs128-lkl0.5/best/HiVTKD-epoch=62-val_minFDE=1.12.ckpt`
- **Triage λ-sweep checkpoints** (`-cal` family, 25% data, 15 ep, lr=3e-3 — apples-to-apples):
  - `kd_ckpt/triage-emb32-lr3e-3-kl{0.0,0.25,0.5,1.0}-cal/best/` (pick lowest `val_minFDE`).
- **Author baselines:** `checkpoints/HiVT-{64,128}/...`; standalone `HiVT-32/gxhl2ug9/...epoch=63...`.
- **Recipe:** bs=128, lr=3e-3, 64 ep full / 15 ep triage (see [[kd-recipe-emb32]]).

## 5. Gotchas (all hit this session)

1. **GPU instability (WSL2 + RTX 5060).** Runs can die silently mid-training/val with a throughput collapse
   first and NO Python traceback. On WSL2 the signature is `misc dxg: dxgk: ... Ioctl failed: -22` in
   `dmesg -T` (NOT NVRM/Xid/WHEA — those are Windows-side, Event Viewer → `nvlddmkm` TDR). It's hardware/driver,
   not code. Mitigation: run in **tmux**, with an auto-restart watchdog + auto-resume from `last.ckpt`:
   ```bash
   until <train-cmd>; do echo "resuming in 30s"; sleep 30; done
   ```
   See [[gpu-pcie-instability]].
2. **Geometry vs calibration metrics tell different stories** — don't expect monotonicity in everything;
   mixNLL conflates both and is fidelity-dependent. Trust the decomposed `b_scale`/`calib_err`/coverage.
3. **DataLoader fork-after-CUDA abort.** Running multiple `trainer.validate()` calls in ONE process with
   `num_workers>0` aborts workers (SIGABRT, "killed by signal: Aborted") because forking after CUDA init breaks
   the worker CUDA context. `eval_lambda_sweep.py` therefore defaults to `num_workers=0`. Plain `eval.py` (one
   validate per process) is fine with workers=8.

## 6. Immediate next step (in progress)

**Generate the λ dose-response figure** — driver was just fixed (gotcha #3). Run:
```bash
python -m KD.eval_lambda_sweep --root /home/manya/argoverse --batch_size 128 \
  --out docs/figures/lambda_sweep.json \
  --entry 0.0=kd_ckpt/triage-emb32-lr3e-3-kl0.0-cal/best/HiVTKD-epoch=12-val_minFDE=1.68.ckpt \
  --entry 0.25=kd_ckpt/triage-emb32-lr3e-3-kl0.25-cal/best/HiVTKD-epoch=14-val_minFDE=1.36.ckpt \
  --entry 0.5=kd_ckpt/triage-emb32-lr3e-3-kl0.5-cal/best/HiVTKD-epoch=13-val_minFDE=1.38.ckpt \
  --entry 1.0=kd_ckpt/triage-emb32-lr3e-3-kl1.0-cal/best/HiVTKD-epoch=14-val_minFDE=1.39.ckpt
python visualisation_other_tests/plot_lambda_sweep.py \
  --data docs/figures/lambda_sweep.json --out docs/figures/kd_lambda_sweep.png
```
(`docs/figures/` does not exist yet; the plotter creates it. JSON/PNG not yet produced.)

## 7. Next experiments (priority order)

1. **Full-data λ=0.25 run** — HIGHEST VALUE. On triage, λ=0.25 dominated λ=0.5 on every metric; the existing
   full run used 0.5 (past the optimum). Run full (64 ep, full data, `--kd_mode mean`, λ=0.25) in tmux+watchdog,
   then `eval.py` it and compare vs the λ=0.5 full-data row — does 0.25 win on **both** minFDE and mixNLL (maybe
   even erase the +20.6% mixNLL regression)? Launcher: `KLS="0.25" SUFFIX="" SUB=full EPOCHS=64 bash KD/run_kd_sweep.sh`
   (⚠️ verify `run_kd_sweep.sh` honors `SUB=full`/`EPOCHS` overrides before trusting it).
2. **Teacher gold-standard** — `eval.py` the HiVT-128 teacher to record its `b_scale`/coverage; that's the
   calibration target v2 should move the student toward.
3. **v2 validation** (`--kd_mode dist`) — smoke → triage *its own* λ (v2 loss magnitude ≠ v1) → one full run.
   Pre-registered success test: keeps `b_scale`/coverage near the λ=0 baseline while retaining the geometry gain.
4. **Efficiency axis** — params + measured latency for HiVT-32/64/128 (the "recover X% accuracy at Y% cost"
   framing). Independent, no training.
5. **Multiple seeds** — only for the final headline geometry comparison; calibration deltas are huge and don't
   need seeds.
6. **Phase D — emb16 transfer** — re-triage LR, then full runs both arms; tests whether KD benefit grows as the
   student shrinks.

## 8. Suggested kickoff prompt for the new conversation

> Read docs/handoff_kd_thesis_state.md and docs/kd_emb32_kl_comparison.md. I'm continuing the KD-for-HiVT
> thesis. [Then state your task, e.g.: "Run the λ-sweep figure (section 6)" or "Launch the full-data λ=0.25
> run (section 7.1) in tmux with the watchdog" or "Implement the efficiency axis (section 7.4)".]