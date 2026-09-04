# Authoritative checkpoint map — HiVT-32 arms of the DriveX/ICRA paper

**Why this file exists.** Choosing a checkpoint by directory name has produced wrong
numbers at least twice in this project (see [[eval-ground-truth]]). Several directories
have plausible-looking names, several checkpoints share a `val_minFDE` in their
filename, and **the filename metric is not `eval.py`'s metric**. This file records
which checkpoint actually reproduces each row of Table II, verified by running
`eval.py` on the full validation set.

**Ground truth is `eval.py`, nothing else.** Not `final_report.MD`, not the filename,
not a re-implementation of the metric. Every number below is a full-val `eval.py` run:

```bash
conda activate hivt_new
python eval.py --root /home/manya/argoverse --ckpt_path <ckpt> \
    --batch_size 32 --num_workers 8
```

---

## The three HiVT-32 arms — VERIFIED, exact on all seven metrics

| Arm | Checkpoint |
|---|---|
| **no-KD** (from-scratch) | `kd_ckpt/triage-emb32-lr3e-3-kl0.0-full/best/last.ckpt` |
| **v1** (mean-target, λ=0.5) | `kd_ckpt/triage-emb32-lr3e-3-kl0.5-full/best/last.ckpt` |
| **v2** (distribution matching, λ=0.5) | `kd_ckpt/emb32-bs128-lkl0.5-distv2-full/best/last.ckpt` |
| teacher | `checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt` |

`eval.py` on full val vs. paper Table II (paper values in parentheses):

| metric | no-KD | v1 | v2 |
|---|---|---|---|
| minADE | 0.7361 (0.736) | 0.6959 (0.696) | 0.6982 (0.698) |
| minFDE | 1.1568 (1.157) | 1.0503 (1.050) | 1.0505 (1.050) |
| MR | 0.1217 (0.122) | 0.1054 (0.105) | 0.1063 (0.106) |
| mixNLL | 26.610 (26.6) | 37.95 (38.0) | 24.18 (24.2) |
| calib err | 0.0325 (0.033) | 0.1835 (0.184) | 0.0275 (0.028) |
| mean b | 0.4179 (0.418) | 0.2675 (0.268) | 0.4126 (0.413) |
| cov@p90 | 0.9034 (0.903) | 0.7110 (0.711) | 0.9085 (0.909) |

For no-KD and v1, `best/last.ckpt` is byte-identical in metrics to
`best/HiVTKD-epoch=63-…`; either path works.

### Ruled out (do not use)

| Checkpoint | why not | minFDE | calib | b | cov90 |
|---|---|---|---|---|---|
| `HiVT-32/gxhl2ug9/.../epoch=63-step=411903` | calibration matches, **geometry does not** | 1.1782 | 0.0332 | 0.4201 | 0.9026 |
| `kd_ckpt/emb32-bs128-lkl0.0/best/last` | different run | 1.200* | 0.0304 | 0.430 | 0.9050 |
| `kd_ckpt/emb32-bs128-lkl0.5/best/*` | different run — was wrongly used as v1 | 1.064* | 0.1577 | 0.297 | 0.7530 |
| `kd_ckpt/triage-emb32-lr3e-3-kl0.0-nh2-full/best/last` | `nh2` variant, close but not it | 1.1622 | 0.0316 | 0.4162 | 0.9033 |

\* subset estimate, not eval.py.

The `gxhl2ug9` case is the instructive one: it matched `calib_err`, `cov@p90` **and**
`mean_b` while being wrong. **Calibration agreement alone never identifies a
checkpoint** — that error was made once during this very investigation before the
geometry columns caught it.

---

## Naming traps — read before picking any checkpoint

1. **`triage-` does not mean 25%-data.** `kd_ckpt/triage-emb32-lr3e-3-kl0.5-full` is a
   **full-data** run — the `-full` suffix is the operative part — and it is the paper's
   v1. The `triage-` prefix without `-full` *is* the 25% proxy.
2. **`kd_ckpt/emb32-bs128-lkl0.5` is NOT the paper's v1**, despite
   [handoff_kd_thesis_state.md](handoff_kd_thesis_state.md) §4 listing it under
   "full-data checkpoints (the headline numbers)". That line is **wrong**. It gives
   `minFDE 1.064, calib 0.158, b 0.298, cov@p90 0.753` — the calibration numbers are
   ~4 points off the paper's, which is what exposed it.
3. **Filename `val_minFDE` ≠ `eval.py` minFDE.** The six `emb32-bs128-lkl0.5/best`
   checkpoints all read `val_minFDE=1.12`; eval.py puts them at ~1.06. Never select on
   the filename.
4. **Checkpoint selection is by `val_minFDE`, which is blind to calibration.** Within a
   run, geometry plateaus while the Laplace scale keeps drifting, so neighbouring
   epochs with identical filename minFDE can differ in `cov@p90`. Always verify the
   calibration columns too, not just geometry.

## Discriminating metrics

`calib_err`, `mean_b` and `cov@p90` separate the candidate *directories* (they differ
by 4+ points between runs). `minADE`/`minFDE`/`MR`/`mixNLL` separate *checkpoints
within* a directory. **Both are required** — matching one family and not the other
means it is the wrong checkpoint.
