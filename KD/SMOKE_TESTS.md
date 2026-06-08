# Pre-Training Smoke Tests — Runbook

Run these in order. Do not start the multi-day KD run until all pass.

## What was verified from your files

- **Scale parity: OK.** `save_teacher_outputs.py` saves `scale = pred[..., 2:]`;
  the student uses `pred_s[..., 2:]`. Same decoder slice, same parametrisation.
  v1 uses only the *student* scale (positive, post-decoder); teacher scale is
  not consumed. Nothing to change.

- **Teacher-cache key: BUG FOUND + FIXED.** Saved/H5 keys are normalised to
  plain strings (`"123"`), but the old `kd_dataset._load_teacher` looked up
  `f"tensor({seq_id})"`, which never matched → `load()` returned `None` →
  `has_teacher=False` → **KD silently never ran**. The cache *validator* used
  the plain key, so it passed and hid the failure. Fixed in the new
  `kd_dataset.py`: it normalises the id the same way the saver does and looks
  up the plain key, with a fallback to the legacy `tensor(...)` form.

## Step 1 — Loss unit tests (seconds, no data needed)

```
python test_kd_loss.py
```
Expect all of: finite loss, permutation-invariance (teacher AND student mode
shuffles leave the loss unchanged), perfect-fit < noisy, `K!=F` works,
gradients reach the student but not the teacher, and the zero-scale clamp
holds. (The math behind these was independently confirmed in numpy:
mode-permutation changed the loss by 0.0e+00.)

## Step 2 — Pipeline smoke test (minutes, uses your real cache + dataset)

```
python smoke_test_pipeline.py \
    --root    /home/manyazog/argoverse \
    --teacher /home/manyazog/HiVT/teacher_outputs/train.h5 \
    --n 64
```
Stages: [0] cache opens and key format is printed, [1] one record has correct
shapes / no NaN / positive scale, [2] **the real dataset seq_id resolves to a
cache entry** (the check that catches the silent-KD bug), [3] `has_teacher` is
True for real items, [4] PyG stacks teacher tensors to `[B,F,H,2]` / `[B,F]`
and the train-time permute matches `num_graphs`. Stage [2] is the one that
would have caught the original bug — confirm it reports a resolving key.

## Step 3 — One-batch overfit (the real end-to-end smoke run)

Use Lightning's built-ins to run a couple of steps end to end before the full
job. Either:

```bash
# fastest: 1 train + 1 val batch, confirms no shape/NaN errors anywhere
python train_student_kd.py ... --fast_dev_run 5
 python -m KD.train_student_kd \
  --teacher_dir /home/manya/HiVT/teacher_outputs/train_fixed.h5 \
  --embed_dim 32 --lambda_task 1.0 --lambda_kl 0.5 --lambda_pi 0 \
  --data_root /home/manya/argoverse --root /home/manya/argoverse \
  --max_epochs 64 --gpus 1 --batch_size 64 \
  --checkpoint_every_n_epochs 10 --fast_dev_run 
```
or, to confirm the loss actually drives learning:
```bash
python train_student_kd.py ...--overfit_batches 1 --max_epochs 50 \
    --lambda_task 1.0 --lambda_kl 0.5 --lambda_pi 0
python -m KD.train_student_kd \
--overfit_batches 1 --max_epochs 50 \
  --teacher_dir /home/manya/HiVT/teacher_outputs/train_fixed.h5 \
  --embed_dim 32 --lambda_task 1.0 --lambda_kl 0.5 --lambda_pi 0 \
  --data_root /home/manya/argoverse --root /home/manya/argoverse \
  --max_epochs 64 --gpus 1 --batch_size 64 \
  --checkpoint_every_n_epochs 10 
```
Watch in wandb / stdout:
- `train/kd/mix_nll` is logged, finite, and **decreasing** (if it never
  appears, `has_teacher` is False → revisit Step 2).
- `train/loss_total` falls toward ~0 on the single overfit batch.
- `train/kd/mix_nll` and `train/loss_task` are in a comparable range; if KD
  dwarfs the task loss, lower `--lambda_kl`.

## Step 4 — (optional) Mode-alignment figure

```
python kd_mode_alignment_diagnostic.py \
    --teacher_ckpt .../HiVT-128/....ckpt \
    --student_ckpt kd_ckpt/best/....ckpt \
    --root /home/manyazog/argoverse --num_scenes 200
```
An "accidental alignment rate" near `1/6 ≈ 0.167` confirms teacher/student mode
indices are unrelated — your justification figure for the permutation-invariant
loss.

## Reminder for the launch command

Set `--lambda_pi 0` (it is now ignored and will warn otherwise). Keep
vanilla-32 and KD-32 at the same epochs / data subset / seeds for a fair
comparison.

## Minor cleanup noted (non-blocking)

`TeacherStore.close()` references `self._h5_file`, but the attribute is
`self._h5`; it will raise if ever called. `__del__` is correct, so this only
matters if you call `close()` explicitly.
