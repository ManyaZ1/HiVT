# KD Training Overview

This repo trains a smaller HiVT student with knowledge distillation from a saved HiVT-128 teacher.

## Main Pieces

- `save_teacher_outputs.py` runs the teacher model once over the dataset and saves per-scene outputs as `.pt` files.
- `prepare_teacher_cache.py` converts those `.pt` files into a single indexed `.h5` file.
- `kd_dataset.py` loads the normal `ArgoverseV1Dataset` sample and attaches the matching teacher tensors.
- `kd_datamodule.py` wraps the dataset logic and gives the trainer a KD-aware train loader plus a normal validation loader.
- `kd_loss.py` computes the KD objective from student and teacher output distributions.
- `hivt_kd.py` combines the student HiVT model, the base task loss, and the KD loss.
- `train_student_kd.py` is the entry point that wires everything together.

## Data Flow

1. Train or load a HiVT-128 teacher.
2. Use `save_teacher_outputs.py` to write one file per scene into `teacher_outputs/train/`.
3. Convert the directory into a single indexed file with `prepare_teacher_cache.py`.
4. Each saved file contains:
   - `loc`: Laplace means with shape `[F, H, 2]`
   - `scale`: Laplace scales with shape `[F, H, 2]`
   - `pi`: mode logits with shape `[F]`
5. During student training, `KDDataset` loads the scene and adds:
   - `data.teacher_loc`
   - `data.teacher_scale`
   - `data.teacher_pi`
   - `data.has_teacher`

You can now point `--teacher_dir` at either the original directory or the new `.h5` file.

## Training Flow

`train_student_kd.py` parses the HiVT args, the KD args, and the trainer args. It then:

- loads `manifest.json` from the teacher directory to record teacher metadata in W&B
- if you use a `.h5` file, reads teacher metadata from the file's `_meta` group
- creates `KDDataModule`
- creates `HiVTKD`
- starts `trainer.fit(model, datamodule)`

Inside `HiVTKD`:

- the inner `HiVT` instance is the actual student network
- `forward()` just delegates to the student
- the task loss matches the base HiVT objective
- if teacher tensors are present, `HiVTKDLoss` adds the distillation term
- validation uses the same metrics as the base model

## KD Loss

`kd_loss.py` uses two terms:

- reverse KL between teacher and student Laplace distributions, weighted by teacher mode probabilities
- cross-entropy between teacher and student mode logits

The loss is intended to make the student match the teacher’s output distribution while still learning from ground truth.

## Important Files On Disk

- teacher cache: `teacher_outputs/train/<seq_id>.pt`
- teacher cache: `teacher_outputs/train.h5`
- teacher manifest: `teacher_outputs/train/manifest.json`
- optional sanity stats: `teacher_outputs/train/sanity_stats.json`

Example conversion command:

```bash
python prepare_teacher_cache.py --input_dir teacher_outputs/train --output_h5 teacher_outputs/train.h5 --overwrite
```

## In One Sentence

The teacher cache supplies soft targets, the datamodule attaches them to each batch, the KD wrapper computes both the normal HiVT loss and the distillation loss, and the trainer optimizes a 64-dim student to imitate the 128-dim teacher.