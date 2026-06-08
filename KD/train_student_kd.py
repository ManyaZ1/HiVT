import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb, json
from pathlib import Path

import h5py

# Run-from-anywhere bootstrap: put the repo root and this KD dir on sys.path so
# the repo packages (models, datasets) and the sibling KD modules import under
# one flat convention. See KD/KD_CHANGES.md.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hivt_kd import HiVTKD
from kd_datamodule import KDDataModule

if __name__ == "__main__":
    import argparse
    # Match train.py's seed so KD runs are reproducible and comparable to the
    # HiVT-32 baseline (a small KD effect can otherwise be lost in init noise).
    pl.seed_everything(2022)
    parser = argparse.ArgumentParser()
    parser = HiVTKD.add_model_specific_args(parser)   # HiVT args + KD args
    parser = KDDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--wandb_project", type=str, default="hivt-kd",
                        help="wandb project — the top-level folder on wandb.ai.")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="wandb group to bucket related runs (e.g. a lambda_kl sweep).")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name; also the checkpoint subfolder under kd_ckpt/. "
                             "Defaults to emb<E>-bs<B>-lkl<L>.")
    args = parser.parse_args()

    # Read teacher param count from manifest for wandb config
    manifest_path = Path(args.teacher_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        teacher_store_path = Path(args.teacher_dir)
        if not teacher_store_path.exists():
            raise FileNotFoundError(f"Teacher cache not found: {teacher_store_path}")
        if teacher_store_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise FileNotFoundError(
                f"Expected manifest.json next to {teacher_store_path}, but it was missing."
            )
        with h5py.File(teacher_store_path, "r") as h5_file:
            meta = h5_file["_meta"].attrs
            manifest = {
                "teacher_params": {
                    "total": int(meta.get("teacher_params_total", 0)),
                    "trainable": int(meta.get("teacher_params_trainable", 0)),
                }
            }

    # --batch_size -> train_batch_size mapping happens later in from_argparse_args,
    # so read the generic flag here for naming.
    _bs = getattr(args, "batch_size", None) or args.train_batch_size
    run_name = args.run_name or f"emb{args.embed_dim}-bs{_bs}-lkl{args.lambda_kl}"

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        config={
            **vars(args),
            "teacher_params": manifest["teacher_params"]["total"],
        },
    )

    best_checkpoint = ModelCheckpoint(
        monitor="val_minFDE",
        save_top_k=5,
        mode="min",
        dirpath=f"kd_ckpt/{run_name}/best",
        filename="HiVTKD-{epoch:02d}-{val_minFDE:.2f}",
        #check_on_train_epoch_end=False,  # Only check after validation, not after training
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=f"kd_ckpt/{run_name}/periodic",
        filename="HiVTKD-{epoch:02d}",
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_top_k=-1,
        save_last=False,  # Avoid duplication; best_checkpoint handles the last model
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[best_checkpoint, periodic_checkpoint,
                   LearningRateMonitor(logging_interval="epoch")],
    )

    model = HiVTKD(**vars(args))
    # Log student param count into wandb config right after init
    counts = model.log_parameter_counts()
    wandb_logger.experiment.config.update({
        "student_params": counts["total"],
        "compression_ratio": manifest["teacher_params"]["total"] / counts["total"],
    })

    datamodule = KDDataModule.from_argparse_args(args)
    trainer.fit(model, datamodule)


# python -m KD.train_student_kd \
#   --teacher_dir teacher_outputs/train \
#   --embed_dim 32 \
#   --lambda_task 1.0 \
#   --lambda_kl 0.5 \
#   --lambda_pi 0.5 \
#   --data_root /home/manya/argoverse \
#   --root /home/manya/argoverse \
#   --max_epochs 64 \
#   --gpus 1 \
#   --batch_size 64 \
#   --checkpoint_every_n_epochs 10 \
# --fast_dev_run 5