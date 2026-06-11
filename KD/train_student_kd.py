import os
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb, json
from pathlib import Path

import h5py

# Run-from-anywhere bootstrap: ensure the repo root is on sys.path so the
# repo packages (models, datasets, utils) and the KD package all import
# absolutely. Run as `python -m KD.train_student_kd` or `python KD/train_student_kd.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from KD.hivt_kd import HiVTKD
from KD.kd_datamodule import KDDataModule

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

    # Read teacher param count for wandb config. Support every cache layout:
    #   - directory cache:            <teacher_dir>/manifest.json
    #   - single .h5 + sidecar json:  <teacher_dir>.manifest.json  (save_teacher_ouputs.py)
    #   - single .h5 with _meta group: legacy bundler (prepare_teacher_cache.py)
    teacher_path = Path(args.teacher_dir)
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher cache not found: {teacher_path}")

    dir_manifest     = teacher_path / "manifest.json"
    sidecar_manifest = teacher_path.with_suffix(teacher_path.suffix + ".manifest.json")
    manifest = None
    if dir_manifest.exists():
        manifest = json.loads(dir_manifest.read_text())
    elif sidecar_manifest.exists():
        manifest = json.loads(sidecar_manifest.read_text())
    elif teacher_path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(teacher_path, "r") as h5_file:
            if "_meta" in h5_file:
                meta = h5_file["_meta"].attrs
                manifest = {"teacher_params": {
                    "total": int(meta.get("teacher_params_total", 0)),
                    "trainable": int(meta.get("teacher_params_trainable", 0)),
                }}
    if manifest is None:
        print(f"[warn] No teacher manifest/_meta found for {teacher_path}; "
              f"logging teacher_params=0.")
        manifest = {"teacher_params": {"total": 0, "trainable": 0}}

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
        save_last=True,  # also keep a rolling last.ckpt (most recent epoch) for crash-safe resume
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath=f"kd_ckpt/{run_name}/periodic",
        filename="HiVTKD-{epoch:02d}",
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_top_k=-1,
        save_last=False,  # Avoid duplication; best_checkpoint owns last.ckpt
    )

    # Auto-resume: if a previous run of this run_name left a last.ckpt, continue
    # from it (unless the user explicitly passed --resume_from_checkpoint).
    last_ckpt = Path(f"kd_ckpt/{run_name}/best/last.ckpt")
    if getattr(args, "resume_from_checkpoint", None) is None and last_ckpt.exists():
        args.resume_from_checkpoint = str(last_ckpt)
        print(f"[auto-resume] Found {last_ckpt} — resuming from it.")
    else:
        print("[auto-resume] No last.ckpt found — starting fresh.")

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