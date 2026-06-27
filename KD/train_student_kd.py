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
                             "Defaults to emb<E>-bs<B>-lkl<L> (with -s<seed> appended "
                             "when --seed is not the default).")
    parser.add_argument("--seed", type=int, default=2022,
                        help="Global RNG seed (pl.seed_everything). Default 2022 matches "
                             "train.py and the original single-seed KD runs; vary it "
                             "(e.g. 1, 2, 3) for multi-seed robustness studies.")
    args = parser.parse_args()

    # Seed BEFORE any model/data construction so weight init and data shuffling are
    # reproducible. Default 2022 reproduces the original runs exactly (a small KD
    # effect can otherwise be lost in init noise, so this matters for comparability).
    pl.seed_everything(args.seed)

    # Read teacher param count for wandb config. Support every cache layout:
    #   - directory cache:            <teacher_dir>/manifest.json
    #   - single .h5 + sidecar json:  <teacher_dir>.manifest.json  (save_teacher_ouputs.py)
    #   - single .h5 with _meta group: legacy bundler (prepare_teacher_cache.py)
    # No teacher_dir -> teacher-free run (e.g. lambda_kl=0 LR triage). Skip the
    # manifest lookup and cache validation entirely so setup stays fast.
    if args.teacher_dir is None:
        print("[teacher] no --teacher_dir given; running teacher-free "
              "(KD term disabled). Ensure --lambda_kl 0.")
        manifest = {"teacher_params": {"total": 0, "trainable": 0}}
        teacher_path = None
    else:
        teacher_path = Path(args.teacher_dir)
    if teacher_path is not None and not teacher_path.exists():
        raise FileNotFoundError(f"Teacher cache not found: {teacher_path}")

    if teacher_path is not None:
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
    # Keep seed-varied runs in distinct checkpoint dirs so they never collide with
    # (or auto-resume) the default-seed run. Explicit --run_name is left untouched.
    if args.run_name is None and args.seed != 2022:
        run_name += f"-s{args.seed}"

    # Auto-resume (decided BEFORE building the logger so wandb can resume too):
    # if a previous run of this run_name left a last.ckpt, continue from it
    # (unless the user explicitly passed --resume_from_checkpoint).
    run_dir       = Path(f"kd_ckpt/{run_name}")
    last_ckpt     = run_dir / "best" / "last.ckpt"
    wandb_id_file = run_dir / "wandb_run_id.txt"

    if getattr(args, "resume_from_checkpoint", None) is None and last_ckpt.exists():
        args.resume_from_checkpoint = str(last_ckpt)
        print(f"[auto-resume] Found {last_ckpt} — resuming from it.")
    elif getattr(args, "resume_from_checkpoint", None) is not None:
        print(f"[auto-resume] Using explicit checkpoint {args.resume_from_checkpoint}.")
    else:
        print("[auto-resume] No last.ckpt found — starting fresh.")

    # Resume the SAME wandb run across crashes. The run id is persisted next to
    # the checkpoints on first launch and read back when we resume from a ckpt,
    # so the dashboard shows one continuous run instead of a new one per restart.
    resuming = getattr(args, "resume_from_checkpoint", None) is not None
    if resuming and wandb_id_file.exists():
        wandb_id     = wandb_id_file.read_text().strip()
        wandb_resume = "allow"   # continue this id; create it if the server lost it
        print(f"[wandb-resume] Continuing wandb run id={wandb_id}.")
    else:
        wandb_id     = wandb.util.generate_id()
        wandb_resume = "never"
        print(f"[wandb-resume] New wandb run id={wandb_id}.")

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        group=args.wandb_group,
        name=run_name,
        id=wandb_id,
        resume=wandb_resume,
        config={
            **vars(args),
            "teacher_params": manifest["teacher_params"]["total"],
        },
    )

    # Persist the id (idempotent) so the next crash/resume reuses this run.
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_id_file.write_text(wandb_id)

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

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[best_checkpoint, periodic_checkpoint,
                   LearningRateMonitor(logging_interval="epoch")],
    )

    model = HiVTKD(**vars(args))
    # Log student param count into wandb config right after init
    counts = model.log_parameter_counts()
    # allow_val_change: on resume these keys already exist in the run config, and
    # compression_ratio can differ by a float ULP between launches — without this,
    # wandb raises ConfigError and aborts the resumed run.
    wandb_logger.experiment.config.update({
        "student_params": counts["total"],
        "compression_ratio": manifest["teacher_params"]["total"] / counts["total"],
    }, allow_val_change=True)

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