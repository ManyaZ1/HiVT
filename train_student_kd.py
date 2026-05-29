# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
# import wandb, json
# from pathlib import Path

# from hivt_kd import HiVTKD
# from kd_datamodule import KDDataModule

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser = HiVTKD.add_model_specific_args(parser)   # HiVT args + KD args
#     parser = KDDataModule.add_argparse_args(parser)
#     parser = pl.Trainer.add_argparse_args(parser)
#     args = parser.parse_args()

#     # Read teacher param count from manifest for wandb config
#     manifest_path = Path(args.teacher_dir) / "manifest.json"
#     manifest = json.loads(manifest_path.read_text())

#     wandb_logger = WandbLogger(
#         project="hivt-kd",
#         name=f"student64-kd-lkl{args.lambda_kl}-lpi{args.lambda_pi}",
#         config={
#             **vars(args),
#             "teacher_params": manifest["teacher_params"]["total"],
#         },
#     )

#     best_checkpoint = ModelCheckpoint(
#         monitor="val_minFDE",
#         save_top_k=5,
#         mode="min",
#         dirpath="kd_ckpt/best",
#         filename="HiVTKD-{epoch:02d}-{val_minFDE:.2f}",
#     )
#     periodic_checkpoint = ModelCheckpoint(
#         dirpath="kd_ckpt/periodic",
#         filename="HiVTKD-{epoch:02d}",
#         every_n_epochs=args.checkpoint_every_n_epochs,
#         save_top_k=-1,
#         save_last=True,
#     )

#     trainer = pl.Trainer.from_argparse_args(
#         args,
#         logger=wandb_logger,
#         callbacks=[best_checkpoint, periodic_checkpoint],
#     )

#     model = HiVTKD(**vars(args))
#     # Log student param count into wandb config right after init
#     counts = model.log_parameter_counts()
#     wandb_logger.experiment.config.update({
#         "student_params": counts["total"],
#         "compression_ratio": manifest["teacher_params"]["total"] / counts["total"],
#     })

#     datamodule = KDDataModule.from_argparse_args(args)
#     trainer.fit(model, datamodule)


# # python train_student_kd.py \
# #   --teacher_dir teacher_outputs/train \
# #   --embed_dim 64 \
# #   --lambda_task 1.0 \
# #   --lambda_kl 0.5 \
# #   --lambda_pi 0.5 \
# #   --data_root /home/manyazog/argoverse \
# #   --root /home/manyazog/argoverse \
# #   --max_epochs 64 \
# #   --gpus 1 \
# #   --batch_size 64 \
# #   --checkpoint_every_n_epochs 5

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb, json
from pathlib import Path

import h5py

from hivt_kd import HiVTKD
from kd_datamodule import KDDataModule

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser = HiVTKD.add_model_specific_args(parser)   # HiVT args + KD args
    parser = KDDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
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

    wandb_logger = WandbLogger(
        project="hivt-kd",
        name=f"student64-kd-lkl{args.lambda_kl}-lpi{args.lambda_pi}",
        config={
            **vars(args),
            "teacher_params": manifest["teacher_params"]["total"],
        },
    )

    best_checkpoint = ModelCheckpoint(
        monitor="val_minFDE",
        save_top_k=5,
        mode="min",
        dirpath="kd_ckpt/best",
        filename="HiVTKD-{epoch:02d}-{val_minFDE:.2f}",
        check_on_train_epoch_end=False,  # Only check after validation, not after training
    )
    periodic_checkpoint = ModelCheckpoint(
        dirpath="kd_ckpt/periodic",
        filename="HiVTKD-{epoch:02d}",
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_top_k=-1,
        save_last=False,  # Avoid duplication; best_checkpoint handles the last model
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[best_checkpoint, periodic_checkpoint],
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


# python train_student_kd.py \
#   --teacher_dir teacher_outputs/train \
#   --embed_dim 64 \
#   --lambda_task 1.0 \
#   --lambda_kl 0.5 \
#   --lambda_pi 0.5 \
#   --data_root /home/manyazog/argoverse \
#   --root /home/manyazog/argoverse \
#   --max_epochs 64 \
#   --gpus 1 \
#   --batch_size 64 \
#   --checkpoint_every_n_epochs 5