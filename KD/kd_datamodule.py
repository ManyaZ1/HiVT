"""
kd_datamodule.py

KDDataModule — wraps ArgoverseV1DataModule so that every training sample
carries its teacher soft targets from the pre-saved .pt files.

Validation uses the plain dataset (no teacher targets needed at eval time).
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.data import DataLoader as PyGDataLoader

from datasets import ArgoverseV1Dataset             # your existing dataset
from kd_dataset import KDDataset                    # the wrapper we wrote
from kd_teacher_store import TeacherStore


class KDDataModule(pl.LightningDataModule):
    """
    Mirrors the interface of ArgoverseV1DataModule.
    The only additions are:
        --teacher_dir    path to saved teacher .pt files
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.teacher_dir  = kwargs.pop("teacher_dir", None)
        self.root         = kwargs["root"]
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.val_batch_size   = kwargs.get("val_batch_size",   32)
        self.num_workers      = kwargs.get("num_workers",       8)
        self.pin_memory       = kwargs.get("pin_memory",      False)
        self.persistent_workers = kwargs.get("persistent_workers", True)
        self.local_radius = kwargs.get("local_radius", 50)
        self.train_transform = kwargs.get("train_transform")
        self.val_transform = kwargs.get("val_transform")
        self._kwargs = kwargs   # forward the rest to ArgoverseV1Dataset

    def prepare_data(self):
        print(f"Preparing Argoverse datasets under {self.root} ...")
        ArgoverseV1Dataset(root=self.root, split="train", local_radius=self.local_radius)
        ArgoverseV1Dataset(root=self.root, split="val", local_radius=self.local_radius)
        print("Dataset preprocessing complete.")

    def _validate_teacher_cache(self, dataset):
        if self.teacher_dir is None:
            return
        store = TeacherStore(self.teacher_dir, cache_size=0)
        print(f"Validating teacher cache at {self.teacher_dir} for {len(dataset.processed_paths)} training scenes ...")
        missing = []
        for processed_path in dataset.processed_paths:
            seq_id = Path(processed_path).stem
            if not store.exists(seq_id):
                missing.append(str(seq_id))
                if len(missing) >= 20:
                    break
        if missing:
            raise FileNotFoundError(
                "Teacher cache is incomplete. Missing seq_ids include: "
                + ", ".join(missing)
            )
        print("Teacher cache validation passed.")

    def setup(self, stage=None):
        print("Building KD datasets ...")
        base_train = ArgoverseV1Dataset(
            root=self.root,
            split="train",
            transform=self.train_transform,
            local_radius=self.local_radius,
        )
        self._validate_teacher_cache(base_train)
        if self.teacher_dir is not None:
            self.train_dataset = KDDataset(base_train, teacher_dir=self.teacher_dir)
        else:
            self.train_dataset = base_train
        self.val_dataset = ArgoverseV1Dataset(
            root=self.root,
            split="val",
            transform=self.val_transform,
            local_radius=self.local_radius,
        )
        print("KD datasets ready.")

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser):
        parser = parent_parser
        # Dataset / datamodule related args (keep consistent with other scripts).
        # Register each option independently so one duplicate does not prevent
        # later compatibility flags from being added.
        def add_argument(*args, **kwargs):
            try:
                parser.add_argument(*args, **kwargs)
            except argparse.ArgumentError:
                pass

        add_argument("--root", type=str, default=None,
                     help="Path to dataset root (preferred)")
        add_argument("--data_root", type=str, default=None,
                     help="Alias for --root (backwards compatibility)")
        add_argument("--train_batch_size", type=int, default=32)
        add_argument("--val_batch_size", type=int, default=32)
        add_argument("--shuffle", type=bool, default=True)
        add_argument("--num_workers", type=int, default=8)
        add_argument("--pin_memory", type=bool, default=False)
        add_argument("--persistent_workers", type=bool, default=True)
        # local_radius is already provided by HiVT.add_model_specific_args.
        # Keep the parser stable by not redefining it here.
        add_argument("--batch_size", type=int, default=None,
                     help="If set, overrides train/val batch sizes")
        add_argument("--checkpoint_every_n_epochs", type=int, default=20,
                     help="Save periodic checkpoint every N epochs")
        # Teacher cache directory
        add_argument("--teacher_dir", type=str, default=None)
        # Can point to either a directory of .pt files or a single .h5 file.
        return parser

    @classmethod
    def from_argparse_args(cls, args):
        # Map legacy / convenience arguments into the constructor kwargs
        # Prefer explicit --root; if not present, fall back to --data_root
        if getattr(args, "root", None) is None and getattr(args, "data_root", None):
            args.root = args.data_root
        # If a generic --batch_size was provided, map it to train/val batch sizes
        if getattr(args, "batch_size", None) is not None:
            args.train_batch_size = args.batch_size
            args.val_batch_size = args.batch_size
        return cls(**vars(args))

    #      # Add missing training-specific CLI flags that other scripts may use
    # try:
    #     parser.add_argument("--checkpoint_every_n_epochs", type=int, default=5,
    #                         help="Save periodic checkpoint every N epochs")
    # except argparse.ArgumentError:
    #     pass