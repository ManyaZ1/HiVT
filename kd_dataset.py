"""
kd_dataset.py

Wraps ArgoverseV1Dataset to attach pre-saved teacher soft targets
to each Data object before it enters the student training loop.

The teacher cache can be either a directory of .pt files written by
save_teacher_outputs.py:
    teacher_outputs/train/<seq_id>.pt
or a single indexed HDF5 file written by prepare_teacher_cache.py:
    teacher_outputs/train.h5
    {
        "seq_id": str,
        "loc":    Tensor [F, H, 2],
        "scale":  Tensor [F, H, 2],
        "pi":     Tensor [F],        # raw logits
    }

Usage:
    base = ArgoverseV1Dataset(root=..., split="train")
    dataset = KDDataset(base, teacher_dir="teacher_outputs/train")
"""

import warnings
from pathlib import Path

import torch
from torch_geometric.data import Data

from kd_teacher_store import TeacherStore


class KDDataset(torch.utils.data.Dataset):
    """
    Thin wrapper around any map-style Dataset whose items are
    torch_geometric Data objects with a .seq_id attribute.

    Attaches teacher tensors as extra attributes on the Data object:
        data.teacher_loc    [F, H, 2]
        data.teacher_scale  [F, H, 2]
        data.teacher_pi     [F]

    If a teacher file is missing the item is returned unchanged and
    a warning is emitted once.  This allows partial teacher caches
    during development without crashing the training run.
    """

    def __init__(self, base_dataset, teacher_dir: str):
        self.base       = base_dataset
        self.teacher_dir = Path(teacher_dir)
        self.teacher_store = TeacherStore(self.teacher_dir, cache_size=1024)
        self._missing_warned = set()   # warn once per missing seq_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]           # torch_geometric Data
        seq_id = data.seq_id

        if not self.teacher_store.exists(seq_id):
            if seq_id not in self._missing_warned:
                warnings.warn(
                    f"[KDDataset] teacher file not found for seq_id={seq_id} in {self.teacher_dir}. "
                    f"KD loss will be skipped for this seq_id.",
                    stacklevel=2,
                )
                self._missing_warned.add(seq_id)
            # Signal downstream that teacher data is absent
            data.has_teacher = False
            return data

        teacher = self.teacher_store.load(seq_id)
        if teacher is None:
            data.has_teacher = False
            return data

        data.teacher_loc   = teacher["loc"]    # [F, H, 2]
        data.teacher_scale = teacher["scale"]  # [F, H, 2]
        data.teacher_pi    = teacher["pi"]     # [F]
        data.has_teacher   = True

        return data