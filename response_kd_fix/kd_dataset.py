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

from .kd_teacher_store_fixed import TeacherStore
# in kd_dataset.py

class KDData(Data):
    """Data subclass that tells PyG how to batch teacher soft targets."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("teacher_loc", "teacher_scale", "teacher_pi"):
            # Stack along a NEW dim-0 → [B, F, H, 2] and [B, F]
            # Returning None means PyG will torch.stack instead of torch.cat
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("teacher_loc", "teacher_scale", "teacher_pi", "has_teacher"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)

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

    # def _load_teacher(self, seq_id):
    #     key = str(seq_id)
    #     return self.teacher_store.load(key)
    def _load_teacher(self, seq_id):
        # Format the key to match the "tensor(ID)" pattern stored in the H5 file
        key = f"tensor({seq_id})"
        return self.teacher_store.load(key)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        base_data = self.base[idx]           # torch_geometric Data
        data = KDData(**{k: v for k, v in base_data})   # copy all existing attrs
        seq_id = data.seq_id

        teacher = self._load_teacher(seq_id)
        if teacher is None:
            if seq_id not in self._missing_warned:
                warnings.warn(
                    f"[KDDataset] teacher file not found for seq_id={seq_id} in {self.teacher_dir}. "
                    f"KD loss will be skipped for this seq_id.",
                    stacklevel=2,
                )
                self._missing_warned.add(seq_id)
            # Signal downstream that teacher data is absent
            data.has_teacher = torch.tensor(False)
            return data

        data.teacher_loc   = teacher["loc"]    # [F, H, 2] [6, 30, 2]  — no unsqueeze needed
        data.teacher_scale = teacher["scale"]  # [F, H, 2]
        data.teacher_pi    = teacher["pi"]     # [F]
        data.has_teacher   = torch.tensor(True)
        # note in old file:
        # data.teacher_loc   = teacher["loc"].unsqueeze(0)    # [6,30,2] → [1,6,30,2]
        # data.teacher_scale = teacher["scale"].unsqueeze(0)  # [6,30,2] → [1,6,30,2]
        # data.teacher_pi    = teacher["pi"].unsqueeze(0)     # [6]      → [1,6]
        # data.has_teacher   = torch.tensor(True)
        return data