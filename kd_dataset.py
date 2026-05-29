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

import h5py
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
        self.teacher_store = None
        self.teacher_records = None
        if self.teacher_dir.suffix.lower() in {".h5", ".hdf5"}:
            self.teacher_records = self._load_hdf5_records(self.teacher_dir)
        else:
            self.teacher_store = TeacherStore(self.teacher_dir, cache_size=1024)
        self._missing_warned = set()   # warn once per missing seq_id

    def _load_hdf5_records(self, teacher_path: Path):
        if not teacher_path.exists():
            return {}

        records = {}
        with h5py.File(teacher_path, "r") as h5_file:
            for key in h5_file.keys():
                if key == "_meta":
                    continue
                group = h5_file[key]
                records[str(key)] = {
                    "loc": torch.from_numpy(group["loc"][()]),
                    "scale": torch.from_numpy(group["scale"][()]),
                    "pi": torch.from_numpy(group["pi"][()]),
                }
        return records

    def _load_teacher(self, seq_id):
        key = str(seq_id)
        if self.teacher_records is not None:
            return self.teacher_records.get(key)
        if self.teacher_store is None:
            return None
        return self.teacher_store.load(key)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]           # torch_geometric Data
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
            data.has_teacher = False
            return data

        data.teacher_loc   = teacher["loc"]    # [F, H, 2]
        data.teacher_scale = teacher["scale"]  # [F, H, 2]
        data.teacher_pi    = teacher["pi"]     # [F]
        data.has_teacher   = True

        return data