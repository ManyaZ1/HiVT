"""
kd_dataset.py

Wraps ArgoverseV1Dataset to attach pre-saved teacher soft targets
to each Data object before it enters the student training loop.

The teacher .pt files are written by save_teacher_outputs.py:
    teacher_outputs/train/<seq_id>.pt
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
        self._missing_warned = set()   # warn once per missing seq_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]           # torch_geometric Data
        seq_id = data.seq_id

        teacher_path = self.teacher_dir / f"{seq_id}.pt"

        if not teacher_path.exists():
            if seq_id not in self._missing_warned:
                warnings.warn(
                    f"[KDDataset] teacher file not found: {teacher_path}. "
                    f"KD loss will be skipped for this seq_id.",
                    stacklevel=2,
                )
                self._missing_warned.add(seq_id)
            # Signal downstream that teacher data is absent
            data.has_teacher = False
            return data

        teacher = torch.load(teacher_path, map_location="cpu")

        # Add an explicit batch axis so PyG collation produces [B, F, ...]
        # instead of flattening modes into a single [B*F, ...] dimension.
        loc = teacher["loc"]
        scale = teacher["scale"]
        pi = teacher["pi"]
        data.teacher_loc = loc.unsqueeze(0) if loc.dim() == 3 else loc
        data.teacher_scale = scale.unsqueeze(0) if scale.dim() == 3 else scale
        data.teacher_pi = pi.unsqueeze(0) if pi.dim() == 1 else pi
        data.has_teacher   = True

        return data