"""
kd_dataset.py

Wraps ArgoverseV1Dataset to attach pre-saved teacher soft targets
to each Data object before it enters the student training loop.

FIX vs previous version
-----------------------
The teacher cache is keyed by the *normalized* seq_id string (e.g. "123"),
because both save_teacher_outputs.py and prepare_teacher_cache.py route the
id through normalize_seq_id() -> str(int). The previous _load_teacher looked
up f"tensor({seq_id})", which never matched, so load() always returned None,
has_teacher was always False, and KD silently never ran (while the cache
validator, which used the plain key, still passed).

_load_teacher now normalizes seq_id the same way the saver did and looks up the
plain string key, with a fallback to the legacy "tensor(<id>)" form so older
caches still work.
"""

import warnings
from pathlib import Path

import torch
#from torch_geometric.data import Data # more data corruptiion from PyG: my favorite      !!!

from utils import TemporalData                # repo module (absolute; repo root on sys.path)
from KD.kd_teacher_store import TeacherStore  # sibling KD module (absolute from repo root)


class KDData(TemporalData):
    """Data subclass that tells PyG how to batch teacher soft targets."""

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("teacher_loc", "teacher_scale", "teacher_pi"):
            # Returning None makes PyG torch.stack along a NEW dim-0
            #   teacher_loc/scale -> [B, F, H, 2],  teacher_pi -> [B, F]
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
    a warning is emitted once.
    """

    def __init__(self, base_dataset, teacher_dir: str):
        self.base          = base_dataset
        self.teacher_dir   = Path(teacher_dir)
        self.teacher_store = TeacherStore(self.teacher_dir, cache_size=1024)
        self._missing_warned = set()

    @staticmethod
    def _normalize_seq_id(seq_id) -> str:
        """Match the normalization used in save_teacher_outputs.normalize_seq_id."""
        if isinstance(seq_id, torch.Tensor):
            seq_id = seq_id.item() if seq_id.numel() == 1 else seq_id.tolist()
        if isinstance(seq_id, (list, tuple)) and len(seq_id) == 1:
            seq_id = seq_id[0]
        return str(seq_id)

    def _load_teacher(self, seq_id):
        key = self._normalize_seq_id(seq_id)
        teacher = self.teacher_store.load(key)
        if teacher is None:
            # backward-compat with any older cache keyed as "tensor(<id>)"
            teacher = self.teacher_store.load(f"tensor({key})")
        return teacher

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        base_data = self.base[idx]                      # torch_geometric Data
        data = KDData(**{k: v for k, v in base_data})   # copy all existing attrs
        seq_id = data.seq_id

        teacher = self._load_teacher(seq_id)
        if teacher is None:
            if seq_id not in self._missing_warned:
                warnings.warn(
                    f"[KDDataset] teacher targets not found for seq_id={seq_id} "
                    f"in {self.teacher_dir}. KD loss will be skipped for this seq_id.",
                    stacklevel=2,
                )
                self._missing_warned.add(seq_id)
            data.has_teacher = torch.tensor(False)
            return data

        data.teacher_loc   = teacher["loc"]     # [F, H, 2]
        data.teacher_scale = teacher["scale"]   # [F, H, 2]
        data.teacher_pi    = teacher["pi"]      # [F]
        data.has_teacher   = torch.tensor(True)
        return data
