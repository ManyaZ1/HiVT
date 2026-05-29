"""
kd_teacher_store.py

Read teacher outputs either from a directory of per-scene .pt files or from a
single indexed HDF5 cache. The store is designed to be opened lazily inside
each DataLoader worker.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

import h5py
import torch


class TeacherStore:
    def __init__(self, teacher_path, cache_size: int = 1024):
        self.path = Path(teacher_path)
        self.cache_size = max(int(cache_size), 0)
        self.is_hdf5 = self.path.suffix.lower() in {".h5", ".hdf5"}
        self._cache = OrderedDict()
        self._h5_file = None
        self._hdf5_keys = None

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def exists(self, seq_id) -> bool:
        key = str(seq_id)
        if self.is_hdf5:
            if not self.path.exists():
                return False
            return self._resolve_hdf5_key(key) is not None
        return self._pt_path(key).exists() or self._pt_path(self._tensor_key(key)).exists()

    def load(self, seq_id) -> Optional[Dict[str, torch.Tensor]]:
        key = str(seq_id)

        if self.cache_size > 0 and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if self.is_hdf5:
            record = self._load_hdf5(key)
        else:
            record = self._load_pt(key)

        if record is not None and self.cache_size > 0:
            self._cache[key] = record
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return record

    def _load_pt(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        teacher_path = self._pt_path(key)
        if not teacher_path.exists():
            teacher_path = self._pt_path(self._tensor_key(key))
        if not teacher_path.exists():
            return None

        teacher = torch.load(teacher_path, map_location="cpu")
        return {
            "loc": teacher["loc"],
            "scale": teacher["scale"],
            "pi": teacher["pi"],
        }

    def _open_hdf5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.path, "r")
        return self._h5_file

    def _ensure_hdf5_keys(self):
        if self._hdf5_keys is None:
            h5_file = self._open_hdf5()
            self._hdf5_keys = {key for key in h5_file.keys() if key != "_meta"}

    def _resolve_hdf5_key(self, key: str):
        self._ensure_hdf5_keys()
        if key in self._hdf5_keys:
            return key
        tensor_key = self._tensor_key(key)
        if tensor_key in self._hdf5_keys:
            return tensor_key
        return None

    def _load_hdf5(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        if not self.path.exists():
            return None

        h5_file = self._open_hdf5()
        resolved_key = self._resolve_hdf5_key(key)
        if resolved_key is None:
            return None

        group = h5_file[resolved_key]
        return {
            "loc": torch.from_numpy(group["loc"][()]),
            "scale": torch.from_numpy(group["scale"][()]),
            "pi": torch.from_numpy(group["pi"][()]),
        }

    def _tensor_key(self, key: str) -> str:
        return f"tensor({key})"

    def _pt_path(self, key: str) -> Path:
        return self.path / f"{key}.pt"

    def _has_key(self, h5_file, key: str) -> bool:
        return self._resolve_hdf5_key(key) is not None
