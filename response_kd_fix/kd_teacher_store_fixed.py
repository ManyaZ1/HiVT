"""
kd_teacher_store_fixed.py
=========================
Read teacher outputs 

after loading from HDF5, squeeze out any spurious leading-1 
dimension that was introduced either during numpy→tensor conversion or by an
older save path that did `tensor.unsqueeze(0)` before writing.

Expected shapes OUT of load():
    loc   : [6, 30, 2]   (F, H, 2)
    scale : [6, 30, 2]
    pi    : [6]           (F,)
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict


_EXPECTED = {
    "loc":   (6, 30, 2),
    "scale": (6, 30, 2),
    "pi":    (6,),
}


def _squeeze_to(tensor: torch.Tensor, expected_ndim: int, key: str) -> torch.Tensor:
    """Remove leading size-1 dimensions until tensor has expected_ndim dims."""
    while tensor.dim() > expected_ndim and tensor.shape[0] == 1: # tensor.shape[0] == 1 checks for leading-1 dim
        tensor = tensor.squeeze(0)
    if tensor.dim() != expected_ndim: #control will never reach here?
        raise ValueError(
            f"TeacherStore: '{key}' has shape {tuple(tensor.shape)} "
            f"after squeezing; expected {expected_ndim}D"
        )
    return tensor


class TeacherStore:
    """
    Reads teacher tensors from either:
      (a) a directory of per-scene .pt files, or
      (b) a single .h5 file produced by prepare_teacher_cache.py

    Always returns tensors in the canonical shapes:
        loc   [F, H, 2]
        scale [F, H, 2]
        pi    [F]
    """

    def __init__(self, path, cache_size: int = 1024):
        self.path = Path(path)
        self._is_h5 = self.path.suffix.lower() in {".h5", ".hdf5"}
        self._h5: Optional[h5py.File] = None
        self._cache: Dict[str, dict] = {} # why not OrderedDict? LRU eviction is handled by popping a random key when cache_size is exceeded, not by tracking access order.
        self._cache_size = cache_size

        if self._is_h5:
            if not self.path.exists():
                raise FileNotFoundError(f"Teacher HDF5 not found: {self.path}")
            self._h5 = h5py.File(self.path, "r")
        else:
            if not self.path.is_dir():
                raise FileNotFoundError(f"Teacher dir not found: {self.path}")
    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
    # ------------------------------------------------------------------ #
    def exists(self, seq_id: str) -> bool:
        if self._is_h5:
            return seq_id in self._h5
        return (self.path / f"{seq_id}.pt").exists()

    # ------------------------------------------------------------------ #
    def load(self, seq_id: str) -> Optional[dict]:
        if seq_id in self._cache:
            return self._cache[seq_id]

        try:
            result = self._load_raw(seq_id)
        except (KeyError, FileNotFoundError):
            return None

        # ── shape normalisation ────────────────────────────────────────
        result["loc"]   = _squeeze_to(result["loc"],   3, "loc")    # [F, H, 2]
        result["scale"] = _squeeze_to(result["scale"], 3, "scale")  # [F, H, 2]
        result["pi"]    = _squeeze_to(result["pi"],    1, "pi")      # [F]

        # ── sanity assert (cheap, catches future regressions) ──────────
        for key, expected in _EXPECTED.items():
            got = tuple(result[key].shape)
            assert got == expected, (
                f"TeacherStore: seq_id={seq_id} key={key} "
                f"shape {got} != expected {expected}"
            )

        # ── LRU-style in-memory cache ──────────────────────────────────
        if self._cache_size > 0:
            if len(self._cache) >= self._cache_size:
                # evict oldest
                self._cache.pop(next(iter(self._cache)))
            self._cache[seq_id] = result

        return result

    # ------------------------------------------------------------------ #
    def _load_raw(self, seq_id: str) -> dict:
        if self._is_h5:
            grp = self._h5[seq_id]          # raises KeyError if missing
            return {
                "loc":   torch.from_numpy(grp["loc"][:].astype(np.float32)),
                "scale": torch.from_numpy(grp["scale"][:].astype(np.float32)),
                "pi":    torch.from_numpy(grp["pi"][:].astype(np.float32)),
            }
        else:
            pt_path = self.path / f"{seq_id}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(pt_path)
            data = torch.load(pt_path, map_location="cpu")
            return {
                "loc":   data["loc"].float(),
                "scale": data["scale"].float(),
                "pi":    data["pi"].float(),
            }

    # ------------------------------------------------------------------ #
    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
