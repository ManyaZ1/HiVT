"""
test.py — quick verification that the KD lane/actor batching is correct.

Run from the repo root with the project's env:

    /home/manya/miniconda3/envs/hivt_new/bin/python test.py
    # or, if that env is already active:
    python test.py

What it checks
--------------
KDData now subclasses utils.TemporalData (not plain PyG Data), so a batch of
KDData objects must collate `lane_actor_index` with TemporalData's asymmetric
per-row increment: row 0 (lane endpoints) shifts by the lane count, row 1 (actor
endpoints) by the node count. The definitive check is that the KD batch is
byte-identical to the base TemporalData batch.
"""

import sys
from pathlib import Path

# Bootstrap: put the repo root on sys.path so `from KD.kd_dataset import ...`
# and the repo packages (utils, datasets) all resolve absolutely.
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch_geometric.data import Batch

from datasets import ArgoverseV1Dataset
from KD.kd_dataset import KDDataset

DATA_ROOT   = "/home/manya/argoverse"
TEACHER_DIR = "teacher_outputs/train.h5"

base = ArgoverseV1Dataset(root=DATA_ROOT, split="train", local_radius=50)
kd_dataset = KDDataset(base, teacher_dir=TEACHER_DIR)

b_kd   = Batch.from_data_list([kd_dataset[0], kd_dataset[1]])
b_base = Batch.from_data_list([kd_dataset.base[0], kd_dataset.base[1]])

print("row0 max / lane count :", int(b_kd.lane_actor_index[0].max()), b_kd.lane_vectors.size(0))
differ = bool((b_kd.lane_actor_index != b_base.lane_actor_index).any())
print("kd vs base differ?    :", differ, "  (expect False after the fix)")
print("exact match?          :", torch.equal(b_kd.lane_actor_index, b_base.lane_actor_index))
print("has_teacher           :", b_kd.has_teacher.tolist())
print("teacher_loc shape     :", tuple(b_kd.teacher_loc.shape), "| teacher_pi:", tuple(b_kd.teacher_pi.shape))

assert not differ, "lane_actor_index batching diverges from base TemporalData — the fix regressed"
print("\nPASS: KDData batches lane_actor_index identically to base TemporalData.")
