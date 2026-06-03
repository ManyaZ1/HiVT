# check_collation.py
import torch
from torch_geometric.data import DataLoader
from datasets import ArgoverseV1Dataset
from kd_dataset import KDDataset

base = ArgoverseV1Dataset(root="data/argoverse", split="train")
dataset = KDDataset(base, teacher_dir="teacher_outputs/train.h5")

# grab 2 raw items before collation
item0 = dataset[0]
item1 = dataset[1]
print("Single item teacher_loc shape:", item0.teacher_loc.shape)  # expect [6, 30, 2]

# collate into a batch of 2
loader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(loader))
print("Batched teacher_loc shape:    ", batch.teacher_loc.shape)   # [2, 6, 30, 2] or [6, 2, 30, 2]?

# student output convention — what HiVT decoder produces
# pred[..., :2] has shape [F, N, H, 2] = [6, 2, 30, 2]
print()
print("Match?", batch.teacher_loc.shape == torch.Size([6, 2, 30, 2]))
print("Needs permute?", batch.teacher_loc.shape == torch.Size([2, 6, 30, 2]))