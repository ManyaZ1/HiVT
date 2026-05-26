#save_teacher_outputs.py
"""
save_teacher_outputs.py

Run HiVT-128 (teacher) over the full Argoverse training split and save
per-scene decoder outputs:
    - loc   [F, N, H, 2]  — Laplace means
    - scale [F, N, H, 2]  — Laplace scales
    - pi    [N, F]         — raw mode logits (pre-softmax)

Saved as:  <output_dir>/<seq_id>.pt
Usage:
    python save_teacher_outputs.py \
        --checkpoint checkpoints/hivt128.ckpt \
        --data_root   data/argoverse/motion-forecasting \
        --output_dir  teacher_outputs/train \
        --split       train \
        --batch_size  32 \
        --num_workers 4
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Adjust these imports to match your repo's module layout
# --------------------------------------------------------------------------- #
from models.hivt import HiVT                          # your LightningModule
from datasets import ArgoverseV1Dataset                # your dataset class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,  help="Path to HiVT-128 .ckpt file")
    p.add_argument("--data_root",    required=True,  help="Root dir of Argoverse dataset")
    p.add_argument("--output_dir",   required=True,  help="Where to write .pt files")
    p.add_argument("--split",        default="train", choices=["train", "val", "test"])
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load the teacher model
    # ------------------------------------------------------------------ #
    print(f"Loading teacher from {args.checkpoint} ...")
    model = HiVT.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    # ------------------------------------------------------------------ #
    # 2. Build the dataloader — NO shuffle, NO drop_last
    #    We need every scene and we need stable ordering for seq_id lookup
    # ------------------------------------------------------------------ #
    dataset = ArgoverseV1Dataset(
        root=args.data_root,
        split=args.split,
        # Pass any preprocessing flags your dataset requires, e.g.:
        # local_radius=50, transform=...
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          # <-- critical: preserves index → seq_id mapping
        drop_last=False,        # <-- critical: save ALL scenes
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    print(f"Dataset size: {len(dataset)} scenes | Batches: {len(loader)}")

    # ------------------------------------------------------------------ #
    # 3. Inference loop
    # ------------------------------------------------------------------ #
    saved = 0
    skipped = 0

    for batch in tqdm(loader, desc="Saving teacher outputs"):
        # Move graph data to device
        batch = batch.to(args.device)

        # Forward pass through the full model.
        # HiVT's forward() returns (pred, pi) from the decoder:
        #   pred : [F, N, H, 4]  (loc + scale concatenated on last dim)
        #   pi   : [N, F]        (raw logits, NOT softmax)
        pred, pi = model(batch)

        # Split pred into loc and scale
        loc   = pred[..., :2]   # [F, N, H, 2]
        scale = pred[..., 2:]   # [F, N, H, 2]

        # Detach and move to CPU before saving
        loc   = loc.detach().cpu()
        scale = scale.detach().cpu()
        pi    = pi.detach().cpu()

        # ---------------------------------------------------------------- #
        # 4. Save per-scene
        #    batch.seq_id is a list of strings, one per agent in the batch.
        #    N here is the number of *target agents* (focal agents) in the
        #    batch.  Adjust indexing if your batching strategy differs.
        # ---------------------------------------------------------------- #
        seq_ids = batch.seq_id          # list of str, length N
        N = len(seq_ids)

        for i in range(N):
            out_path = out_dir / f"{seq_ids[i]}.pt"

            # Skip already-saved scenes (useful for resuming interrupted runs)
            if out_path.exists():
                skipped += 1
                continue

            torch.save(
                {
                    "seq_id": seq_ids[i],
                    # Shape comments assume single-scene slice (N=1 after indexing)
                    "loc":    loc[:, i, :, :],     # [F, H, 2]
                    "scale":  scale[:, i, :, :],   # [F, H, 2]
                    "pi":     pi[i, :],             # [F]   — raw logits
                },
                out_path,
            )
            saved += 1

    print(f"\nDone.  Saved: {saved}  |  Skipped (already existed): {skipped}")
    print(f"Output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()