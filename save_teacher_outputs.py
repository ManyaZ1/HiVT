"""
save_teacher_outputs.py

Run HiVT-128 (teacher) over the full Argoverse training split and save
per-scene decoder outputs:
    loc   [F, H, 2]  — Laplace means
    scale [F, H, 2]  — Laplace scales
    pi    [F]        — raw mode logits (pre-softmax)

Saved as:  <output_dir>/<seq_id>.pt
Also writes:
    <output_dir>/manifest.json      — provenance record
    <output_dir>/sanity_stats.json  — distribution health check

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
import json
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Adjust these imports to match your repo layout
# --------------------------------------------------------------------------- #
from models.hivt import HiVT
from datasets import ArgoverseV1Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--split",       default="train", choices=["train", "val", "test"])
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sanity_samples", type=int, default=1000,
                   help="Number of saved files to sample for the sanity check")
    return p.parse_args()


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def sanity_check(out_dir: Path, n_samples: int) -> dict:
    """
    Load a random subset of saved files and compute basic distribution stats.
    Catches NaNs, collapsed scales, and degenerate pi distributions early.
    """
    all_files = list(out_dir.glob("*.pt"))
    sample = random.sample(all_files, min(n_samples, len(all_files)))

    scale_vals, pi_entropy_vals, loc_vals = [], [], []
    nan_files = []

    for path in tqdm(sample, desc="Sanity check"):
        data = torch.load(path, map_location="cpu")
        loc   = data["loc"]    # [F, H, 2]
        scale = data["scale"]  # [F, H, 2]
        pi    = data["pi"]     # [F]

        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(pi).any():
            nan_files.append(str(path.name))
            continue

        scale_vals.append(scale.mean().item())
        loc_vals.append(loc.abs().mean().item())

        # Entropy of the mode distribution (nats)
        probs   = torch.softmax(pi, dim=0)
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        pi_entropy_vals.append(entropy)

    def safe_stats(vals):
        if not vals:
            return {}
        t = torch.tensor(vals)
        return {
            "mean":  t.mean().item(),
            "std":   t.std().item(),
            "min":   t.min().item(),
            "max":   t.max().item(),
        }

    return {
        "files_sampled":    len(sample),
        "nan_files_found":  len(nan_files),
        "nan_file_names":   nan_files[:20],   # cap list length
        "scale_stats":      safe_stats(scale_vals),
        "loc_abs_stats":    safe_stats(loc_vals),
        "pi_entropy_stats": safe_stats(pi_entropy_vals),
    }


@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load teacher
    # ------------------------------------------------------------------ #
    print(f"Loading teacher from {args.checkpoint} ...")
    model = HiVT.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    param_counts = count_parameters(model)
    print(f"Teacher parameters — total: {param_counts['total']:,}  "
          f"trainable: {param_counts['trainable']:,}")

    # ------------------------------------------------------------------ #
    # 2. Dataloader — no shuffle, no drop_last
    # ------------------------------------------------------------------ #
    dataset = ArgoverseV1Dataset(root=args.data_root, split=args.split)
    from torch_geometric.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    print(f"Dataset: {len(dataset)} scenes | Batches: {len(loader)}")

    # ------------------------------------------------------------------ #
    # 3. Inference + save
    # ------------------------------------------------------------------ #
    saved = skipped = 0
    output_shape = None   # record once for the manifest

    for batch in tqdm(loader, desc="Saving teacher outputs"):
        batch = batch.to(args.device)
        pred, pi = model(batch)

        loc   = pred[..., :2].detach().cpu()   # [F, N, H, 2]
        scale = pred[..., 2:].detach().cpu()   # [F, N, H, 2]
        pi    = pi.detach().cpu()              # [N, F]

        seq_ids = batch.seq_id
        N = len(seq_ids)

        for i in range(N):
            out_path = out_dir / f"{seq_ids[i]}.pt"

            if out_path.exists():
                skipped += 1
                continue

            scene_loc   = loc[:, i, :, :]    # [F, H, 2]
            scene_scale = scale[:, i, :, :]  # [F, H, 2]
            scene_pi    = pi[i, :]            # [F]

            torch.save(
                {
                    "seq_id": seq_ids[i],
                    "loc":    scene_loc,
                    "scale":  scene_scale,
                    "pi":     scene_pi,
                },
                out_path,
            )

            if output_shape is None:
                output_shape = {
                    "loc":   list(scene_loc.shape),
                    "scale": list(scene_scale.shape),
                    "pi":    list(scene_pi.shape),
                }

            saved += 1

    print(f"\nSaved: {saved}  |  Skipped (existed): {skipped}")

    # ------------------------------------------------------------------ #
    # 4. Write manifest — permanent provenance record
    # ------------------------------------------------------------------ #
    manifest = {
        "created_at":      datetime.utcnow().isoformat() + "Z",
        "checkpoint":      str(Path(args.checkpoint).resolve()),
        "data_root":       str(Path(args.data_root).resolve()),
        "split":           args.split,
        "device":          args.device,
        "total_scenes":    saved + skipped,
        "scenes_saved":    saved,
        "scenes_skipped":  skipped,
        "output_dir":      str(out_dir.resolve()),
        "output_shapes":   output_shape,
        "teacher_params":  param_counts,
        #
        # Fill in manually or read from your checkpoint's hparams:
        # "teacher_embed_dim": 128,
        # "teacher_num_modes": 6,
        # "teacher_future_steps": 30,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written → {manifest_path}")

    # ------------------------------------------------------------------ #
    # 5. Sanity check
    # ------------------------------------------------------------------ #
    print(f"\nRunning sanity check on {args.sanity_samples} random files ...")
    stats = sanity_check(out_dir, args.sanity_samples)
    stats_path = out_dir / "sanity_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    if stats["nan_files_found"] > 0:
        print(f"\nWARNING: {stats['nan_files_found']} files contain NaNs. "
              f"Check the model checkpoint before proceeding.")
    else:
        print("\nSanity check passed — no NaNs found.")

    print(f"Stats written → {stats_path}")


if __name__ == "__main__":
    main()

# --checkpoint /home/manyazog/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt 
# --data_root /home/manyazog/argoverse
# #  --output_dir /home/manyazog/HiVT/teacher_outputs/train