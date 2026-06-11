"""
save_teacher_outputs.py

Run HiVT-128 (teacher) over the full Argoverse training split and save the
focal-agent decoder outputs DIRECTLY into a single HDF5 file (no intermediate
per-scene .pt files — those waste a lot of disk in filesystem slack + inodes).

Per scene we store, under a group named by the normalized seq_id:
    loc   [F, H, 2]  — Laplace means   (focal agent, rotated-local frame)
    scale [F, H, 2]  — Laplace scales
    pi    [F]        — raw mode logits (pre-softmax)

This is exactly the layout KD/kd_teacher_store.py reads: hf[seq_id]["loc"|"scale"|"pi"].

Also writes alongside the .h5:
    <output_h5>.manifest.json      — provenance record
    <output_h5>.sanity_stats.json  — distribution health check

Usage:
    python save_teacher_ouputs.py \
        --checkpoint checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt \
        --data_root  /home/manya/argoverse \
        --output_h5  teacher_outputs/train_fix.h5 \
        --split      train \
        --batch_size 32 \
        --num_workers 4
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
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
    p.add_argument("--output_h5",   required=True,
                   help="Path to the single HDF5 file to write (e.g. teacher_outputs/train_fix.h5)")
    p.add_argument("--split",       default="train", choices=["train", "val", "test"])
    p.add_argument("--local_radius", type=float, default=50.0,
                   help="Must match the student's local_radius so frames are consistent")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gzip",        type=int, default=4,
                   help="gzip level 0-9 for the HDF5 datasets (0 disables compression)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sanity_samples", type=int, default=1000,
                   help="Number of saved groups to sample for the sanity check")
    return p.parse_args()


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def normalize_seq_id(seq_id) -> str:
    if isinstance(seq_id, torch.Tensor):
        if seq_id.numel() == 1:
            seq_id = seq_id.item()
        else:
            seq_id = seq_id.tolist()
    if isinstance(seq_id, (list, tuple)) and len(seq_id) == 1:
        seq_id = seq_id[0]
    return str(seq_id)


def sanity_check(h5_path: Path, n_samples: int) -> dict:
    """
    Load a random subset of saved groups and compute basic distribution stats.
    Catches NaNs, collapsed scales, and degenerate pi distributions early.
    """
    scale_vals, pi_entropy_vals, loc_vals = [], [], []
    nan_keys = []

    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
        sample = random.sample(keys, min(n_samples, len(keys)))

        for key in tqdm(sample, desc="Sanity check"):
            grp   = hf[key]
            loc   = torch.from_numpy(grp["loc"][:])    # [F, H, 2]
            scale = torch.from_numpy(grp["scale"][:])  # [F, H, 2]
            pi    = torch.from_numpy(grp["pi"][:])     # [F]

            if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(pi).any():
                nan_keys.append(key)
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
        "nan_keys_found":   len(nan_keys),
        "nan_key_names":    nan_keys[:20],   # cap list length
        "scale_stats":      safe_stats(scale_vals),
        "loc_abs_stats":    safe_stats(loc_vals),
        "pi_entropy_stats": safe_stats(pi_entropy_vals),
    }


@torch.no_grad()
def main():
    args = parse_args()
    out_h5 = Path(args.output_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load teacher
    # ------------------------------------------------------------------ #
    print(f"Loading teacher from {args.checkpoint} ...")
    # PyTorch 2.6+ defaults torch.load(weights_only=True), which rejects the
    # Lightning checkpoint (it pickles a ModelCheckpoint callback). This is our
    # own trusted local checkpoint, so allow full unpickling.
    import functools
    torch.load = functools.partial(torch.load, weights_only=False)
    model = HiVT.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    param_counts = count_parameters(model)
    print(f"Teacher parameters — total: {param_counts['total']:,}  "
          f"trainable: {param_counts['trainable']:,}")

    # ------------------------------------------------------------------ #
    # 2. Dataloader — no shuffle, no drop_last
    # ------------------------------------------------------------------ #
    dataset = ArgoverseV1Dataset(root=args.data_root, split=args.split,
                                 local_radius=args.local_radius)
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
    # 3. Inference + save directly into a single HDF5 file
    #    Opened in "a" (append) mode so an interrupted run is resumable:
    #    groups already present are skipped.
    # ------------------------------------------------------------------ #
    saved = skipped = 0
    output_shape = None                          # record once for the manifest
    comp = dict(compression="gzip", compression_opts=args.gzip) if args.gzip > 0 else {}

    hf = h5py.File(out_h5, "a")
    try:
        for batch in tqdm(loader, desc="Saving teacher outputs"):
            batch = batch.to(args.device)
            pred, pi = model(batch)

            loc   = pred[..., :2].detach().cpu()   # [F, N, H, 2]  (all actors)
            scale = pred[..., 2:].detach().cpu()   # [F, N, H, 2]
            pi    = pi.detach().cpu()              # [N, F]

            seq_ids = batch.seq_id
            # HiVT emits a trajectory for EVERY actor in one forward pass; the
            # node dimension above stacks all scenes' actors together. The focal
            # agent of each scene is selected via batch.agent_index, which PyG has
            # already offset into batched node space. Indexing by the scene
            # position i instead — the previous bug — saved an arbitrary actor's
            # (mostly scene-0's) trajectory under each seq_id.
            agent_idx = batch.agent_index          # [num_scenes] global node indices
            N = len(seq_ids)

            for i in range(N):
                seq_id = normalize_seq_id(seq_ids[i])

                if seq_id in hf:                   # resume-safe skip
                    skipped += 1
                    continue

                node = int(agent_idx[i])           # focal agent of scene i
                scene_loc   = loc[:, node, :, :].numpy()    # [F, H, 2]
                scene_scale = scale[:, node, :, :].numpy()  # [F, H, 2]
                scene_pi    = pi[node, :].numpy()           # [F]

                grp = hf.create_group(seq_id)
                grp.create_dataset("loc",   data=scene_loc,   **comp)
                grp.create_dataset("scale", data=scene_scale, **comp)
                grp.create_dataset("pi",    data=scene_pi,    **comp)

                if output_shape is None:
                    output_shape = {
                        "loc":   list(scene_loc.shape),
                        "scale": list(scene_scale.shape),
                        "pi":    list(scene_pi.shape),
                    }

                saved += 1
    finally:
        hf.close()

    print(f"\nSaved: {saved}  |  Skipped (existed): {skipped}")

    # ------------------------------------------------------------------ #
    # 4. Write manifest — permanent provenance record
    # ------------------------------------------------------------------ #
    manifest = {
        "created_at":      datetime.utcnow().isoformat() + "Z",
        "checkpoint":      str(Path(args.checkpoint).resolve()),
        "data_root":       str(Path(args.data_root).resolve()),
        "split":           args.split,
        "local_radius":    args.local_radius,
        "device":          args.device,
        "total_scenes":    saved + skipped,
        "scenes_saved":    saved,
        "scenes_skipped":  skipped,
        "output_h5":       str(out_h5.resolve()),
        "output_shapes":   output_shape,
        "gzip":            args.gzip,
        "teacher_params":  param_counts,
    }
    manifest_path = out_h5.with_suffix(out_h5.suffix + ".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written → {manifest_path}")

    # ------------------------------------------------------------------ #
    # 5. Sanity check
    # ------------------------------------------------------------------ #
    print(f"\nRunning sanity check on {args.sanity_samples} random groups ...")
    stats = sanity_check(out_h5, args.sanity_samples)
    stats_path = out_h5.with_suffix(out_h5.suffix + ".sanity_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    if stats["nan_keys_found"] > 0:
        print(f"\nWARNING: {stats['nan_keys_found']} groups contain NaNs. "
              f"Check the model checkpoint before proceeding.")
    else:
        print("\nSanity check passed — no NaNs found.")

    print(f"Stats written → {stats_path}")


if __name__ == "__main__":
    main()
