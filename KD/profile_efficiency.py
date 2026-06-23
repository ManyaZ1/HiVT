"""profile_efficiency.py

Efficiency / cost axis for the KD thesis: parameter count, peak GPU memory and
inference latency for HiVT at several embedding widths (teacher 128, student 32,
smaller student 16). These are ARCHITECTURE-determined — they do NOT depend on
trained weights — so this runs without any checkpoint, and the dim-16 numbers are
available before that student is ever trained.

The accuracy / calibration axis is produced separately (eval.py /
eval_lambda_sweep.py) and overlaid against the param counts from here to draw the
"accuracy & calibration vs model size" frontier.

Usage:
    # params only (instant, CPU, no data needed):
    python -m KD.profile_efficiency --params_only

    # full: params + latency + peak GPU memory on real val batches:
    python -m KD.profile_efficiency --root /home/manya/argoverse \
        --batch_size 32 --n_batches 20 --warmup 5 --out kd_ckpt/_triage_logs/efficiency.json

Latency is reported per BATCH and per SCENE (batch / n_graphs) at the chosen
batch size — report the batch size alongside the number, latency is not
batch-size invariant.
"""
import json
import os
import time
from argparse import ArgumentParser

import torch

from models.hivt import HiVT

# Canonical HiVT-Argoverse config (train.py / HiVT.add_model_specific_args
# defaults); only embed_dim varies across the rows of the efficiency table.
BASE_CFG = dict(
    historical_steps=20, future_steps=30, num_modes=6, rotate=True,
    node_dim=2, edge_dim=2, num_heads=8, dropout=0.1,
    num_temporal_layers=4, num_global_layers=3, local_radius=50.0,
    parallel=True, lr=5e-4, weight_decay=1e-4, T_max=64,
)

# (label, embed_dim). 128 = teacher, 32 = current student, 16 = smaller student.
WIDTHS = [("teacher-128", 128), ("student-32", 32), ("student-16", 16)]


def count_params(model):
    """Total params + a coarse breakdown over the three HiVT sub-modules."""
    def n(m):
        return sum(p.numel() for p in m.parameters())
    return {
        "total": n(model),
        "local_encoder": n(model.local_encoder),
        "global_interactor": n(model.global_interactor),
        "decoder": n(model.decoder),
    }


@torch.no_grad()
def measure_latency(model, batch, device, n_batches, warmup):
    """Median/mean forward latency (ms) over repeated runs of one real batch,
    plus peak allocated GPU memory (MB). Single fixed batch keeps the timing
    free of dataloader jitter; report alongside the batch's scene count."""
    model.eval().to(device)
    batch = batch.to(device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(batch)
    if cuda:
        torch.cuda.synchronize(device)

    times = []
    for _ in range(n_batches):
        if cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(batch)
        if cuda:
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    times.sort()
    peak_mb = (torch.cuda.max_memory_allocated(device) / 2**20) if cuda else None
    return {
        "batch_ms_median": times[len(times) // 2],
        "batch_ms_mean": sum(times) / len(times),
        "batch_ms_min": times[0],
        "peak_mem_mb": peak_mb,
    }


def main():
    ap = ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="Argoverse root; required unless --params_only.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--params_only", action="store_true",
                    help="Skip latency/memory; just count params (no data, CPU).")
    ap.add_argument("--out", default="kd_ckpt/_triage_logs/efficiency.json")
    args = ap.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    batch = None
    if not args.params_only:
        assert args.root is not None, "--root required unless --params_only"
        from torch_geometric.loader import DataLoader
        from datasets import ArgoverseV1Dataset
        ds = ArgoverseV1Dataset(root=args.root, split="val",
                                local_radius=BASE_CFG["local_radius"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        n_scenes = int(batch.num_graphs)
        print(f"[profile] device={device}  batch_size={args.batch_size}  "
              f"scenes_in_batch={n_scenes}")

    results = {}
    for label, dim in WIDTHS:
        cfg = dict(BASE_CFG, embed_dim=dim)
        model = HiVT(**cfg)
        row = {"embed_dim": dim, "params": count_params(model)}
        if not args.params_only:
            lat = measure_latency(model, batch, device, args.n_batches, args.warmup)
            row["latency"] = lat
            row["batch_size"] = args.batch_size
            row["scenes_in_batch"] = int(batch.num_graphs)
            row["latency"]["scene_ms_median"] = lat["batch_ms_median"] / batch.num_graphs
        results[label] = row
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Pretty table to stdout.
    tp = results["teacher-128"]["params"]["total"]
    print(f"\n{'model':<14}{'embed':>6}{'params':>12}{'rel%':>8}", end="")
    if not args.params_only:
        print(f"{'batch_ms':>11}{'scene_ms':>10}{'peak_MB':>10}", end="")
    print()
    for label, dim in WIDTHS:
        r = results[label]
        p = r["params"]["total"]
        print(f"{label:<14}{dim:>6}{p:>12,}{100.0 * p / tp:>7.1f}%", end="")
        if not args.params_only and "latency" in r:
            lat = r["latency"]
            sm = lat.get("scene_ms_median")
            pm = lat.get("peak_mem_mb")
            print(f"{lat['batch_ms_median']:>11.2f}"
                  f"{(sm if sm is not None else 0):>10.3f}"
                  f"{(pm if pm is not None else 0):>10.1f}", end="")
        print()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()