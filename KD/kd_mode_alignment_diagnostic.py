"""
kd_mode_alignment_diagnostic.py

Empirically measures whether the teacher's mode index-k corresponds to the
student's mode index-k. If it does NOT (the expected result), this is the
justification figure for using a permutation-invariant KD loss instead of an
index-aligned one.

What it computes
----------------
For a handful of validation scenes, for the focal agent in each:
  1. Run teacher and student, get their K / F mode-mean trajectories.
  2. For each student mode, find the NEAREST teacher mode (min mean-L2 over H).
  3. Record (student_idx -> nearest_teacher_idx).

Two summaries:
  * "accidental alignment rate" = fraction of cases where the nearest teacher
    mode index equals the student mode index. For F modes, pure chance is 1/F
    (~0.167 for F=6). A value near chance => indices are unrelated => use a
    permutation-invariant loss.
  * an F x F frequency matrix M[s, t] = how often student mode s maps to
    teacher mode t, saved as a PNG heatmap. A near-uniform matrix => no
    correspondence; a strong diagonal => indices already align.

IMPORTANT — this script is a STARTING POINT, not tested against your repo.
Adjust the two blocks marked  # >>> ADJUST  to match how your project loads
checkpoints and builds the val dataloader. It assumes:
  * HiVT and HiVTKD expose `.load_from_checkpoint(path)`.
  * The model forward returns (pred, pi) with pred [M, N, H, 4] and an
    `agent_index` on the batch (same convention as training_step).
Run on CPU is fine; only a few scenes are needed.
"""

import argparse
import os
import sys

import numpy as np
import torch

# Run-from-anywhere bootstrap: ensure the repo root is on sys.path so the repo
# packages (models, datasets) and the KD package import absolutely. Run as
# `python -m KD.kd_mode_alignment_diagnostic`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.hivt import HiVT                 # repo module (absolute)
from datasets import ArgoverseV1Dataset      # repo module (absolute)
from KD.hivt_kd import HiVTKD                 # sibling KD module (absolute from repo root)
from torch_geometric.data import DataLoader as PyGDataLoader


@torch.no_grad()
def mode_means(model, data):
    """Return focal-agent mode-mean trajectories [M, H, 2] and logits [M]."""
    pred, pi = model(data)                       # [M, N, H, 4], [N, M]
    ai = data.agent_index                        # [B]; B==1 here
    loc = pred[:, ai, :, :2]                      # [M, B, H, 2]
    loc = loc[:, 0]                               # [M, H, 2]  (B==1)
    logits = pi[ai][0]                            # [M]
    return loc.cpu(), logits.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--root", required=True, help="Argoverse dataset root")
    ap.add_argument("--num_scenes", type=int, default=200)
    ap.add_argument("--out_png", default="mode_alignment.png")
    args = ap.parse_args()

    # >>> ADJUST: checkpoint loading to match your classes' signatures
    teacher = HiVT.load_from_checkpoint(args.teacher_ckpt).eval()
    student = HiVTKD.load_from_checkpoint(args.student_ckpt).eval()

    # >>> ADJUST: val dataset / radius to match your datamodule
    val = ArgoverseV1Dataset(root=args.root, split="val", local_radius=50)
    loader = PyGDataLoader(val, batch_size=1, shuffle=False, num_workers=4)

    F_modes = None
    align_hits, align_total = 0, 0
    freq = None

    for i, data in enumerate(loader):
        if i >= args.num_scenes:
            break
        t_loc, _ = mode_means(teacher, data)      # [F, H, 2]
        s_loc, _ = mode_means(student, data)      # [K, H, 2]

        if F_modes is None:
            F_modes = t_loc.shape[0]
            K_modes = s_loc.shape[0]
            freq = np.zeros((K_modes, F_modes), dtype=np.int64)

        # mean-L2 over horizon between every student and teacher mode
        # s_loc [K,H,2], t_loc [F,H,2] -> dist [K,F]
        diff = s_loc[:, None] - t_loc[None]       # [K, F, H, 2]
        dist = diff.norm(dim=-1).mean(dim=-1)     # [K, F]
        nearest_t = dist.argmin(dim=1)            # [K]

        for s_idx, t_idx in enumerate(nearest_t.tolist()):
            freq[s_idx, t_idx] += 1
            align_total += 1
            if s_idx == t_idx:
                align_hits += 1

    rate = align_hits / max(align_total, 1)
    chance = 1.0 / F_modes if F_modes else float("nan")
    print(f"scenes used                 : {min(i, args.num_scenes)}")
    print(f"accidental alignment rate   : {rate:.3f}")
    print(f"chance level (1/F)          : {chance:.3f}")
    print("student->nearest-teacher frequency matrix (rows=student, cols=teacher):")
    print(freq)
    if abs(rate - chance) < 0.05:
        print(">> Near chance: mode indices do NOT correspond across models. "
              "A permutation-invariant KD loss is justified.")
    else:
        print(">> Above chance: some index correspondence exists; report the "
              "matrix and decide whether matching is still worthwhile.")

    # Optional heatmap (requires matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(freq, cmap="viridis")
        ax.set_xlabel("teacher mode index")
        ax.set_ylabel("student mode index")
        ax.set_title(f"mode assignment freq (align={rate:.2f}, chance={chance:.2f})")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        print(f"saved heatmap -> {args.out_png}")
    except Exception as e:
        print(f"(heatmap skipped: {e})")


if __name__ == "__main__":
    main()
