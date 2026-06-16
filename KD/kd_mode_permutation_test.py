"""
kd_mode_permutation_test.py  —  Thesis Option 2: empirical premise test.

GOAL
----
Verify *without any training* the premise behind the permutation-invariant KD
loss: that a WTA-trained teacher and an independently trained student do NOT
share a mode-slot ordering, so index-aligned distillation (match teacher slot k
to student slot k) compares unrelated trajectories.

If the premise holds we expect, on the validation set:
  * the optimal teacher<->student mode permutation is almost never the identity;
  * pairing modes by index (identity) is much farther apart than the optimal
    matching;
  * the student->nearest-teacher assignment is ~uniform (near the 1/F chance
    level), i.e. the slot index carries no cross-model meaning.

This is the justification figure/table for the thesis claim. It runs inference
only (a few hundred val scenes is plenty); no gradients, no training.

WHAT IT COMPUTES  (focal agent of each scene)
---------------------------------------------
For teacher mode-means t_loc [F,H,2] and student mode-means s_loc [K,H,2], with
a cost matrix C[k,f] = endpoint-L2 (FDE-style) between student mode k and
teacher mode f:

  1. Greedy nearest  — each student mode's nearest teacher mode. Gives the
     student->teacher frequency matrix and the "accidental alignment rate"
     (how often nearest-teacher-index == student-index; chance = 1/F). Many
     student modes collapsing onto one teacher mode shows up here.

  2. Optimal permutation  — the one-to-one assignment minimising total cost
     (exact brute force over F! permutations; F is small, e.g. 6 -> 720).
     Reports how often the identity permutation is optimal, and the mean cost
     under identity vs. optimal pairing (the index-alignment penalty).

  3. Top-mode (rank-aware) check  — does the student's most-probable mode match
     (geometrically) the teacher's most-probable mode? Mixture models often DO
     learn a consistent dominant mode; reporting this separately keeps the
     argument honest (the *tail* is where alignment breaks down).

Outputs: prints a summary table, writes a JSON of all numbers and two PNG
heatmaps (greedy frequency + optimal-permutation frequency) under --out_dir.

USAGE
-----
  python -m KD.kd_mode_permutation_test \
      --teacher_ckpt checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt \
      --student_ckpt HiVT-32/gxhl2ug9/checkpoints/epoch=63-step=411903.ckpt \
      --root /home/manya/argoverse \
      --num_scenes 500

  # Use a KD-trained student instead of the from-scratch baseline:
  #   --student_ckpt kd_ckpt/.../best/last.ckpt  --student_type hivtkd
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

# Run-from-anywhere bootstrap: put repo root on sys.path so repo packages
# (models, datasets) and the KD package import absolutely. Run as
#   python -m KD.kd_mode_permutation_test
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.hivt import HiVT                 # repo module (absolute)
from datasets import ArgoverseV1Dataset      # repo module (absolute)
from KD.hivt_kd import HiVTKD                 # sibling KD module (absolute from repo root)
from torch_geometric.data import DataLoader as PyGDataLoader


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
# torch >=2.6 defaults torch.load(weights_only=True), which rejects the Lightning
# ModelCheckpoint objects pickled in these (locally trained, trusted) ckpts.
# Force the legacy full-unpickle path.
_torch_load = torch.load
def _full_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load(*a, **kw)
torch.load = _full_load


def load_model(ckpt, kind, device):
    """Load a HiVT (or HiVTKD-wrapped) model and return the inner HiVT in eval."""
    if kind == "hivt":
        model = HiVT.load_from_checkpoint(ckpt, map_location=device)
    elif kind == "hivtkd":
        model = HiVTKD.load_from_checkpoint(ckpt, map_location=device).student
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return model.to(device).eval()


@torch.no_grad()
def focal_modes(model, data):
    """Focal-agent mode-mean trajectories [M,H,2] and softmax mode probs [M].

    forward() returns y_hat [M,N,H,4] (loc[:2] + scale[2:]) and pi [N,M].
    Predictions do not depend on data.y, so running teacher then student on the
    same batch is safe — each forward rebuilds rotate_mat from rotate_angles.
    """
    y_hat, pi = model(data)                       # [M,N,H,4], [N,M]
    ai = data.agent_index                         # [B]; B == 1 here
    loc = y_hat[:, ai, :, :2][:, 0]               # [M,H,2]
    probs = torch.softmax(pi[ai][0], dim=-1)      # [M]
    return loc.cpu(), probs.cpu()


# --------------------------------------------------------------------------- #
# Assignment helpers
# --------------------------------------------------------------------------- #
def best_permutation(cost, perms):
    """Exact min-cost one-to-one assignment for a square cost matrix [n,n].

    perms: precomputed array [n!, n] of all permutations of range(n).
    Returns (perm [n], total_cost).
    """
    n = cost.shape[0]
    rows = np.arange(n)
    totals = cost[rows[None, :], perms].sum(axis=1)   # [n!]
    j = int(totals.argmin())
    return perms[j], float(totals[j])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--teacher_type", default="hivt", choices=["hivt", "hivtkd"])
    ap.add_argument("--student_type", default="hivt", choices=["hivt", "hivtkd"],
                    help="hivt for the from-scratch baseline; hivtkd for a KD-trained student")
    ap.add_argument("--root", required=True, help="Argoverse dataset root")
    ap.add_argument("--local_radius", type=float, default=50.0)
    ap.add_argument("--num_scenes", type=int, default=500)
    ap.add_argument("--cost", default="fde", choices=["fde", "ade"],
                    help="mode-distance cost: fde=endpoint L2, ade=mean-over-horizon L2")
    ap.add_argument("--out_dir", default=os.path.join(_REPO_ROOT, "KD", "diagnostics"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  cost: {args.cost}")

    teacher = load_model(args.teacher_ckpt, args.teacher_type, device)
    student = load_model(args.student_ckpt, args.student_type, device)

    val = ArgoverseV1Dataset(root=args.root, split="val", local_radius=args.local_radius)
    loader = PyGDataLoader(val, batch_size=1, shuffle=False, num_workers=4)

    F_modes = K_modes = perms = None
    greedy_freq = None            # [K,F] student -> nearest teacher
    perm_freq = None              # [K,F] student -> optimal-matched teacher
    align_hits = 0                # greedy nearest-index == student-index
    identity_optimal = 0          # optimal permutation == identity
    top_agree = 0                 # student top-prob mode geom-matches teacher top-prob mode
    cost_identity_sum = 0.0
    cost_optimal_sum = 0.0
    ratio_list = []
    n_used = 0

    for i, data in enumerate(loader):
        if i >= args.num_scenes:
            break
        data = data.to(device)
        t_loc, t_probs = focal_modes(teacher, data)   # [F,H,2], [F]
        s_loc, s_probs = focal_modes(student, data)   # [K,H,2], [K]

        if F_modes is None:
            F_modes, K_modes = t_loc.shape[0], s_loc.shape[0]
            greedy_freq = np.zeros((K_modes, F_modes), dtype=np.int64)
            perm_freq = np.zeros((K_modes, F_modes), dtype=np.int64)
            if K_modes == F_modes and K_modes <= 8:
                perms = np.array(list(itertools.permutations(range(K_modes))), dtype=np.int64)
            else:
                print(f"[warn] K={K_modes} != F={F_modes} (or >8): permutation "
                      f"metrics need a square matrix; only greedy stats reported.")

        # Cost matrix C[k,f] between student mode k and teacher mode f.
        if args.cost == "fde":
            diff = s_loc[:, None, -1] - t_loc[None, :, -1]    # [K,F,2]  endpoints
            C = diff.norm(dim=-1)                             # [K,F]
        else:  # ade
            diff = s_loc[:, None] - t_loc[None]               # [K,F,H,2]
            C = diff.norm(dim=-1).mean(dim=-1)                # [K,F]
        C = C.numpy()

        # --- (1) greedy nearest teacher for each student mode ---
        nearest = C.argmin(axis=1)                            # [K]
        for s_idx, t_idx in enumerate(nearest):
            greedy_freq[s_idx, t_idx] += 1
            if s_idx == t_idx:
                align_hits += 1

        # --- (3) top-mode rank-aware check ---
        s_top = int(s_probs.argmax())
        t_top = int(t_probs.argmax())
        if int(nearest[s_top]) == t_top:
            top_agree += 1

        # --- (2) optimal one-to-one permutation (square case only) ---
        if perms is not None:
            perm, cost_opt = best_permutation(C, perms)       # perm[k] = matched teacher
            for s_idx, t_idx in enumerate(perm):
                perm_freq[s_idx, int(t_idx)] += 1
            if np.array_equal(perm, np.arange(K_modes)):
                identity_optimal += 1
            cost_id = float(C[np.arange(K_modes), np.arange(K_modes)].mean())
            cost_op = cost_opt / K_modes
            cost_identity_sum += cost_id
            cost_optimal_sum += cost_op
            ratio_list.append(cost_id / cost_op if cost_op > 1e-9 else float("nan"))

        n_used += 1
        if n_used % 100 == 0:
            print(f"  ...{n_used} scenes")

    # ----------------------------------------------------------------------- #
    # Summary
    # ----------------------------------------------------------------------- #
    align_total = n_used * K_modes
    chance = 1.0 / F_modes
    align_rate = align_hits / max(align_total, 1)
    top_rate = top_agree / max(n_used, 1)

    summary = {
        "scenes": n_used,
        "teacher_modes_F": F_modes,
        "student_modes_K": K_modes,
        "cost": args.cost,
        "greedy_accidental_alignment_rate": align_rate,
        "chance_level_1_over_F": chance,
        "top_mode_geom_agreement_rate": top_rate,
    }
    if perms is not None:
        ratios = np.array([r for r in ratio_list if not np.isnan(r)])
        summary.update({
            "identity_is_optimal_rate": identity_optimal / max(n_used, 1),
            "mean_cost_identity_m": cost_identity_sum / max(n_used, 1),
            "mean_cost_optimal_m": cost_optimal_sum / max(n_used, 1),
            "identity_over_optimal_cost_ratio_mean": float(ratios.mean()),
            "identity_over_optimal_cost_ratio_median": float(np.median(ratios)),
        })

    print("\n================ MODE-PERMUTATION DIAGNOSTIC ================")
    print(f"scenes used                         : {n_used}")
    print(f"teacher modes F / student modes K   : {F_modes} / {K_modes}")
    print(f"greedy accidental alignment rate    : {align_rate:.3f}   (chance 1/F = {chance:.3f})")
    print(f"top-prob mode geometric agreement   : {top_rate:.3f}")
    if perms is not None:
        print(f"identity permutation is optimal in  : {summary['identity_is_optimal_rate']:.3f} of scenes")
        print(f"mean pairing distance  identity     : {summary['mean_cost_identity_m']:.3f} m")
        print(f"mean pairing distance  optimal      : {summary['mean_cost_optimal_m']:.3f} m")
        print(f"identity / optimal distance ratio   : {summary['identity_over_optimal_cost_ratio_mean']:.2f}x "
              f"(median {summary['identity_over_optimal_cost_ratio_median']:.2f}x)")
    print("\ninterpretation:")
    print("  alignment rate ~ chance, identity rarely optimal, and identity")
    print("  distance >> optimal distance  =>  mode slots do NOT correspond")
    print("  across the two models  =>  index-aligned KD is ill-posed; a")
    print("  permutation-invariant loss is justified.")
    print("============================================================\n")

    tag = f"{args.cost}_n{n_used}"
    json_path = os.path.join(args.out_dir, f"mode_perm_{tag}.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary,
                   "greedy_freq": greedy_freq.tolist(),
                   "perm_freq": perm_freq.tolist() if perms is not None else None},
                  f, indent=2)
    print(f"saved numbers -> {json_path}")

    # Heatmaps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def heatmap(mat, title, path):
            fig, ax = plt.subplots(figsize=(4.2, 4))
            im = ax.imshow(mat, cmap="viridis")
            ax.set_xlabel("teacher mode index")
            ax.set_ylabel("student mode index")
            ax.set_xticks(range(mat.shape[1]))
            ax.set_yticks(range(mat.shape[0]))
            ax.set_title(title)
            for (r, c), v in np.ndenumerate(mat):
                ax.text(c, r, str(int(v)), ha="center", va="center",
                        color="w" if v < mat.max() * 0.6 else "k", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"saved heatmap -> {path}")

        heatmap(greedy_freq,
                f"student->nearest teacher\n(align={align_rate:.2f}, chance={chance:.2f})",
                os.path.join(args.out_dir, f"mode_perm_greedy_{tag}.png"))
        if perms is not None:
            heatmap(perm_freq,
                    f"optimal-permutation assignments\n(identity optimal={summary['identity_is_optimal_rate']:.2f})",
                    os.path.join(args.out_dir, f"mode_perm_optimal_{tag}.png"))
    except Exception as e:
        print(f"(heatmaps skipped: {e})")


if __name__ == "__main__":
    main()
