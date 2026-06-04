"""
diagnose_shapes.py
==================
Run this BEFORE training to verify that every tensor in the KD pipeline
has the expected shape at every stage:

  1. Raw .h5 cache  — what TeacherStore returns
  2. KDDataset single item  — what __getitem__ attaches to Data
  3. PyG DataLoader batch — what the collator produces
  4. kd_loss alignment — whether teacher/student shapes are compatible

Usage
-----
(hivt_new) manya@DESKTOP-SBMUF07:~/HiVT$ 
python -m response_kd_fix.diagnose_shapes  --teacher_h5  teacher_outputs/train.h5         --data_root   /home/m
anya/argoverse         --batch_size  4


Expected output (all PASS):
    [1] h5 raw load       loc [6,30,2]  scale [6,30,2]  pi [6]  → PASS
    [2] KDDataset item    teacher_loc [6,30,2]           → PASS
    [3] Collated batch=4  teacher_loc [4,6,30,2]         → PASS
    [4] Loss input align  loc_t permuted [6,4,30,2]      → PASS
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.loader import DataLoader   # new PyG API

# ── project imports ──────────────────────────────────────────────────────────
# Adjust sys.path if you run from a different directory
sys.path.insert(0, str(Path(__file__).parent))

from datasets import ArgoverseV1Dataset          # noqa: E402
from kd_dataset import KDDataset                # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(condition, label, got, expected=None):
    status = PASS if condition else FAIL
    exp_str = f"  expected {expected}" if expected is not None else ""
    print(f"  [{status}] {label}: got {got}{exp_str}")
    return condition


# ────────────────────────────────────────────────────────────────────────────
# 1.  Raw HDF5 load
# ────────────────────────────────────────────────────────────────────────────
def check_h5_raw(h5_path: str):
    print("\n── [1] Raw HDF5 cache ──────────────────────────────────────────")
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"  {FAIL} file not found: {h5_path}")
        return False

    all_ok = True
    with h5py.File(h5_path, "r") as f:
        keys = [k for k in f.keys() if not k.startswith("_")]
        print(f"  Total scenes in cache: {len(keys)}")
        if not keys:
            print(f"  {FAIL} no scene keys found")
            return False

        # check first 3 entries
        for seq_id in keys[:3]:
            grp = f[seq_id]
            loc   = grp["loc"][:]    # numpy
            scale = grp["scale"][:]
            pi    = grp["pi"][:]
            ok_loc   = check(loc.shape   == (6, 30, 2), f"  {seq_id}/loc  shape",   loc.shape,   "(6,30,2)")
            ok_scale = check(scale.shape == (6, 30, 2), f"  {seq_id}/scale shape",  scale.shape, "(6,30,2)")
            ok_pi    = check(pi.shape    == (6,),        f"  {seq_id}/pi    shape",  pi.shape,    "(6,)")
            ok_nan   = check(not np.isnan(loc).any() and not np.isnan(scale).any(),
                             f"  {seq_id} no NaN",  "clean")
            all_ok = all_ok and ok_loc and ok_scale and ok_pi and ok_nan

    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# 2.  KDDataset single-item shapes
# ────────────────────────────────────────────────────────────────────────────
def check_dataset_item(data_root: str, h5_path: str):
    print("\n── [2] KDDataset single item ───────────────────────────────────")
    base    = ArgoverseV1Dataset(root=data_root, split="train")
    dataset = KDDataset(base, teacher_dir=h5_path)

    item = dataset[0]
    all_ok = True

    # teacher tensors must exist
    has_t = check(hasattr(item, "teacher_loc"),   "has teacher_loc",  hasattr(item, "teacher_loc"))
    all_ok = all_ok and has_t
    if not has_t:
        return False

    tl  = item.teacher_loc
    ts  = item.teacher_scale
    tp  = item.teacher_pi

    print(f"  teacher_loc.shape   = {tuple(tl.shape)}")
    print(f"  teacher_scale.shape = {tuple(ts.shape)}")
    print(f"  teacher_pi.shape    = {tuple(tp.shape)}")

    ok_loc   = check(tl.shape  == torch.Size([6, 30, 2]),  "teacher_loc shape",   tuple(tl.shape),  "[6,30,2]")
    ok_scale = check(ts.shape  == torch.Size([6, 30, 2]),  "teacher_scale shape", tuple(ts.shape),  "[6,30,2]")
    ok_pi    = check(tp.shape  == torch.Size([6]),          "teacher_pi shape",    tuple(tp.shape),  "[6]")
    ok_has_t = check(item.has_teacher.item() is True,              "has_teacher flag",    item.has_teacher)

    all_ok = all_ok and ok_loc and ok_scale and ok_pi and ok_has_t

    # Extra squeeze-check: warn if there's an unintended leading-1 dim
    if tl.dim() == 4 and tl.shape[0] == 1:
        print(f"  \033[93mWARN\033[0m teacher_loc has unexpected leading dim 1 → shape {tuple(tl.shape)}")
        print(f"       This is the known bug. Fix: squeeze in TeacherStore or KDDataset.__getitem__.")
        all_ok = False

    return all_ok


# ────────────────────────────────────────────────────────────────────────────
# 3.  PyG DataLoader collation
# ────────────────────────────────────────────────────────────────────────────
def check_collation(data_root: str, h5_path: str, batch_size: int = 4):
    print(f"\n── [3] DataLoader collation (batch_size={batch_size}) ──────────")
    base    = ArgoverseV1Dataset(root=data_root, split="train")
    dataset = KDDataset(base, teacher_dir=h5_path)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch   = next(iter(loader))

    tl = batch.teacher_loc
    ts = batch.teacher_scale
    tp = batch.teacher_pi

    print(f"  batch.teacher_loc.shape   = {tuple(tl.shape)}")
    print(f"  batch.teacher_scale.shape = {tuple(ts.shape)}")
    print(f"  batch.teacher_pi.shape    = {tuple(tp.shape)}")

    expected_loc = torch.Size([batch_size, 6, 30, 2])
    expected_pi  = torch.Size([batch_size, 6])

    ok_loc  = check(tl.shape == expected_loc, "teacher_loc batched", tuple(tl.shape), list(expected_loc))
    ok_pi   = check(tp.shape == expected_pi,  "teacher_pi batched",  tuple(tp.shape), list(expected_pi))
    
    # Check whether a permute is needed to match HiVT [F, N, H, 2] convention
    # HiVT decoder: [F, N_total, H, 2]  with N_total = sum of agents across graphs
    # teacher stored: per focal-agent, so N_total == batch_size for focal-only cache
    if tl.shape == expected_loc:
        permuted = tl.permute(1, 0, 2, 3)  # → [6, B, 30, 2]
        print(f"  After permute(1,0,2,3): {tuple(permuted.shape)}  ← matches [F, N, H, 2] for focal-agent KD")
        check(True, "permute gives [F,B,H,2]", tuple(permuted.shape), f"[6,{batch_size},30,2]")
    # Verify no spurious leading-1 on single items (unsqueeze hack not present)
    item = dataset[0]
    assert item.teacher_loc.shape == torch.Size([6, 30, 2]), \
    f"Single item has wrong shape {item.teacher_loc.shape} — remove unsqueeze from kd_dataset.py"
    return ok_loc and ok_pi


# ────────────────────────────────────────────────────────────────────────────
# 4.  kd_loss forward-pass dry run (no real model needed)
# ────────────────────────────────────────────────────────────────────────────
def check_loss_forward(batch_size: int = 4):
    print(f"\n── [4] kd_loss forward pass (synthetic tensors, B={batch_size}) ─")
    from kd_loss import HiVTKDLoss

    F, H = 6, 30
    agents_per_scene = 20
    N_total = agents_per_scene * batch_size

    # Full decoder output — all agents
    loc_pred   = torch.randn(F, N_total, H, 2)
    scale_pred = torch.rand(F, N_total, H, 2).abs() + 1e-3
    pi_pred    = torch.randn(N_total, F)

    # Focal agent indices — one per scene
    focal_idx = torch.arange(batch_size) * agents_per_scene  # [B]

    # Slice to focal agents
    loc_s   = loc_pred[:, focal_idx, :, :]    # [F, B, H, 2]
    scale_s = scale_pred[:, focal_idx, :, :]  # [F, B, H, 2]
    pi_s    = pi_pred[focal_idx, :]            # [B, F]

    # Teacher: collated [B,F,H,2] → permute → [F,B,H,2]
    loc_t_collated   = torch.randn(batch_size, F, H, 2)
    scale_t_collated = torch.rand(batch_size, F, H, 2).abs() + 1e-3
    pi_t             = torch.randn(batch_size, F)           # [B,F] — no permute needed

    loc_t   = loc_t_collated.permute(1, 0, 2, 3)    # [F, B, H, 2]
    scale_t = scale_t_collated.permute(1, 0, 2, 3)  # [F, B, H, 2]

    print(f"  student loc:  {tuple(loc_s.shape)}   teacher loc: {tuple(loc_t.shape)}")
    print(f"  student pi:   {tuple(pi_s.shape)}     teacher pi:  {tuple(pi_t.shape)}")

    assert loc_s.shape == loc_t.shape, f"loc mismatch: {loc_s.shape} vs {loc_t.shape}"
    assert pi_s.shape  == pi_t.shape,  f"pi mismatch:  {pi_s.shape} vs {pi_t.shape}"

    loss_fn = HiVTKDLoss(lambda_kl=0.5, lambda_pi=0.5)
    try:
        total, metrics = loss_fn(loc_s, scale_s, pi_s, loc_t, scale_t, pi_t)
        ok = check(total.isfinite(), "loss is finite", f"{total.item():.4f}")
        check(not torch.isnan(total), "loss not NaN", f"{total.item():.4f}")
        print(f"  kd/kl_laplace = {metrics['kd/kl_laplace']:.4f}")
        print(f"  kd/pi_ce      = {metrics['kd/pi_ce']:.4f}")
        print(f"  kd/total      = {metrics['kd/total']:.4f}")
        return ok
    except Exception as e:
        print(f"  {FAIL} kd_loss raised: {e}")
        import traceback; traceback.print_exc()
        return False

# ────────────────────────────────────────────────────────────────────────────
# 5.  Shape mismatch between teacher-cache convention and loss convention
# ────────────────────────────────────────────────────────────────────────────
def check_convention_summary():
    print("\n── [5] Convention summary ──────────────────────────────────────")
    rows = [
        ("Teacher cache (per scene)",        "[6, 30, 2]",     "[6, 30, 2]",    "[6]"),
        ("KDDataset item (expected)",         "[6, 30, 2]",     "[6, 30, 2]",    "[6]"),
        ("Collated batch B=4 (expected)",     "[4, 6, 30, 2]",  "[4, 6, 30, 2]", "[4, 6]"),
        ("After permute(1,0,2,3) B=4",        "[6, 4, 30, 2]",  "[6, 4, 30, 2]", "[4, 6]"),
        ("HiVT decoder output [F,N,H,2]",     "[6, N, 30, 2]",  "[6, N, 30, 2]", "[N, 6]"),
        ("kd_loss expects",                   "[F, N, H, 2]",   "[F, N, H, 2]",  "[N, F]"),
    ]
    print(f"  {'Source':<42} {'loc':<18} {'scale':<18} {'pi'}")
    print(f"  {'-'*42} {'-'*18} {'-'*18} {'-'*12}")
    for r in rows:
        print(f"  {r[0]:<42} {r[1]:<18} {r[2]:<18} {r[3]}")

    print("""
  KEY RULE for hivt_kd.py training_step:
    teacher tensors from batch are [B, F, H, 2] / [B, F]
    → permute loc/scale to [F, B, H, 2]  before passing to kd_loss
    → permute pi from [B, F] to [B, F]   (already correct)

    student loc/scale from HiVT decoder is [F, N_total, H, 2]
    → slice out focal agents: student_focal = loc_s[:, batch.agent_index, :, :]
      where agent_index marks the focal agent for each scene in the batch
    → result: [F, B, H, 2]  — now both are aligned
  """)


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_h5",  required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--batch_size",  type=int, default=4)
    args = parser.parse_args()

    results = []
    results.append(check_h5_raw(args.teacher_h5))
    results.append(check_dataset_item(args.data_root, args.teacher_h5))
    results.append(check_collation(args.data_root, args.teacher_h5, args.batch_size))
    results.append(check_loss_forward(args.batch_size))
    check_convention_summary()

    print("\n── Summary ─────────────────────────────────────────────────────")
    labels = ["h5 raw", "dataset item", "collation", "loss fwd"]
    for label, ok in zip(labels, results):
        print(f"  {PASS if ok else FAIL}  {label}")

    if not all(results):
        print("\n  Fix the FAILs above before running smoke_train.py")
        sys.exit(1)
    else:
        print("\n  All checks passed. Run smoke_train.py next.")


if __name__ == "__main__":
    main()
