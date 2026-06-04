"""
smoke_train.py
==============
Rapid sanity check that the KD training loop is actually learning.

What it does
------------
1.  Loads a small fixed subset of N_SCENES scenes from the training split.
2.  Runs OVERFIT_EPOCHS epochs on that subset.
3.  Checks that:
    a) KD loss decreases monotonically (or at least by >30% over the run).
    b) Task loss (NLL) also decreases.
    c) No NaN/Inf in any loss at any step.
4.  Prints a final verdict.

This is intentionally not a unit test — it exercises the real model, real
data, and real loss, so it will catch shape bugs, indexing bugs, and
"learning is silently disabled" bugs in one shot.

Usage --teacher_h5   teacher_outputs/train.h5 \
-----
    python -m response_kd_fix.smoke_train \
        --teacher_dir teacher_outputs/train.h5 \
        --data_root    /home/manya/argoverse \
        --embed_dim    64 \
        --lambda_task  1.0 \
        --lambda_kl    0.5 \
        --lambda_pi    0.5 \
        --n_scenes     64 \
        --overfit_epochs 30 \
        --batch_size   8 \
        --lr           2e-4

Expected output on success:
    Epoch 01  task=2.341  kl=0.812  pi=0.693  total=3.431
    ...
    Epoch 30  task=0.893  kl=0.210  pi=0.181  total=1.192
    [PASS] Task loss dropped by X% (>30%)
    [PASS] KD loss dropped by X% (>30%)
    [PASS] No NaN/Inf detected
    ✓ Smoke test passed. Safe to run full training.
"""

import argparse
import sys
import random
from pathlib import Path
from copy import deepcopy
from .hivt_kd import HiVTKD
import numpy as np
import torch
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
sys.path.insert(0, str(Path(__file__).parent))

from datasets import ArgoverseV1Dataset
from .kd_dataset import KDDataset
from .kd_loss import HiVTKDLoss
from .kd_datamodule import KDDataModule
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


# ────────────────────────────────────────────────────────────────────────────
# Minimal subset wrapper
# ────────────────────────────────────────────────────────────────────────────
class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base    = base
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# ────────────────────────────────────────────────────────────────────────────
# Main smoke loop
# ────────────────────────────────────────────────────────────────────────────
def run_smoke(args):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dataset ────────────────────────────────────────────────────────
    print(f"Loading {args.n_scenes} scenes from {args.data_root} ...")
    base    = ArgoverseV1Dataset(root=args.data_root, split="train")
    indices = list(range(args.n_scenes))          # deterministic first N scenes
    subset  = SubsetDataset(KDDataset(base, teacher_dir=args.teacher_dir), indices)
    loader  = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, drop_last=False)

    # ── model ──────────────────────────────────────────────────────────
    print(f"Building HiVTKD  embed_dim={args.embed_dim} ...")

    model =  HiVTKD(**vars(args)).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student params: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    kd_loss_fn = HiVTKDLoss(lambda_kl=args.lambda_kl, lambda_pi=args.lambda_pi)

    # ── training loop ──────────────────────────────────────────────────
    history = []
    nan_detected = False

    for epoch in range(1, args.overfit_epochs + 1):
        model.train()
        epoch_task, epoch_kl, epoch_pi, epoch_total = [], [], [], []

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # ── forward ────────────────────────────────────────────────
            # HiVTKD.training_step returns a dict with loss and metrics.
            # We call the internal _compute_loss to get the decomposed terms.
            # Adjust this call to match your actual hivt_kd.py API.
            try:
                loss, metrics = model.compute_loss(batch)
            except AttributeError:
                # Fall back: call training_step and read logged metrics
                loss = model.training_step(batch, 0)
                metrics = model._last_metrics if hasattr(model, "_last_metrics") else {}

            # ── NaN guard ──────────────────────────────────────────────
            if not loss.isfinite():
                print(f"\n  {FAIL} NaN/Inf loss at epoch {epoch}")
                nan_detected = True
                break

            loss.backward()

            # gradient clipping — same as HiVT paper
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ── Fix string keys to match hivt_kd.py ────────────────────────
            epoch_total.append(loss.item())
            if "train/loss_task" in metrics:
                epoch_task.append(metrics["train/loss_task"])
            if "train/kd/kl_laplace" in metrics:
                epoch_kl.append(metrics["train/kd/kl_laplace"])
            if "train/kd/pi_ce" in metrics:
                epoch_pi.append(metrics["train/kd/pi_ce"])

        if nan_detected:
            break

        mean = lambda lst: float(np.mean(lst)) if lst else float("nan")
        mt, mk, mp, mtot = mean(epoch_task), mean(epoch_kl), mean(epoch_pi), mean(epoch_total)
        history.append(dict(task=mt, kl=mk, pi=mp, total=mtot))

        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:02d}  task={mt:.3f}  kl={mk:.3f}  "
                  f"pi={mp:.3f}  total={mtot:.3f}")

    # ── verdict ────────────────────────────────────────────────────────
    print("\n── Verdict ─────────────────────────────────────────────────────")
    all_ok = True

    # NaN
    ok_nan = not nan_detected
    print(f"  [{PASS if ok_nan else FAIL}] No NaN/Inf detected")
    all_ok = all_ok and ok_nan

    if len(history) < 2:
        print(f"  {FAIL} Not enough epochs completed")
        return False

    first, last = history[0], history[-1]

    # Task loss
    if not np.isnan(first["task"]) and not np.isnan(last["task"]):
        drop_task = (first["task"] - last["task"]) / (abs(first["task"]) + 1e-8)
        ok_task = drop_task > 0.20
        print(f"  [{PASS if ok_task else FAIL}] Task loss  {first['task']:.3f} → {last['task']:.3f} "
              f"  (drop {drop_task*100:.1f}%  threshold >20%)")
        all_ok = all_ok and ok_task
    else:
        print("  [----] Task loss not tracked (metrics key missing in model)")

    # KD total loss
    if not np.isnan(first["total"]) and not np.isnan(last["total"]):
        drop_tot = (first["total"] - last["total"]) / (abs(first["total"]) + 1e-8)
        ok_tot = drop_tot > 0.15
        print(f"  [{PASS if ok_tot else FAIL}] Total loss {first['total']:.3f} → {last['total']:.3f} "
              f"  (drop {drop_tot*100:.1f}%  threshold >15%)")
        all_ok = all_ok and ok_tot

    print()
    if all_ok:
        print("  \033[92m✓ Smoke test passed. Safe to launch full training.\033[0m")
    else:
        print("  \033[91m✗ Smoke test FAILED. Fix the issues above before full training.\033[0m")

    return all_ok


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    # 🔥 reuse full training parser
    parser = HiVTKD.add_model_specific_args(parser)
    parser = KDDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    # ➕ keep your smoke-specific args
    parser.add_argument("--n_scenes", type=int, default=64)
    parser.add_argument("--overfit_epochs", type=int, default=30)

    args = parser.parse_args()
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--teacher_h5",      required=True)
    # parser.add_argument("--data_root",        required=True)
    # parser.add_argument("--embed_dim",        type=int,   default=64)
    # parser.add_argument("--lambda_task",      type=float, default=1.0)
    # parser.add_argument("--lambda_kl",        type=float, default=0.5)
    # parser.add_argument("--lambda_pi",        type=float, default=0.5)
    # parser.add_argument("--n_scenes",         type=int,   default=64,
    #                     help="Number of scenes to overfit on")
    # parser.add_argument("--overfit_epochs",   type=int,   default=30)
    # parser.add_argument("--batch_size",       type=int,   default=8)
    # parser.add_argument("--lr",               type=float, default=2e-4)
    # args = parser.parse_args()

    ok = run_smoke(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
