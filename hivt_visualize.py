# Copyright (c) 2022, Zikang Zhou. All rights reserved.
# Licensed under the Apache License, Version 2.0

from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_city_name(city_attr):
    if isinstance(city_attr, str):
        return city_attr
    if isinstance(city_attr, (list, tuple)) and len(city_attr) > 0:
        return str(city_attr[0])
    return str(city_attr)


def _extract_origin_xy(origin_attr):
    t = origin_attr.detach().cpu().numpy() if torch.is_tensor(origin_attr) else np.asarray(origin_attr)
    t = np.squeeze(t)
    return (t.reshape(-1)[:2] if t.shape != (2,) else t).astype(np.float32)


def _extract_theta(theta_attr):
    t = theta_attr.detach().cpu().numpy() if torch.is_tensor(theta_attr) else np.asarray(theta_attr)
    return float(np.squeeze(t))


def _get_local_lane_centerlines(data, avm, map_radius):
    city   = _extract_city_name(data.city)
    origin = _extract_origin_xy(data.origin)
    theta  = _extract_theta(data.theta)
    R      = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
    ids    = avm.get_lane_ids_in_xy_bbox(origin[0], origin[1], city, map_radius)
    cls    = []
    for lid in ids:
        c = avm.get_lane_segment_centerline(lid, city)[:, :2].astype(np.float32)
        cls.append((c - origin) @ R)
    return cls


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_model_outputs(y_hat, pi):
    """
    Print statistics that reveal how pi and y_hat should be interpreted.

    pi format possibilities:
      - LOG-PROBS  : all values < 0  AND  exp(pi) sums to ~1 per agent
      - PROBS      : all values >= 0 AND  pi sums to ~1 per agent
      - RAW LOGITS : anything else

    y_hat format possibilities:
      - Cumulative relative offsets from anchor (most common in HiVT):
          t=0 value is very small (first predicted step from anchor),
          magnitudes grow over time.
      - Per-step deltas:
          t=0 can be non-zero, magnitudes don't grow monotonically.
          Need cumsum before adding anchor.
    """
    pi_c    = pi.detach().cpu()
    yhat_c  = y_hat.detach().cpu()

    print("\n" + "="*60)
    print("DIAGNOSTIC — model output shapes & statistics")
    print("="*60)
    print(f"\n  y_hat : {tuple(yhat_c.shape)}   (F, N, T, D)")
    print(f"  pi    : {tuple(pi_c.shape)}   (N, F)")

    # ── pi ──
    pi_min = pi_c.min().item()
    pi_max = pi_c.max().item()
    row_sum     = pi_c.sum(dim=-1)           # [N]
    exp_row_sum = pi_c.exp().sum(dim=-1)     # [N]

    print(f"\n  pi   range       : [{pi_min:.4f}, {pi_max:.4f}]")
    print(f"  pi   row-sum     : [{row_sum.min():.4f}, {row_sum.max():.4f}]")
    print(f"  exp(pi) row-sum  : [{exp_row_sum.min():.4f}, {exp_row_sum.max():.4f}]")

    if pi_max < 0 and (exp_row_sum - 1.0).abs().max() < 0.05:
        pi_label = "LOG-PROBABILITIES  →  use exp(pi)"
    elif pi_min >= 0 and (row_sum - 1.0).abs().max() < 0.05:
        pi_label = "PROBABILITIES      →  use pi directly"
    else:
        pi_label = "RAW LOGITS         →  use softmax(pi, dim=-1)"
    print(f"\n  ✔  pi  is: {pi_label}")

    # ── y_hat ──
    t0_mag   = yhat_c[..., 0, :2].norm(dim=-1).mean().item()
    tlast_mg = yhat_c[..., -1, :2].norm(dim=-1).mean().item()
    print(f"\n  y_hat t=0  mean |xy| : {t0_mag:.4f} m")
    print(f"  y_hat t=-1 mean |xy| : {tlast_mg:.4f} m")

    mags         = yhat_c[..., :2].norm(dim=-1)          # [F, N, T]
    frac_incr    = ((mags[..., 1:] - mags[..., :-1]) > 0).float().mean().item()
    print(f"  Fraction of increasing-magnitude steps : {frac_incr:.2f}")

    if frac_incr > 0.7:
        yhat_label = "CUMULATIVE offsets from anchor  →  pred + anchor"
    else:
        yhat_label = "PER-STEP deltas                 →  cumsum(pred) + anchor"
    print(f"\n  ✔  y_hat is: {yhat_label}")
    print("="*60 + "\n")

# find interesting samples based on diagnostics:
# Add this to your script to scan for interesting samples
def find_multimodal_scenes(dataloader, model, n_samples=200, min_prob_gap=0.3):
    """
    Scan samples and score them by how DIVERSE their predicted modes are.
    A good multimodal scene has:
      1. High variance in mode probabilities (one mode clearly preferred)
      2. High spatial spread between predicted endpoints
    """
    scores = []
    for count, batch in enumerate(dataloader):
        if count >= n_samples:
            break
        with torch.no_grad():
            y_hat, pi = model(batch)

        pred  = y_hat[..., :2].detach().cpu()   # [F, N, T, 2]
        probs = pi.detach().cpu().exp()          # [N, F]  — assuming log-probs

        # Focus on the focal agent (index 0 is usually focal in Argoverse)
        agent_probs = probs[0]                   # [F]
        agent_pred  = pred[:, 0, -1, :]          # [F, 2] — final predicted positions

        # Spatial spread: std of endpoint positions across modes
        endpoint_std = agent_pred.std(dim=0).norm().item()

        # Probability peakiness: max - min  (0=flat, 1=one dominant mode)
        prob_gap = (agent_probs.max() - agent_probs.min()).item()

        scores.append((count, endpoint_std, prob_gap))
        print(f"Sample {count:4d} | endpoint_spread={endpoint_std:6.2f}m | prob_gap={prob_gap:.3f}")

    # Sort by endpoint spread descending
    scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 most multimodal samples (by endpoint spread):")
    for idx, spread, gap in scores[:10]:
        print(f"  sample {idx:4d} — spread={spread:.2f}m  prob_gap={gap:.3f}")
# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_predictions(
    data,
    y_hat,
    pi,
    avm,
    map_radius : float = 80.0,
    max_agents : int   = 6,
    save_dir: Optional[str] = None,
    sample_idx : int   = 0,
    show_uncertainty: bool = False,
    uncertainty_scale: float = 1.5,
):
    """
    One figure per visible agent; one subplot per mode, sorted by
    descending probability.  The pi / y_hat format is auto-detected.
    """

    # ── 1. to cpu ─────────────────────────────────────────────────────────────
    pred   = y_hat.detach().cpu() if torch.is_tensor(y_hat) else torch.as_tensor(y_hat)
    logits = pi.detach().cpu()    if torch.is_tensor(pi)    else torch.as_tensor(pi)
    pos    = data.positions.detach().cpu()
    y_true = data.y.detach().cpu()

    num_modes  = pred.shape[0]
    num_agents = pred.shape[1]

    pred = pred[..., :2]   # [F, N, T, 2] — drop variance terms if present

    # ── 2. recover probabilities ──────────────────────────────────────────────
    exp_sum = logits.exp().sum(dim=-1)
    if (exp_sum - 1.0).abs().max() < 0.05:
        probs = logits.exp()
        print("[vis] pi = log-probs → exp(pi)")
    elif logits.min() >= 0 and (logits.sum(dim=-1) - 1.0).abs().max() < 0.05:
        probs = logits
        print("[vis] pi = probs → as-is")
    else:
        probs = torch.softmax(logits, dim=-1)
        print("[vis] pi = raw logits → softmax")

    # ── 3. undo per-actor rotation → scene-local frame ───────────────────────
    R_batch = getattr(data, 'rotate_mat', None)
    if R_batch is not None:
        R     = R_batch.detach().cpu()
        R_inv = R.transpose(1, 2)                          # [N, 2, 2]
        y_true = torch.bmm(y_true, R_inv)                  # [N, T, 2]
        pred   = torch.einsum('fntd,nde->fnte', pred, R_inv)  # [F, N, T, 2]

    # ── 4. relative → absolute ────────────────────────────────────────────────
    anchor = pos[:, 19, :]                                 # [N, 2]
    t0_mag = pred[..., 0, :].norm(dim=-1).mean().item()

    if t0_mag < 1.0:
        # cumulative offsets (standard HiVT output)
        y_true_abs = y_true + anchor.unsqueeze(1)
        pred_abs   = pred   + anchor.unsqueeze(0).unsqueeze(2)
    else:
        # per-step deltas — need cumulative sum first
        y_true_abs = y_true.cumsum(dim=1) + anchor.unsqueeze(1)
        pred_abs   = pred.cumsum(dim=2)   + anchor.unsqueeze(0).unsqueeze(2)

    # ── 5. visible agents ─────────────────────────────────────────────────────
    pmask        = data['padding_mask'].detach().cpu()     # [N, 50]
    visible_mask = ~pmask[:, 19].numpy()
    agent_idxs   = np.where(visible_mask)[0]
    if len(agent_idxs) == 0:
        agent_idxs = np.arange(num_agents)
    if len(agent_idxs) > max_agents:
        agent_idxs = agent_idxs[:max_agents]

    # ── 6. map ────────────────────────────────────────────────────────────────
    lane_cls  = _get_local_lane_centerlines(data, avm, map_radius)
    city_name = _extract_city_name(data.city)

    # ── 7. colours ────────────────────────────────────────────────────────────
    # Most probable mode (col 0) → brightest colour; least likely → darkest
    mode_colors = plt.cm.plasma(np.linspace(0.85, 0.15, num_modes))

    # ── 8. draw ───────────────────────────────────────────────────────────────
    for agent_idx in agent_idxs:
        agent_probs  = probs[agent_idx].numpy()            # [F]
        sorted_modes = np.argsort(agent_probs)[::-1]       # prob descending

        has_future = ~pmask[agent_idx, 20:].numpy()        # [T]
        has_gt     = has_future.any()

        fig, axes = plt.subplots(1, num_modes,
                                 figsize=(4.2 * num_modes, 5.2),
                                 squeeze=False)
        axes = axes[0]   # [F]

        hist = pos[agent_idx, :20].numpy()                 # [20, 2]
        gt   = y_true_abs[agent_idx].numpy()               # [T, 2]

        for col, mode_idx in enumerate(sorted_modes):
            ax = axes[col]

            # map centerlines
            for cl in lane_cls:
                ax.plot(cl[:, 0], cl[:, 1],
                        color='#c0c0c0', lw=0.8, alpha=0.5, zorder=1)

            # history
            ax.plot(hist[:, 0], hist[:, 1],
                    color='steelblue', lw=1.8, zorder=3)
            ax.plot(*hist[-1], 'o', color='steelblue', ms=5, zorder=4)
            ax.plot(*hist[0],  '*', color='navy',      ms=9, zorder=5,
                    label='History')

            # ground truth
            if has_gt:
                ax.plot(gt[has_future, 0], gt[has_future, 1],
                        color='limegreen', lw=2.0, ls='--', zorder=3,
                        label='Ground Truth')
                last = np.where(has_future)[0][-1]
                ax.plot(*gt[last], '*', color='darkgreen', ms=9, zorder=5)

            # predicted trajectory for THIS mode only
            traj = pred_abs[mode_idx, agent_idx].numpy()   # [T, 2]
            c    = mode_colors[col]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=c, lw=2.2, zorder=4,
                    label=f'Mode {mode_idx + 1}')
            ax.plot(*traj[0],  'o', color=c, ms=5,  zorder=5)   # pred start
            ax.plot(*traj[-1], '^', color=c, ms=7,  zorder=5)   # pred end
            # Draw uncertainty ribbon if requested and available
            print(f"pred_abs.shape: {pred_abs.shape}, show_uncertainty: {show_uncertainty}")
            if show_uncertainty and pred_abs.shape[-1] >= 4:
                b = pred_abs[mode_idx, agent_idx, :, 2:4]  # [T, 2] (b_x, b_y)
                mu_x = traj[:, 0]
                mu_y = traj[:, 1]
                b_y = b[:, 1].numpy()
                print(f"Uncertainty b_y for agent {agent_idx}, mode {mode_idx}: {b_y}")
                ax.fill_between(
                    mu_x,
                    mu_y - uncertainty_scale * b_y,
                    mu_y + uncertainty_scale * b_y,
                    color=c, alpha=0.4, zorder=2,
                    label='Uncertainty' if col == 0 else None
                )
            p = agent_probs[mode_idx]
            ax.set_title(f'Mode {mode_idx + 1}  (rank {col + 1})\np = {p:.4f}',
                         fontsize=9)
            ax.set_xlabel('X (m)', fontsize=8)
            if col == 0:
                ax.set_ylabel('Y (m)', fontsize=8)
            ax.grid(True, alpha=0.18)
            ax.legend(loc='best', fontsize=7)
            ax.set_aspect('equal', adjustable='datalim')

        fig.suptitle(
            f'HiVT — Agent {agent_idx}  |  {city_name}  |  sample {sample_idx}'
            + ('' if has_gt else '  [no ground truth]'),
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fp = os.path.join(save_dir,
                              f'sample{sample_idx:04d}_agent{agent_idx:03d}.png')
            fig.savefig(fp, dpi=150, bbox_inches='tight')
            print(f'  saved → {fp}')
        else:
            plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root',       type=str, required=True)
    parser.add_argument('--ckpt_path',  type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--map_radius', type=float, default=80.0)
    parser.add_argument('--max_agents', type=int,   default=6)
    parser.add_argument('--save_dir',   type=str,   default='/home/manyazog/HiVT/visualisations/')
    parser.add_argument('--diagnose',   action='store_true',
                        help='Print diagnostics only, skip plotting')
    parser.add_argument('--find_multimodal', action='store_true',
    help='Scan dataset for highly multimodal scenes and print their indices.')
    parser.add_argument('--show_uncertainty', action='store_true',
    help='Draw shaded uncertainty ribbon for each predicted trajectory mode (if available)')
    parser.add_argument('--uncertainty_scale', type=float, default=1.5,
    help='Scale factor for Laplace b (e.g., 1.5 ≈ 80%% interval, 3 ≈ 95%%)')
    args = parser.parse_args()

    print(f'Loading model from {args.ckpt_path} …')
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path,
                                      parallel=True)
    model.eval()
    avm = ArgoverseMap()

    print(f'Loading dataset from {args.root} …')
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val',
                                     local_radius=model.hparams.local_radius)
    dataloader  = DataLoader(val_dataset, batch_size=1,
                             shuffle=False, num_workers=0)

    for count, batch in enumerate(dataloader):
        if count != args.sample_idx:
            continue

        with torch.no_grad():
            y_hat, pi = model(batch)

        diagnose_model_outputs(y_hat, pi)   # always printed
        if args.find_multimodal:
            find_multimodal_scenes(dataloader, model, n_samples=200, min_prob_gap=0.3)
            exit(0)
        if not args.diagnose:
            visualize_predictions(
                data       = batch,
                y_hat      = y_hat,
                pi         = pi,
                avm        = avm,
                map_radius = args.map_radius,
                max_agents = args.max_agents,
                save_dir   = args.save_dir,
                sample_idx = args.sample_idx,
                show_uncertainty=args.show_uncertainty,
                uncertainty_scale=args.uncertainty_scale,
            )
        break
    else:
        print(f'ERROR: sample_idx {args.sample_idx} out of range '
              f'({len(val_dataset)} samples total).')
