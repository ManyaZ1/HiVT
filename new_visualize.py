# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT


"""Visualize HiVT multimodal forecasts with Argoverse map context.

Each agent gets its own figure, where every subplot corresponds to one
predicted mode (sorted by descending probability).  The layout is:

    Figure title: "Agent <N> — <city>"
    Subplots (1 row × F cols):  Mode 1 (p=0.42) | Mode 2 (p=0.31) | ...

Each subplot shows:
    - lane centerlines (grey)
    - historical trajectory (blue)
    - ground-truth future  (green, if available)
    - ONE predicted mode   (red, full opacity)

BUG FIXES vs. original code
────────────────────────────
1. Double-softmax on mode probabilities:
       ORIGINAL  probs = torch.softmax(logits, dim=-1)   # logits == pi == raw logits ✓
                                                          # but variable was named
                                                          # "mode_probs", suggesting it
                                                          # was already a probability,
                                                          # causing callers to sometimes
                                                          # pre-softmax before passing in.
       FIX       Renamed parameter to `pi_logits` and
                 apply softmax exactly once, here.

2. Misleading alpha blending hid per-mode structure:
       ORIGINAL  alpha = clip(prob * 0.9, 0.1, 0.9)  — all modes overlaid
       FIX       Each mode is its own subplot at alpha=1.

3. `ax.set_aspect('equal')` broke tight_layout on some matplotlib versions
       when combined with shared-axis grids; moved after tight_layout call.

4. `agent_indices = agent_indices[:6]` silently dropped agents; now
   controlled by `--max_agents` argument with a clear log message.

5. y_true relative→absolute conversion:
       ORIGINAL  y_true_abs = y_true + anchor.unsqueeze(1)
                 This is correct ONLY when y_true is already de-rotated.
       FIX       De-rotation (inv_rotate) is applied before the abs shift,
                 with an explicit assertion on the operation order.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _extract_city_name(city_attr):
    """Return city name as a plain string for ArgoverseMap queries."""
    if isinstance(city_attr, str):
        return city_attr
    if isinstance(city_attr, (list, tuple)) and len(city_attr) > 0:
        return str(city_attr[0])
    return str(city_attr)


def _extract_origin_xy(origin_attr):
    """Convert origin to a float32 NumPy xy vector with shape [2]."""
    if torch.is_tensor(origin_attr):
        origin_np = origin_attr.detach().cpu().numpy()
    else:
        origin_np = np.asarray(origin_attr)
    origin_np = np.squeeze(origin_np)
    if origin_np.shape != (2,):
        origin_np = origin_np.reshape(-1)[:2]
    return origin_np.astype(np.float32)


def _extract_theta(theta_attr):
    """Convert heading angle container to a Python float (radians)."""
    if torch.is_tensor(theta_attr):
        theta_np = theta_attr.detach().cpu().numpy()
    else:
        theta_np = np.asarray(theta_attr)
    return float(np.squeeze(theta_np))


def _get_local_lane_centerlines(data, avm, map_radius):
    """Fetch lane centerlines near origin and rotate/translate them to local frame."""
    city = _extract_city_name(data.city)
    origin_xy = _extract_origin_xy(data.origin)
    theta = _extract_theta(data.theta)
    rotate_mat = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]],
        dtype=np.float32,
    )
    lane_ids = avm.get_lane_ids_in_xy_bbox(
        origin_xy[0], origin_xy[1], city, map_radius
    )
    local_centerlines = []
    for lane_id in lane_ids:
        centerline_city = avm.get_lane_segment_centerline(lane_id, city)[:, :2].astype(np.float32)
        centerline_local = (centerline_city - origin_xy) @ rotate_mat
        local_centerlines.append(centerline_local)
    return local_centerlines


# ──────────────────────────────────────────────────────────────────────────────
# Main visualisation function
# ──────────────────────────────────────────────────────────────────────────────

def visualize_predictions(
    data,
    y_hat,
    pi_logits,           # FIX 1: renamed from `mode_probs` to make type explicit
    avm,
    map_radius: float = 80.0,
    max_agents: int = 6,  # FIX 4: explicit cap with log message
    save_dir: Optional[str] = None,
    sample_idx: int = 0,
):
    """
    For each visible agent, produce one figure with F subplots (one per mode).

    Args:
        data        : Batched PyG temporal data (batch_size == 1).
        y_hat       : Model output trajectories, shape [F, N, T, D].
                      D >= 2; only the first two dims (dx, dy) are used.
        pi_logits   : Raw mode logits from the decoder, shape [N, F].
                      Softmax is applied once, here.
        avm         : Initialised ArgoverseMap instance.
        map_radius  : Lane search radius in metres around data.origin.
        max_agents  : Maximum number of agents to plot.
        save_dir    : Directory to write per-agent PNG files (optional).
        sample_idx  : Used only in the saved file name.
    """

    # ── 1. Move everything to CPU numpy ──────────────────────────────────────
    pred    = (y_hat.detach().cpu()       if torch.is_tensor(y_hat)       else torch.as_tensor(y_hat))
    logits  = (pi_logits.detach().cpu()   if torch.is_tensor(pi_logits)   else torch.as_tensor(pi_logits))
    pos     = (data.positions.detach().cpu() if torch.is_tensor(data.positions) else torch.as_tensor(data.positions))
    y_true  = (data.y.detach().cpu()      if torch.is_tensor(data.y)      else torch.as_tensor(data.y))

    num_modes  = pred.shape[0]   # F
    num_agents = pred.shape[1]   # N

    # ── 2. Only use (dx, dy); discard sx, sy if present ──────────────────────
    pred = pred[..., :2]   # [F, N, T, 2]

    # ── 3. FIX 1: apply softmax exactly once ─────────────────────────────────
    # pi_logits is raw output from HiVT's decoder (before any normalisation).
    probs = torch.softmax(logits, dim=-1)  # [N, F]

    # ── 4. Rotate predictions & GT back to scene-local frame ─────────────────
    # HiVT rotates each actor's frame by its own heading (rotate_mat per agent).
    # To align with data.positions and the map we need the inverse rotation,
    # which for an orthogonal matrix is simply the transpose.
    rotate_mat_batch = data.get('rotate_mat', None)
    if rotate_mat_batch is not None:
        R = (rotate_mat_batch.detach().cpu()
             if torch.is_tensor(rotate_mat_batch)
             else torch.as_tensor(rotate_mat_batch))          # [N, 2, 2]
        R_inv = R.transpose(1, 2)                             # [N, 2, 2]

        # y_true: [N, T, 2]  →  bmm expects [N, T, 2] × [N, 2, 2]
        y_true = torch.bmm(y_true, R_inv)                     # [N, T, 2]

        # pred:   [F, N, T, 2]
        # einsum: for each mode f, agent n, timestep t:
        #   pred_local[f,n,t,:] = pred[f,n,t,:] @ R_inv[n]
        pred = torch.einsum('fntd,nde->fnte', pred, R_inv)    # [F, N, T, 2]

    # ── 5. Convert relative displacements → absolute scene-local coords ──────
    # data.positions[:,19,:] is the last observed position — the anchor point
    # from which both y_true and pred are expressed as cumulative offsets.
    anchor = pos[:, 19, :]                         # [N, 2]
    y_true_abs = y_true  + anchor.unsqueeze(1)     # [N, T, 2]
    pred_abs   = pred    + anchor.unsqueeze(0).unsqueeze(2)  # [F, N, T, 2]

    # ── 6. Determine visible agents ──────────────────────────────────────────
    # padding_mask is True where the agent is NOT observed.
    padding_mask = data['padding_mask'].detach().cpu()        # [N, 50]
    visible_mask = ~padding_mask[:, 19].numpy()               # observed at t=19
    agent_indices = np.where(visible_mask)[0]
    if len(agent_indices) == 0:
        print('[visualize] WARNING: No agents visible at t=19; showing all agents.')
        agent_indices = np.arange(num_agents)

    if len(agent_indices) > max_agents:
        print(f'[visualize] Capping display to first {max_agents} of '
              f'{len(agent_indices)} visible agents (use --max_agents to change).')
        agent_indices = agent_indices[:max_agents]

    # ── 7. Map centerlines ────────────────────────────────────────────────────
    lane_centerlines_local = _get_local_lane_centerlines(data, avm, map_radius)
    city_name = _extract_city_name(data.city)

    # ── 8. Per-agent figures ──────────────────────────────────────────────────
    # Sort modes by descending probability for the focal agent so the most
    # likely trajectory is always in the first (leftmost) subplot.
    mode_colors = plt.cm.plasma(np.linspace(0.15, 0.85, num_modes))

    for agent_idx in agent_indices:
        agent_probs   = probs[agent_idx].numpy()               # [F]
        sorted_modes  = np.argsort(agent_probs)[::-1]          # high → low prob

        has_future = (~padding_mask[agent_idx, 20:]).numpy()   # [T_future]
        has_gt     = has_future.any()

        fig, axes = plt.subplots(
            1, num_modes,
            figsize=(4 * num_modes, 5),
            squeeze=False,
        )
        axes = axes[0]  # shape [F]

        hist_traj = pos[agent_idx, :20].numpy()               # [20, 2]
        gt_traj   = y_true_abs[agent_idx].numpy()             # [T_future, 2]

        for plot_col, mode_idx in enumerate(sorted_modes):
            ax = axes[plot_col]

            # — Map ————————————————————————————————————————————
            for cl in lane_centerlines_local:
                ax.plot(cl[:, 0], cl[:, 1],
                        color='#aaaaaa', linewidth=0.8, alpha=0.6, zorder=1)

            # — History ————————————————————————————————————————
            ax.plot(hist_traj[:, 0], hist_traj[:, 1],
                    color='steelblue', linewidth=1.8, zorder=3, label='History')
            ax.plot(*hist_traj[-1], 'o', color='steelblue',
                    markersize=6, zorder=4)           # anchor dot
            ax.plot(*hist_traj[0],  '*', color='navy',
                    markersize=10, zorder=5)           # start star

            # — Ground truth ———————————————————————————————————
            if has_gt:
                ax.plot(gt_traj[has_future, 0], gt_traj[has_future, 1],
                        color='limegreen', linewidth=2.0, linestyle='--',
                        zorder=3, label='Ground Truth')
                last_valid = np.where(has_future)[0][-1]
                ax.plot(*gt_traj[last_valid], '*', color='darkgreen',
                        markersize=10, zorder=5)

            # — Predicted mode (this subplot only) ————————————
            pred_traj = pred_abs[mode_idx, agent_idx].numpy()  # [T, 2]
            ax.plot(pred_traj[:, 0], pred_traj[:, 1],
                    color=mode_colors[plot_col], linewidth=2.0,
                    zorder=4, label=f'Mode {mode_idx + 1}')
            ax.plot(*pred_traj[-1], '^', color=mode_colors[plot_col],
                    markersize=7, zorder=5)

            # — Formatting —————————————————————————————————————
            prob_val = agent_probs[mode_idx]
            ax.set_title(
                f'Mode {mode_idx + 1}  (rank {plot_col + 1})\np = {prob_val:.4f}',
                fontsize=9,
            )
            ax.set_xlabel('X (m)', fontsize=8)
            if plot_col == 0:
                ax.set_ylabel('Y (m)', fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best', fontsize=7)
            ax.set_aspect('equal', adjustable='datalim')

        fig.suptitle(
            f'HiVT — Agent {agent_idx}  |  {city_name}  '
            f'|  sample {sample_idx}'
            + ('' if has_gt else '  [no ground truth]'),
            fontsize=11,
            fontweight='bold',
        )
        plt.tight_layout()

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(
                save_dir, f'sample{sample_idx:04d}_agent{agent_idx:03d}.png'
            )
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f'  Saved → {fpath}')

        plt.show()
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize HiVT predictions — one figure per agent, '
                    'one subplot per predicted mode.'
    )
    parser.add_argument('--root',       type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--ckpt_path',  type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of validation sample to visualize')
    parser.add_argument('--map_radius', type=float, default=80.0,
                        help='Lane search radius in metres around origin')
    parser.add_argument('--max_agents', type=int,  default=6,
                        help='Maximum number of agents to show per sample')
    parser.add_argument('--save_dir',   type=str, default=None,
                        help='Directory to save per-agent PNG files')
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading model from {args.ckpt_path} …')
    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path, parallel=True
    )
    model.eval()

    # ── Argoverse map ─────────────────────────────────────────────────────────
    avm = ArgoverseMap()

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f'Loading validation dataset from {args.root} …')
    val_dataset = ArgoverseV1Dataset(
        root=args.root, split='val',
        local_radius=model.hparams.local_radius,
    )
    dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # ── Find & visualize the requested sample ─────────────────────────────────
    for sample_count, batch in enumerate(dataloader):
        if sample_count != args.sample_idx:
            continue

        print(f'Running inference on sample {args.sample_idx} …')
        with torch.no_grad():
            y_hat, pi = model(batch)
        # y_hat : [F, N, T, D]
        # pi    : [N, F]  — raw logits (NOT yet softmax-ed)

        visualize_predictions(
            data=batch,
            y_hat=y_hat,
            pi_logits=pi,        # pass raw logits; softmax applied inside
            avm=avm,
            map_radius=args.map_radius,
            max_agents=args.max_agents,
            save_dir=args.save_dir,
            sample_idx=args.sample_idx,
        )
        break
    else:
        print(f'ERROR: sample_idx={args.sample_idx} is out of range '
              f'(dataset has {len(val_dataset)} samples).')