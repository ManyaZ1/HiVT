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

This script loads one validation scene, runs forward inference, and plots:
- lane centerlines from ArgoverseMap in the model local frame,
- observed history,
- ground-truth future,
- multimodal predictions weighted by mode confidence.
"""


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
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )

    lane_ids = avm.get_lane_ids_in_xy_bbox(origin_xy[0], origin_xy[1], city, map_radius)
    local_centerlines = []
    for lane_id in lane_ids:
        centerline_city = avm.get_lane_segment_centerline(lane_id, city)[:, :2].astype(np.float32)
        centerline_local = (centerline_city - origin_xy) @ rotate_mat
        local_centerlines.append(centerline_local)
    return local_centerlines


def visualize_predictions(data, predictions, mode_probs, avm, map_radius=80.0, sample_idx=0, save_path=None):
    """
    Visualize predicted trajectories vs ground truth.
    
    Args:
        data: Temporal data sample with historical and ground truth trajectories
        predictions: Model predictions [F, N, T, 2] (F modes, N agents, T timesteps, 2 coords)
        mode_probs: Mode probabilities [N, F]
        avm: Initialized ArgoverseMap instance
        map_radius: Lane retrieval radius (meters) around sample origin
        sample_idx: Reserved for compatibility with caller-side indexing
        save_path: Path to save the figure (optional)
    """
    # Convert everything to CPU tensors for consistent frame conversion.
    pred = predictions.detach().cpu() if torch.is_tensor(predictions) else torch.as_tensor(predictions)
    logits = mode_probs.detach().cpu() if torch.is_tensor(mode_probs) else torch.as_tensor(mode_probs)
    positions = data.positions.detach().cpu() if torch.is_tensor(data.positions) else torch.as_tensor(data.positions)
    y_true = data.y.detach().cpu() if torch.is_tensor(data.y) else torch.as_tensor(data.y)

    num_modes = pred.shape[0]
    num_agents = pred.shape[1]

    # Decoder outputs [dx, dy, sx, sy]; plot only displacement components.
    pred = pred[..., :2]

    # In model.forward with rotate=True, both pred and data.y are in actor-rotated frames.
    # Bring them back to scene-local frame to align with data.positions and map centerlines.
    rotate_mat = data['rotate_mat'] if 'rotate_mat' in data else None
    if rotate_mat is not None:
        rotate_mat = rotate_mat.detach().cpu() if torch.is_tensor(rotate_mat) else torch.as_tensor(rotate_mat)
        inv_rotate = rotate_mat.transpose(1, 2)
        y_true = torch.bmm(y_true, inv_rotate)
        pred = torch.einsum('fnth,nhd->fntd', pred, inv_rotate)

    # Convert relative future displacements to absolute scene-local coordinates.
    anchor = positions[:, 19, :]
    y_true_abs = y_true + anchor.unsqueeze(1)
    pred_abs = pred + anchor.unsqueeze(0).unsqueeze(2)

    # Convert logits to probabilities.
    probs = torch.softmax(logits, dim=-1)

    lane_centerlines_local = _get_local_lane_centerlines(data, avm, map_radius)
    
    # Show only agents visible at current time step to avoid misleading padded tracks.
    visible_mask = (~data['padding_mask'][:, 19]).detach().cpu().numpy()
    agent_indices = np.where(visible_mask)[0]
    if len(agent_indices) == 0:
        agent_indices = np.arange(num_agents)
    agent_indices = agent_indices[:6]

    # Determine grid layout based on number of agents
    n_agents_to_show = len(agent_indices)
    if n_agents_to_show == 0:
        print('No agents available to visualize for this sample.')
        return
    cols = min(3, n_agents_to_show)
    rows = (n_agents_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for plot_idx, agent_idx in enumerate(agent_indices):
        ax = axes.flatten()[plot_idx]

        # Plot map centerlines in the same local frame as trajectories.
        for centerline in lane_centerlines_local:
            ax.plot(centerline[:, 0], centerline[:, 1], color='0.65', linewidth=1.0, alpha=0.7, zorder=1)
        
        # Plot historical trajectory
        hist_traj = positions[agent_idx, :20].numpy()
        ax.plot(hist_traj[:, 0], hist_traj[:, 1], 'b-o', linewidth=2, markersize=4, label='History', zorder=3)
        
        # Plot ground truth future trajectory
        gt_traj = y_true_abs[agent_idx].numpy()
        future_valid = (~data['padding_mask'][agent_idx, 20:]).detach().cpu().numpy()
        if future_valid.any():
            ax.plot(gt_traj[future_valid, 0], gt_traj[future_valid, 1], 'g-s', linewidth=2, markersize=4,
                    label='Ground Truth', zorder=3)
        
        # Plot predicted trajectories with transparency based on mode probability
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, num_modes))
        
        # Use softmax probabilities from decoder logits.
        agent_probs = probs[agent_idx].numpy()
        
        for mode_idx in range(num_modes):
            pred_traj = pred_abs[mode_idx, agent_idx].numpy()
            alpha = np.clip(agent_probs[mode_idx] * 0.9, 0.1, 0.9)  # Clamp to [0.1, 0.9] for visibility
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-^', 
                   linewidth=1.5, markersize=3, alpha=alpha, 
                   color=colors[mode_idx], zorder=2)
        
        # Mark start and end points
        ax.plot(hist_traj[0, 0], hist_traj[0, 1], 'b*', markersize=15, zorder=5)
        if future_valid.any():
            last_valid_idx = np.where(future_valid)[0][-1]
            ax.plot(gt_traj[last_valid_idx, 0], gt_traj[last_valid_idx, 1], 'g*', markersize=15, zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Agent {agent_idx}\n(Top mode prob: {agent_probs.max():.3f})')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_agents_to_show, len(axes.flatten())):
        axes.flatten()[idx].set_visible(False)
    
    plt.suptitle(f'HiVT Trajectory Predictions ({num_modes} modes)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Visualize HiVT predictions with Argoverse map centerlines for one validation sample.'
    )
    parser.add_argument('--root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to visualize')
    parser.add_argument('--map_radius', type=float, default=80.0,
                        help='Lane search radius (meters) around batch.origin')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save figure')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.ckpt_path}...")
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=True)
    model.eval()
    avm = ArgoverseMap()
    
    # Load dataset
    print(f"Loading dataset from {args.root}...")
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Get sample
    sample_count = 0
    found_sample = False
    for batch in dataloader:
        if sample_count == args.sample_idx:
            print(f"Visualizing sample {args.sample_idx}...")
            
            # Forward pass
            with torch.no_grad():
                y_hat, pi = model(batch)
            
            # Visualize
            visualize_predictions(batch, y_hat, pi, avm=avm, map_radius=args.map_radius, save_path=args.save_path)
            found_sample = True
            break
        sample_count += 1
    
    if not found_sample:
        dataset_size = len(val_dataset)
        print(f"Error: Sample index {args.sample_idx} out of range (dataset has {dataset_size} samples)")
