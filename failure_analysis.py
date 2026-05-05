# failure_analysis.py
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_scenario_metrics(pred, gt, mask):
    """
    pred: (N, K, T, 2)  — N agents, K modes, T timesteps
    gt:   (N, T, 2)
    mask: (N, T)         — valid timestep mask
    Returns per-agent minFDE, minADE, miss (FDE > 2m)
    """
    # Displacement at each step: (N, K, T)
    diff = pred - gt.unsqueeze(1)  # broadcast over K
    dist = torch.norm(diff, dim=-1)

    # ADE: mean over valid timesteps
    valid = mask.unsqueeze(1).float()  # (N, 1, T)
    ade = (dist * valid).sum(-1) / valid.sum(-1).clamp(min=1)  # (N, K)

    # FDE: last valid timestep
    last_idx = mask.long().sum(-1) - 1  # (N,)
    fde = dist[torch.arange(len(last_idx)), :, last_idx]  # (N, K)

    min_ade = ade.min(-1).values   # (N,)
    min_fde = fde.min(-1).values   # (N,)
    miss    = (min_fde > 2.0).float()

    return min_ade, min_fde, miss


def tag_scenario(data):
    """
    Heuristic tags based on map/agent geometry.
    Returns a list of string tags for a scenario.
    """
    tags = []

    # Turning agent: heading change > 30 deg over history
    hist = data['x']  # (N, T, 2) — agent histories
    if hist.shape[1] >= 2:
        delta = hist[:, 1:] - hist[:, :-1]
        angles = torch.atan2(delta[..., 1], delta[..., 0])
        heading_change = (angles[:, -1] - angles[:, 0]).abs() * 180 / np.pi
        if (heading_change > 30).any():
            tags.append('turning')

    # Intersection: many lane centerlines meeting near agent
    # (requires access to map polylines in your data loader)
    # Proxy: high number of map polylines within 30m
    if hasattr(data, 'lane_positions'):
        agent_pos = hist[0, -1]  # ego agent last position
        dists = torch.norm(data.lane_positions - agent_pos, dim=-1)
        nearby_lanes = (dists < 30).sum().item()
        if nearby_lanes > 8:
            tags.append('intersection')

    # Multi-lane merge: agents within 10m laterally
    if hist.shape[0] > 1:
        last_pos = hist[:, -1, :]  # (N, 2)
        pairwise = torch.cdist(last_pos, last_pos)
        close = (pairwise < 10).sum() - len(last_pos)  # subtract diagonal
        if close > 0:
            tags.append('dense_traffic')

    if not tags:
        tags.append('straight')

    return tags


def run_failure_analysis(model, dataloader, device, output_dir='failure_analysis'):
    Path(output_dir).mkdir(exist_ok=True)

    run = wandb.init(project='hivt-robustness', name='failure_analysis')
    table = wandb.Table(columns=[
        'scenario_id', 'tags', 'min_fde', 'min_ade', 'miss', 'n_agents'
    ])

    all_records = []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            pred = model(data)  # (N, K, T, 2) — adjust to your HiVT output format

            min_ade, min_fde, miss = compute_scenario_metrics(
                pred, data.y, data.padding_mask[:, 20:]  # AV1: 20 hist, 30 future
            )

            tags = tag_scenario(data)
            tag_str = '+'.join(tags)

            for i in range(len(min_fde)):
                record = {
                    'scenario_id': f'{batch_idx}_{i}',
                    'tags': tag_str,
                    'min_fde': min_fde[i].item(),
                    'min_ade': min_ade[i].item(),
                    'miss': miss[i].item(),
                    'n_agents': len(min_fde),
                    'batch_idx': batch_idx,
                    'agent_idx': i,
                }
                all_records.append(record)
                table.add_data(
                    record['scenario_id'], tag_str,
                    round(min_fde[i].item(), 3),
                    round(min_ade[i].item(), 3),
                    int(miss[i].item()),
                    len(min_fde)
                )

    wandb.log({'failure_table': table})

    # Find worst scenarios per tag
    import pandas as pd
    df = pd.DataFrame(all_records)
    df.to_csv(f'{output_dir}/all_scenarios.csv', index=False)

    print("\n=== Miss Rate by Tag ===")
    print(df.groupby('tags')['miss'].mean().sort_values(ascending=False))
    print("\n=== Mean minFDE by Tag ===")
    print(df.groupby('tags')['min_fde'].mean().sort_values(ascending=False))

    # Return top-K worst for visualization
    worst = df.nlargest(50, 'min_fde')
    worst.to_csv(f'{output_dir}/worst_50.csv', index=False)

    wandb.finish()
    return df

if __name__ == '__main__':
    # Example usage:
    from torch_geometric.data import DataLoader
    from datasets import ArgoverseV1Dataset
    from models.hivt import HiVT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HiVT.load_from_checkpoint('/home/manyazog/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt').to(device)
    dataset = ArgoverseV1Dataset(root='/home/manyazog/argoverse', split='val')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    df = run_failure_analysis(model, dataloader, device)