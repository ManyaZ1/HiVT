import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from datasets.argoverse_v1_dataset import ArgoverseV1Dataset
from models.hivt import HiVT
#python measure_mode_diversity.py --root /home/manyazog/argoverse --ckpt_path /home/manyazog/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt

# Utility: pairwise euclidean distances between all modes
def pairwise_mode_distances(trajs):
    # trajs: [F, T, 2] (F = num_modes)
    F = trajs.shape[0]
    dists = np.zeros((F, F))
    for i in range(F):
        for j in range(i+1, F):
            # Use final displacement (last point)
            dist = np.linalg.norm(trajs[i, -1] - trajs[j, -1])
            dists[i, j] = dists[j, i] = dist
    return dists

def main(args):
    # Load dataset
    dataset = ArgoverseV1Dataset(args.root, split="val")
    model = HiVT.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    model.eval()

    all_pairwise = []
    for idx in tqdm(range(len(dataset)), desc="Measuring mode diversity"):
        data = dataset[idx]
        with torch.no_grad():
            y_hat, pi = model(data)
        # y_hat: [F, N, T, 2], pi: [N, F]
        y_hat = y_hat.cpu().numpy()
        for agent in range(y_hat.shape[1]):
            trajs = y_hat[:, agent, :, :2]  # [F, T, 2]
            dists = pairwise_mode_distances(trajs)
            # Only upper triangle, nonzero
            vals = dists[np.triu_indices_from(dists, k=1)]
            all_pairwise.extend(vals.tolist())

    all_pairwise = np.array(all_pairwise)
    print("\nMode Diversity Statistics (final displacement):")
    print(f"Mean: {all_pairwise.mean():.3f}")
    print(f"Std:  {all_pairwise.std():.3f}")
    print(f"Min:  {all_pairwise.min():.3f}")
    print(f"Max:  {all_pairwise.max():.3f}")
    print(f"Median: {np.median(all_pairwise):.3f}")
    print(f"10th percentile: {np.percentile(all_pairwise, 10):.3f}")
    print(f"90th percentile: {np.percentile(all_pairwise, 90):.3f}")

    # Optionally, plot histogram
    try:
        import matplotlib.pyplot as plt
        # plt.hist(all_pairwise, bins=40, alpha=0.7)
        # plt.xlabel('Final displacement between modes (m)')
        # plt.ylabel('Count')
        # plt.title('Pairwise Mode Diversity')
        # plt.show()
        # plot zoomed in histograms for distances < 4m
        plt.hist(all_pairwise[all_pairwise < 4.0], bins=40, alpha=0.7)
        plt.xlabel('Final displacement between modes (m)')
        plt.ylabel('Count')
        plt.title('Pairwise Mode Diversity (Zoomed <4m)')
        plt.show()

    except ImportError:
        print("matplotlib not installed, skipping histogram plot.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    main(args)
