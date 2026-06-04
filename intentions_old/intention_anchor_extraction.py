import numpy as np
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_gt_endpoints(data_root: str, split: str = "train") -> np.ndarray:
    """
    Extract ground-truth trajectory endpoints from Argoverse 1.
    Argoverse 1: 20 observed + 30 future steps at 10Hz
    Endpoints = final position at t=50 (30 steps into future)
    """
    DATA_DIR = f"{data_root}/{split}/data/" 
    afl = ArgoverseForecastingLoader(DATA_DIR) # needs csv
    
    endpoints = []
    
    for seq in afl: # seq = scenario/csv
        # Get the focal agent's future trajectory (steps 20-49)
        agent_traj = seq.agent_traj  # shape: (50, 2) in global coords
        
        # Observed: steps 0-19, Future: steps 20-49
        obs_traj    = agent_traj[:20]   # (20, 2)
        future_traj = agent_traj[20:]   # (30, 2)
        
        # --- Normalize to agent-centric frame ---
        # Origin = last observed position
        origin = obs_traj[-1]           # (2,)
        
        # Heading = direction of last observed step
        delta = obs_traj[-1] - obs_traj[-2]
        theta = np.arctan2(delta[1], delta[0])
        
        # Rotation matrix (rotate so heading aligns with +x axis)
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        R = np.array([[cos_t, -sin_t],
                      [sin_t,  cos_t]])
        
        # Transform future trajectory to local frame
        future_local = (future_traj - origin) @ R.T  # (30, 2)
        
        # Endpoint = final predicted position
        endpoint = future_local[-1]     # (2,)
        endpoints.append(endpoint)
    
    return np.array(endpoints)          # (N, 2)

def generate_intention_points(
    endpoints: np.ndarray,
    K: int = 64,
    random_state: int = 42
) -> np.ndarray:
    """
    Cluster GT endpoints into K intention points using k-means.
    Mirrors MTR's approach: each cluster center = one motion mode
    (encodes both direction and velocity magnitude implicitly).
    
    Returns: intention_points (K, 2) in agent-centric frame
    """
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    kmeans.fit(endpoints) # 
    
    intention_points = kmeans.cluster_centers_  # (K, 2)
    return intention_points


def visualize_intention_points(
    endpoints: np.ndarray,
    intention_points: np.ndarray,
    K: int
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Raw endpoint distribution
    axes[0].scatter(endpoints[:, 0], endpoints[:, 1],
                    alpha=0.05, s=2, c='steelblue')
    axes[0].set_title(f"GT Endpoints Distribution (N={len(endpoints)})")
    axes[0].set_xlabel("x (forward)")
    axes[0].set_ylabel("y (lateral)")
    axes[0].set_aspect('equal')
    axes[0].grid(True)
    
    # Intention points overlaid
    axes[1].scatter(endpoints[:, 0], endpoints[:, 1],
                    alpha=0.03, s=2, c='steelblue', label='GT endpoints')
    axes[1].scatter(intention_points[:, 0], intention_points[:, 1],
                    s=80, c='red', zorder=5, label=f'K={K} intention points')
    axes[1].set_title(f"K={K} Intention Points (k-means centers)")
    axes[1].set_xlabel("x (forward)")
    axes[1].set_ylabel("y (lateral)")
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("intention_points.png", dpi=150)
    plt.show()


# ── Offline: run once before training ──────────────────────────────
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file from current directory
data_root = os.getenv('PATH_ARGOVERSE1')
endpoints = extract_gt_endpoints(data_root, split="train")
intention_points = generate_intention_points(endpoints, K=64)
np.save("intention_points_k64.npy", intention_points)
visualize_intention_points(endpoints, intention_points, K=64)

# import os
# import numpy as np
# import torch
# from sklearn.cluster import KMeans
# from tqdm import tqdm
# # Iterates over  training set,
# # Extracts the local-frame final displacement vectors for each agent,
# # Runs KMeans to find F intention anchors,
# # Saves the anchors as both intention_anchors.npy and intention_anchors.pt.
# # Adjust these paths and parameters as needed

# TRAIN_DATA_DIR = '/home/manyazog/argoverse'  # Path to your processed training data
# ANCHOR_OUT_NPY = 'intention_anchors.npy'
# ANCHOR_OUT_PT = 'intention_anchors.pt'
# F = 6  # Number of intention anchors (modes)

# # Helper: load your processed data
# # This assumes you have a way to iterate over your training set and get the local-frame future positions
# # You may need to adapt this to your actual data pipeline

# def extract_final_displacements():
#     """
#     Returns:
#         endpoints: [N_total, 2] numpy array of final displacement vectors in local agent frame
#     """
#     endpoints = []
#     # Example: iterate over all training samples
#     # You may need to adapt this to your data loading logic
#     for root, dirs, files in os.walk(TRAIN_DATA_DIR):
#         for file in files:
#             if file.endswith('.npz') or file.endswith('.pt'):
#                 path = os.path.join(root, file)
#                 try:
#                     if file.endswith('.npz'):
#                         data = np.load(path)
#                         y = data['y']  # [N, 30, 2] in local frame
#                         valid = ~data['padding_mask'][:, 20:]  # [N, 30]
#                     else:
#                         data = torch.load(path)
#                         y = data['y'].numpy()  # [N, 30, 2]
#                         valid = (~data['padding_mask'][:, 20:]).numpy()  # [N, 30]
#                     # For each agent, get the last valid future position
#                     for i in range(y.shape[0]):
#                         valid_idx = np.where(valid[i])[0]
#                         if len(valid_idx) > 0:
#                             final_disp = y[i, valid_idx[-1]]  # [2]
#                             endpoints.append(final_disp)
#                 except Exception as e:
#                     print(f"Skipping {path}: {e}")
#     endpoints = np.stack(endpoints, axis=0)  # [N_total, 2]
#     return endpoints

# def main():
#     print("Extracting final displacement vectors...")
#     endpoints = extract_final_displacements()
#     print(f"Collected {endpoints.shape[0]} endpoints.")
#     print("Running KMeans clustering...")
#     kmeans = KMeans(n_clusters=F, random_state=0).fit(endpoints)
#     anchor_positions = kmeans.cluster_centers_  # [F, 2]
#     print(f"Saving anchors to {ANCHOR_OUT_NPY} and {ANCHOR_OUT_PT}")
#     np.save(ANCHOR_OUT_NPY, anchor_positions)
#     torch.save(torch.from_numpy(anchor_positions), ANCHOR_OUT_PT)
#     print("Done.")

# if __name__ == '__main__':
#     main()
