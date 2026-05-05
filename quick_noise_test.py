# quick_noise_test.py
# Run this on your ALREADY TRAINED HiVT checkpoint.
# Purpose: verify the problem exists. Takes ~2 hours total.

#from torch_geometric.loader import DataLoader
from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT
from argoverse.data_loading.argoverse_forecasting_loader import \
    ArgoverseForecastingLoader
from torch.utils.data import Subset
import torch
import numpy as np
def evaluate_minFDE(model, loader, device, mask_last_k=0, 
                    debug=False):
    model.eval()
    all_fde, all_miss = [], []

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)

            # --- Apply masking ---
            if mask_last_k > 0:
                data = data.clone()
                # Zero position values for last k history steps
                data.x[:, -mask_last_k:, :2] = 0.0
                # Tell attention to ignore these steps
                # padding_mask: (N, 50) — True = valid
                # History occupies columns 0..19
                data.padding_mask[:, 20 - mask_last_k : 20] = False

            # --- Forward pass ---
            y_hat, pi = model(data)
            # y_hat: (K, N, T, 4)  where 4 = (x, y, var_x, var_y)
            # --- Reshape to (N, K, T, 2) ---
            # permute (K,N,T,4) -> (N,K,T,4), take first 2 of last dim
            agent_idx = data['agent_index'] if 'agent_index' in data else 0
            y_hat_agent = y_hat[:, agent_idx, :, :2]
            pred_final = y_hat_agent.permute(1, 0, 2, 3)[:, :, -1, :]  # (B, K, 2)
            gt_final = data.y[agent_idx, -1, :2]  # (B, 2)
            diff = pred_final - gt_final.unsqueeze(1)  # (B, K, 2)
            fde = torch.norm(diff, dim=-1)             # (B, K)
            min_fde = fde.min(dim=1).values             # (B,) — min over K modes
            # --- Debug shapes once ---
            if debug and batch_idx == 0:
                print(f"  y_hat:         {y_hat.shape}")
                print(f"  data.y:        {data.y.shape}")
                print(f"  padding_mask:  {data.padding_mask.shape}")
                print(f"  data.x:        {data.x.shape}")
                
           

            all_fde.extend(min_fde.detach().cpu().tolist())
            all_miss.extend((min_fde > 2.0).detach().cpu().tolist())

    return {
        'minFDE6': float(np.mean(all_fde)),
        'MR6':     float(np.mean(all_miss)),
        'n':       len(all_fde),
    }
def evaluate_minFDEold(model, loader, device, mask_last_k=0):
    """Single-pass evaluation with optional suffix masking."""
    model.eval() 
    all_fde = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device) # moves all tensors inside the data object to the specified device (usually a GPU or CPU).

            if mask_last_k > 0:
                # AV1 history = 20 timesteps (2s at 10Hz)
                # Zero out the last k timesteps of all agents
                data.x = data.x.clone()
                data.x[:, -mask_last_k:, :2] = 0.0
                data.padding_mask[:, 20 - mask_last_k : 20] = False

            y_hat, _ = model(data) #y_hat (K, N, T, 4), pi: predicted probabilities (logits) (K,T)?
            gt_final = data.y[:, -1, :]          # (N, 2)
            pred_final = y_hat[:, :, -1, :2]     # (N, K, 2)      # (N, K, 2)
            pred_final = y_hat.permute(1, 0, 2, 3)[:, :, -1, :2]  # (Ν,Κ,Τ,4)->Ν,Κ,lastT,2->(190, 6, 2)
            # minFDE: best mode final displacement
           
            #pred_final = pred[:, :, -1, :]        # (N, K, 2)
            #print("y_hat shape:", y_hat.shape)
            #print("pred_final shape:", pred_final.shape)
            #print("gt_final shape:", gt_final.shape)
            fde = torch.norm(pred_final - gt_final.unsqueeze(1), dim=-1) #gt_final.unsqueeze(1)->(N, 1, 2)
            # N,K,2 for pred so inorder to compare with ground truth we add an extra dimension to gt_final to make it (N, 1, 2). The norm is then computed across the last dimension (the 2D coordinates) to get the final displacement error for each mode.
            min_fde = fde.min(dim=-1).values      # (N,)
            all_fde.extend(min_fde.cpu().tolist())

    return {
        'minFDE6': np.mean(all_fde),
        'MR6':     np.mean([f > 2.0 for f in all_fde]),
        'n':       len(all_fde)
    }


device = torch.device('cuda')
model  = HiVT.load_from_checkpoint('/home/manyazog/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt').to(device)

# Use your existing val loader — small subset is fine for this test
#val_loader = DataLoader(your_val_dataset, batch_size=32, shuffle=False)
from datasets.argoverse_v1_dataset import ArgoverseV1Dataset  # HiVT dataset class
data_root = '/home/manyazog/argoverse'
#dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
#data_root = "/home/manyazog/datasets/argoverse_forecasting"  # update to your AV1 root
val_dataset = ArgoverseV1Dataset(root=data_root, split="val", local_radius=50)
from torch_geometric.data import DataLoader
# Optional: evaluate on a smaller subset for quick verification
max_samples =5000 #10
if len(val_dataset) > max_samples:
    val_dataset = Subset(val_dataset, range(max_samples))

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4, #nproc=8
    pin_memory=True,
)

# AV1 history window = 20 steps
# masking k steps = k/20 ratio
results = {}
for k in [0, 2, 4, 6, 8, 10, 14, 18]:
    ratio = k / 20
    m = evaluate_minFDE(model, val_loader, device, mask_last_k=k,debug=True)
    results[k] = m
    print(f"mask last {k:2d} steps ({ratio:.0%}) | "
          f"minFDE6={m['minFDE6']:.3f}m | MR6={m['MR6']:.3f}")



''' 3D array slicing (typically with numpy or torch.Tensor).It selects a subset of data from a 3D structure, often used for extracting specific tokens, sequences, or spatial regions from a batch.Breakdown of array[:,-k,:2]Assuming a 3D array (or tensor) with shape (batch, time/sequence, features):: (First dimension): Selects all elements in the first dimension (e.g., all batches).-k (Second dimension): Selects from the \(k\)-th element from the end up to the end of that dimension.:2 (Third dimension): Selects the first two elements (\(0\) and \(1\)) of the third dimension.'''