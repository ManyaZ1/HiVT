# test_kd_shapes.py
import torch
import torch.distributions as D
from kd_loss import HiVTKDLoss

F, N, H = 6, 4, 30   # small batch for readability

# ── Simulate student decoder output (HiVT convention: F first) ──
loc_s   = torch.randn(F, N, H, 2)
scale_s = torch.rand(F, N, H, 2).abs() + 1e-3
pi_s    = torch.randn(N, F)

# ── Simulate teacher tensors AS SAVED: per-agent [F, H, 2] ──
# After PyG collation of N agents, stacked on dim 0 → [N, F, H, 2]
loc_t_saved   = torch.randn(F, H, 2)          # single agent as saved
scale_t_saved = torch.rand(F, H, 2).abs() + 1e-3
pi_t_saved    = torch.randn(F)

# Simulate PyG stacking N of these
loc_t_batched   = loc_t_saved.unsqueeze(0).expand(N, F, H, 2)   # [N, F, H, 2]
scale_t_batched = scale_t_saved.unsqueeze(0).expand(N, F, H, 2)
pi_t_batched    = pi_t_saved.unsqueeze(0).expand(N, F)           # [N, F]

print("=== Shape check ===")
print(f"loc_s   : {loc_s.shape}          ---  student  (F, N, H, 2)")
print(f"loc_t   : {loc_t_batched.shape}  --- teacher  (N, F, H, 2)")
print(f"MATCH?    {loc_s.shape == loc_t_batched.shape}")   # True but WRONG — same shape, different layout

# ── What the loss sees ──
loss_fn = HiVTKDLoss(lambda_kl=0.5, lambda_pi=0.5)
total, logs = loss_fn(loc_s, scale_s, pi_s,
                      loc_t_batched, scale_t_batched, pi_t_batched)
print(f"\nLoss: {total.item():.4f}  (if very large, shapes are misaligned)")
print(logs)

# ── Correct usage: permute teacher to match student convention ──
loc_t_fixed   = loc_t_batched.permute(1, 0, 2, 3)    # [N,F,H,2] → [F,N,H,2]
scale_t_fixed = scale_t_batched.permute(1, 0, 2, 3)
# pi is [N, F] in both cases — no permute needed

total_fixed, logs_fixed = loss_fn(loc_s, scale_s, pi_s,
                                  loc_t_fixed, scale_t_fixed, pi_t_batched)
print(f"\nFixed loss: {total_fixed.item():.4f}  (should be much smaller)")
print(logs_fixed)

# ── Sanity: identical distributions should give KL=0 ──
total_zero, _ = loss_fn(loc_s, scale_s, pi_s,
                        loc_s.detach(), scale_s.detach(), pi_s.detach())
print(f"\nSelf-KL (should be ~0): {total_zero.item():.6f}")