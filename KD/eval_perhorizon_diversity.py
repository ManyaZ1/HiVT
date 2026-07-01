"""Per-horizon accuracy + focal-agent mode diversity, batched, one pass per ckpt.

Per-horizon: truncate to 1/2/3 s (10/20/30 steps), re-select best mode by FDE at
that cutoff, report minADE (mean over cutoff steps) and minFDE (endpoint), averaged
over focal agents -- exactly HiVT's ADE/FDE convention (metrics/ade.py, fde.py).

Diversity: for each focal agent, pairwise final-point distance among the K=6 modes;
'spread' = mean over pairs+agents, 'collapsed' = fraction of pairs within 1 m.
"""
from argparse import ArgumentParser
import torch
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT

_orig = torch.load
def _load(*a, **k):
    k.setdefault('weights_only', False); return _orig(*a, **k)
torch.load = _load


def build_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']
    if any(k.startswith('student.') for k in sd):
        hp = dict(ckpt['hyper_parameters'])
        keys = {'historical_steps','future_steps','num_modes','rotate','node_dim',
                'edge_dim','embed_dim','num_heads','dropout','num_temporal_layers',
                'num_global_layers','local_radius','parallel','lr','weight_decay','T_max'}
        kw = {k: v for k, v in hp.items() if k in keys}; kw['parallel'] = True
        m = HiVT(**kw)
        m.load_state_dict({k[len('student.'):]: v for k, v in sd.items()
                           if k.startswith('student.')}, strict=True)
        return m
    return HiVT.load_from_checkpoint(ckpt_path, parallel=True)


def main(args):
    dev = torch.device('cuda' if args.gpus and torch.cuda.is_available() else 'cpu')
    model = build_model(args.ckpt_path).to(dev).eval()
    ds = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    cutoffs = [10, 20, 30]
    ade_sum = {c: 0.0 for c in cutoffs}; fde_sum = {c: 0.0 for c in cutoffs}
    n_agents = 0
    pair_sum = 0.0; pair_cnt = 0; collapsed = 0
    with torch.no_grad():
        for data in dl:
            data = data.to(dev)
            y_hat, pi = model(data)                       # [F,N,T,4], [N,F]
            ai = data['agent_index']
            loc = y_hat[:, ai, :, :2]                     # [F,B,T,2]
            y = data.y[ai]                                # [B,T,2]
            B = y.size(0); n_agents += B
            for c in cutoffs:
                lp = loc[:, :, :c, :]; yt = y[:, :c, :]
                fde = torch.norm(lp[:, :, -1] - yt[:, -1], p=2, dim=-1)   # [F,B]
                best = fde.argmin(dim=0)                                   # [B]
                lb = lp[best, torch.arange(B)]                            # [B,c,2]
                ade_sum[c] += torch.norm(lb - yt, p=2, dim=-1).mean(dim=-1).sum().item()
                fde_sum[c] += torch.norm(lb[:, -1] - yt[:, -1], p=2, dim=-1).sum().item()
            # diversity: final points of the F modes, [F,B,2]
            fin = loc[:, :, -1, :]
            F = fin.size(0)
            for i in range(F):
                for j in range(i + 1, F):
                    d = torch.norm(fin[i] - fin[j], p=2, dim=-1)          # [B]
                    pair_sum += d.sum().item(); pair_cnt += d.numel()
                    collapsed += (d < 1.0).sum().item()

    print(f"CKPT {args.ckpt_path}")
    print(f"n_focal_agents={n_agents}")
    for c in cutoffs:
        print(f"  @{c//10}s  minADE={ade_sum[c]/n_agents:.4f}  minFDE={fde_sum[c]/n_agents:.4f}")
    print(f"  diversity spread={pair_sum/pair_cnt:.3f}  collapsed_frac={collapsed/pair_cnt:.4f}")


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--ckpt_path', required=True)
    p.add_argument('--batch_size', type=int, default=32); p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpus', type=int, default=1)
    main(p.parse_args())
