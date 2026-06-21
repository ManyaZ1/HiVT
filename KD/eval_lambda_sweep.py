"""Evaluate a set of (lambda_kl -> checkpoint) pairs on the full val set and dump
a single {lambda: {metric: value}} JSON — the input to
visualisation_other_tests/plot_lambda_sweep.py AND the raw data behind the
lambda-matrix table in docs/kd_emb32_kl_comparison.md.

Uses the IDENTICAL load + validation path as eval.py (strips the `student.`
prefix from KD checkpoints into a plain HiVT), so every metric — geometry,
brier/p-, mixNLL, and the Task-2 diagnostics (b_scale, calib_err, cov_p*,
pi_entropy) — is computed exactly as in a normal `eval.py` run. Metric keys are
stored WITHOUT the `val_` prefix so they match the plotter's PANELS.

Usage (one entry per lambda; pick the lowest-val_minFDE ckpt in each dir):
    python -m KD.eval_lambda_sweep \
        --root /home/manya/argoverse --batch_size 128 --num_workers 4 \
        --out /tmp/lambda_sweep.json \
        --entry 0.0=kd_ckpt/triage-emb32-lr3e-3-kl0.0-cal/best/HiVTKD-epoch=12-val_minFDE=1.68.ckpt \
        --entry 0.25=kd_ckpt/triage-emb32-lr3e-3-kl0.25-cal/best/HiVTKD-epoch=14-val_minFDE=1.36.ckpt \
        --entry 0.5=kd_ckpt/triage-emb32-lr3e-3-kl0.5-cal/best/HiVTKD-epoch=13-val_minFDE=1.38.ckpt \
        --entry 1.0=kd_ckpt/triage-emb32-lr3e-3-kl1.0-cal/best/HiVTKD-epoch=14-val_minFDE=1.39.ckpt
"""
import json
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from models.hivt import HiVT

# PyTorch >= 2.6 defaults torch.load to weights_only=True; our checkpoints pickle
# callbacks, so restore the old behaviour (same shim as eval.py).
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

_HIVT_KEYS = {
    'historical_steps', 'future_steps', 'num_modes', 'rotate', 'node_dim',
    'edge_dim', 'embed_dim', 'num_heads', 'dropout', 'num_temporal_layers',
    'num_global_layers', 'local_radius', 'parallel', 'lr', 'weight_decay', 'T_max',
}


def load_hivt(ckpt_path):
    """Load a checkpoint as a plain HiVT (handles KD `student.`-prefixed ckpts)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    if any(k.startswith('student.') for k in state_dict):
        hp = dict(ckpt['hyper_parameters'])
        kwargs = {k: v for k, v in hp.items() if k in _HIVT_KEYS}
        kwargs['parallel'] = True
        model = HiVT(**kwargs)
        sd = {k[len('student.'):]: v for k, v in state_dict.items()
              if k.startswith('student.')}
        model.load_state_dict(sd, strict=True)
    else:
        model = HiVT.load_from_checkpoint(checkpoint_path=ckpt_path, parallel=True)
    model.eval()
    return model


def main():
    ap = ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=0,
                    help='Keep 0: this driver runs multiple validate() calls in '
                         'ONE process, and forking DataLoader workers after CUDA '
                         'is initialized aborts them (SIGABRT). 0 = load in the '
                         'main process, no fork.')
    ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--out', required=True, help='output sweep JSON path')
    ap.add_argument('--entry', action='append', required=True,
                    metavar='LAMBDA=CKPT', help='repeatable; e.g. 0.25=path/to.ckpt')
    args = ap.parse_args()

    pl.seed_everything(2022)
    pairs = []
    for e in args.entry:
        lam, ckpt = e.split('=', 1)
        pairs.append((lam, ckpt))

    # Build the val dataset once (local_radius is identical across these models).
    first = load_hivt(pairs[0][1])
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val',
                                     local_radius=first.hparams.local_radius)

    sweep = {}
    for lam, ckpt in pairs:
        model = load_hivt(ckpt)
        loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=args.num_workers > 0)
        trainer = pl.Trainer(gpus=args.gpus, logger=False)
        results = trainer.validate(model, loader)[0]   # dict of val_* metrics
        # Strip the `val_` prefix so keys match plot_lambda_sweep.py's PANELS.
        sweep[lam] = {k[len('val_'):] if k.startswith('val_') else k: float(v)
                      for k, v in results.items()}
        print(f"[sweep] lambda={lam}: minFDE={sweep[lam].get('minFDE'):.4f} "
              f"b_scale={sweep[lam].get('b_scale'):.4f} "
              f"calib_err={sweep[lam].get('calib_err'):.4f}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(sweep, f, indent=2)
    print(f"wrote {args.out}  ({len(sweep)} lambda points)")


if __name__ == '__main__':
    main()