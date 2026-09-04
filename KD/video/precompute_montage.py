"""Precompute pass for the ICRA video's coverage montage.

Runs the from-scratch / v1 / v2 students (and the HiVT-128 teacher, stored but
not rendered by default) over Argoverse val scenes and dumps everything the
renderer needs into a pickle. No plotting happens here; no torch is needed to
render. See docs/handoff_video_preparation.md for the design rationale.

Two phases in one run:

  Phase A (--scan N) walks N scenes and accumulates ONLY the coverage tallies
    and a scene-ranking score. This is the validation gate: the aggregate
    coverage it prints must reproduce the paper's cov@p90 (0.903 / 0.711 /
    0.909 for HiVT-32) or the renderer is lying and nothing downstream is
    trustworthy.

  Phase B re-visits the top --keep scenes by ranking score and stores full
    geometry (map, history, ground truth, per-model modes and uncertainty
    band corners) for the renderer.

The coverage statistic replicates models/hivt.py::validation_step exactly:
focal agent only, best mode selected by min FDE, Laplace central-interval
half-width t(p) = -b * ln(1 - p), counted per (timestep, coordinate) point
under reg_mask. It is computed on the RAW model frame -- y_hat and data.y are
both in the per-agent rotated frame after forward(), so no rotation is applied
for the tally. Rotation back to the scene frame happens for DISPLAY only.
"""
from argparse import ArgumentParser
import os
import pickle

import numpy as np
import torch

# PyTorch >= 2.6 defaults torch.load to weights_only=True, which fails on
# checkpoints that pickle the ModelCheckpoint callback. Same shim as eval.py.
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

from argoverse.map_representation.map_api import ArgoverseMap  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from datasets import ArgoverseV1Dataset  # noqa: E402
from models.hivt import HiVT  # noqa: E402

# Nominal levels must match metrics/calibration.py::_DEFAULT_LEVELS.
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# The three montage arms plus the teacher (stored for a possible 4th panel).
# These are the VERIFIED paper-Table-II checkpoints, confirmed by full-val
# eval.py on all seven metrics -- see docs/CHECKPOINTS.md. Do NOT swap them for
# a plausible-looking neighbour: emb32-bs128-lkl0.0, emb32-bs128-lkl0.5 and
# HiVT-32/gxhl2ug9 all look right and are all wrong.
DEFAULT_ARMS = {
    'noKD': 'kd_ckpt/triage-emb32-lr3e-3-kl0.0-full/best/last.ckpt',
    'v1':   'kd_ckpt/triage-emb32-lr3e-3-kl0.5-full/best/last.ckpt',
    'v2':   'kd_ckpt/emb32-bs128-lkl0.5-distv2-full/best/last.ckpt',
    'teacher': 'checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt',
}

_HIVT_KEYS = {
    'historical_steps', 'future_steps', 'num_modes', 'rotate', 'node_dim',
    'edge_dim', 'embed_dim', 'num_heads', 'dropout', 'num_temporal_layers',
    'num_global_layers', 'local_radius', 'parallel', 'lr', 'weight_decay',
    'T_max',
}


def load_hivt(ckpt_path, device):
    """Load a plain HiVT, stripping the `student.` prefix off KD checkpoints.

    Byte-for-byte the same path as eval.py, so the model here is the model that
    produced the paper's numbers.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    if any(k.startswith('student.') for k in state_dict):
        hp = dict(ckpt['hyper_parameters'])
        kwargs = {k: v for k, v in hp.items() if k in _HIVT_KEYS}
        kwargs['parallel'] = True
        model = HiVT(**kwargs)
        student_sd = {k[len('student.'):]: v for k, v in state_dict.items()
                      if k.startswith('student.')}
        model.load_state_dict(student_sd, strict=True)
    else:
        model = HiVT.load_from_checkpoint(checkpoint_path=ckpt_path, parallel=True)
    return model.eval().to(device)


def _agent_index(data):
    ai = data['agent_index']
    return int(ai[0]) if torch.is_tensor(ai) and ai.numel() > 0 else int(ai)


def run_one(model, data, device):
    """Forward one single-scene batch and extract the focal agent's mixture.

    Returns everything in the RAW model frame (per-agent rotated). `data` is
    consumed -- forward() mutates data.y in place -- so callers must pass a
    fresh clone per model.
    """
    data = data.to(device)
    with torch.no_grad():
        y_hat, pi = model(data)          # [F, N, 30, 4], [N, F]

    ai = _agent_index(data)
    reg_mask = ~data['padding_mask'][:, 20:]          # [N, 30]

    y_hat_agent = y_hat[:, ai, :, :2]                  # [F, 30, 2]
    b_agent = y_hat[:, ai, :, 2:]                      # [F, 30, 2]
    y_agent = data.y[ai]                               # [30, 2]  (rotated in-place by forward)
    mask = reg_mask[ai]                                # [30]

    # Best mode by final displacement error -- identical to validation_step.
    fde = torch.norm(y_hat_agent[:, -1] - y_agent[-1], p=2, dim=-1)   # [F]
    best = int(fde.argmin())

    probs = torch.softmax(pi[ai], dim=-1)              # [F]
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())

    return {
        'mu': y_hat_agent.cpu().numpy(),               # [F, 30, 2] raw frame
        'b': b_agent.cpu().numpy(),                    # [F, 30, 2] raw frame
        'y': y_agent.cpu().numpy(),                    # [30, 2]    raw frame
        'mask': mask.cpu().numpy(),                    # [30]
        'probs': probs.cpu().numpy(),                  # [F]
        'best': best,
        'entropy': entropy,
        'rotate_mat': (data['rotate_mat'][ai].cpu().numpy()
                       if data['rotate_mat'] is not None else np.eye(2, dtype=np.float32)),
        'agent_index': ai,
    }


def tally(rec):
    """Per-level inside-counts for the best mode. Mirrors LaplaceCoverage.

    Returns (counts[9], n) where n = valid_steps * 2 coordinates.
    """
    mu = rec['mu'][rec['best']]        # [30, 2]
    b = np.maximum(rec['b'][rec['best']], 1e-12)
    err = np.abs(rec['y'] - mu)        # [30, 2]
    m = rec['mask'][:, None]           # [30, 1]

    counts = np.zeros(len(LEVELS), dtype=np.int64)
    for i, p in enumerate(LEVELS):
        half = -b * np.log(1.0 - p)
        counts[i] = int(((err <= half) & m).sum())
    return counts, int(rec['mask'].sum()) * 2


def mode_spread(rec):
    """Mean pairwise distance between the six predicted mode endpoints.

    The same diversity quantity the paper reports (teacher 5.15 m). Unlike
    mode-weight entropy it measures whether the hypotheses go to DIFFERENT
    PLACES, which is what makes a scene worth showing. Entropy saturates near
    ln(6)=1.792 on a straight road where all six modes overlap, so ranking by
    entropy selects visually trivial scenes.
    """
    ends = rec['mu'][:, -1, :]                       # [F, 2]
    d = np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=-1)
    iu = np.triu_indices(len(ends), k=1)
    return float(d[iu].mean())


def band_corners(rec, level, R_inv, anchor):
    """Per-timestep band rectangles for the best mode, in the SCENE frame.

    The rectangle is axis-aligned in the raw (per-agent) frame, where b_x and
    b_y are defined; its four corners are then rotated as POINTS. Rotating b
    itself as a vector -- what the old visualizer implied -- is wrong.

    Returns [30, 4, 2] with corners ordered for a closed polygon.
    """
    mu = rec['mu'][rec['best']]                              # [30, 2]
    b = rec['b'][rec['best']]                                # [30, 2]
    half = -b * np.log(1.0 - level)                          # [30, 2]

    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)  # [4, 2]
    corners = mu[:, None, :] + signs[None, :, :] * half[:, None, :]           # [30, 4, 2]
    return corners @ R_inv + anchor


def to_scene(pts, R_inv, anchor):
    """Rotate raw-frame points back to the scene frame and un-anchor."""
    return pts @ R_inv + anchor


def get_lanes(data, avm, radius, center_scene):
    """Lane centerlines in the scene frame (AV-centred, AV-heading-aligned).

    The query is centred on `center_scene` -- the FOCAL AGENT's position, not
    the AV's. The scene frame is anchored at the AV, but the focal agent can be
    100 m+ away, so querying around the AV returns lanes that do not cover the
    trajectory at all.
    """
    city = data['city']
    if isinstance(city, (list, tuple)):
        city = city[0]
    city = str(city)

    origin = np.asarray(data['origin']).reshape(-1)[:2].astype(np.float32)
    theta = float(np.squeeze(np.asarray(data['theta'])))
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]], dtype=np.float32)

    # scene -> world is the inverse of world -> scene, which is (xy - origin) @ R.
    center_world = np.asarray(center_scene, dtype=np.float32) @ R.T + origin

    out = []
    for lid in avm.get_lane_ids_in_xy_bbox(center_world[0], center_world[1],
                                           city, radius):
        cl = avm.get_lane_segment_centerline(lid, city)[:, :2].astype(np.float32)
        out.append((cl - origin) @ R)
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument('--root', type=str, default='/home/manya/argoverse')
    ap.add_argument('--out', type=str, required=True, help='output .pkl path')
    ap.add_argument('--scan', type=int, default=2000,
                    help='Phase A: scenes to scan for the coverage validation gate')
    ap.add_argument('--keep', type=int, default=15,
                    help='Phase B: scenes to store geometry for (top by ranking score)')
    ap.add_argument('--level', type=float, default=0.9, help='band nominal level to store')
    ap.add_argument('--map_radius', type=float, default=65.0)
    ap.add_argument('--rank_by', type=str, default='mode_spread',
                    choices=['teacher_entropy', 'mode_spread'],
                    help='scene selection criterion (stated on screen in the video)')
    ap.add_argument('--reuse_scores', type=str, default=None,
                    help='load per-scene scores from an existing pickle and skip '
                         'Phase A entirely (re-select scenes without re-scanning)')
    ap.add_argument('--pct_lo', type=float, default=80.0)
    ap.add_argument('--pct_hi', type=float, default=97.0,
                    help='select from this percentile WINDOW of the ranking score, '
                         'not the top. The extreme tail is pathological: 32 m of mode '
                         'spread means uncertainty so large the band dwarfs the '
                         'trajectory and every model mispredicts, which shows failure '
                         'rather than a calibration difference.')
    ap.add_argument('--max_fde', type=float, default=3.0,
                    help='reject candidates whose v2 best-of-6 endpoint error exceeds '
                         'this. Keeps the montage in the regime where the models are '
                         'right and only their honesty differs.')
    # b_ratio_lo = v1_mean_b / v2_mean_b. The global value is 0.648 (0.2675/0.4126). Selecting NEAR it gives typical shrinkage; selecting below it is cherry-picking.
    ap.add_argument('--b_ratio_lo', type=float, default=0.0,
                    help='min v1/v2 mean-b ratio. Global value is 0.648 '
                         '(0.2675/0.4126). Selecting NEAR it gives typical '
                         'shrinkage; selecting below it is cherry-picking.')
    # b_ratio_hi = v1_mean_b / v2_mean_b. The global value is 0.648 (0.2675/0.4126). Selecting NEAR it gives typical shrinkage; selecting above it is cherry-picking.
    ap.add_argument('--b_ratio_hi', type=float, default=1e9)
    # cov_gap_lo = v2_cov@p90 - v1_cov@p90. The global value is 0.198 (0.909-0.711). Selecting NEAR it gives typical shrinkage; selecting below it is cherry-picking.
    ap.add_argument('--cov_gap_lo', type=float, default=-1e9,
                    help='min per-scene (v2 - v1) p90 coverage gap. Global 0.198.')
    ap.add_argument('--cov_gap_hi', type=float, default=1e9)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    arms = dict(DEFAULT_ARMS)

    print('Loading models ...')
    models = {name: load_hivt(path, device) for name, path in arms.items()}
    for name, m in models.items():
        n_par = sum(p.numel() for p in m.parameters())
        print(f'  {name:8s} embed_dim={m.hparams.embed_dim:3d}  params={n_par/1e3:.0f}k')

    local_radius = models['v2'].hparams.local_radius
    print(f'Loading val dataset from {args.root} ...')
    ds = ArgoverseV1Dataset(root=args.root, split='val', local_radius=local_radius)
    n_scan = min(args.scan, len(ds))
    print(f'  {len(ds)} scenes; scanning {n_scan}')

    # ---------------- Phase A: tallies + ranking score ----------------------
    totals = {name: np.zeros(len(LEVELS), dtype=np.int64) for name in arms}
    n_points = 0
    # Both criteria are recorded for every scene so the selection can be
    # changed without re-scanning.
    scores = {'teacher_entropy': np.zeros(n_scan, dtype=np.float32),
              'mode_spread': np.zeros(n_scan, dtype=np.float32)}

    if args.reuse_scores:
        with open(args.reuse_scores, 'rb') as fh:
            prev = pickle.load(fh)
        scores = prev['scores']
        totals = prev['totals']
        n_points = prev['n_points']
        n_scan = prev['n_scan']
        print(f'Phase A skipped -- reusing scores for {n_scan} scenes '
              f'from {args.reuse_scores}')

    for i in range(0 if not args.reuse_scores else n_scan, n_scan):
        base = ds[i]
        n_this = None
        for name, model in models.items():
            # Fresh clone per model: forward() rotates data.y IN PLACE, so
            # reusing one batch would rotate the target once per model.
            rec = run_one(model, Batch.from_data_list([base.clone()]), device)
            c, n = tally(rec)
            totals[name] += c
            n_this = n
            if name == 'teacher':
                scores['teacher_entropy'][i] = rec['entropy']
                scores['mode_spread'][i] = mode_spread(rec)
        n_points += n_this or 0

        if (i + 1) % 200 == 0:
            cov = {k: totals[k][-1] / max(n_points, 1) for k in arms}
            msg = '  '.join(f'{k}={cov[k]:.3f}' for k in ('noKD', 'v1', 'v2'))
            print(f'  [{i+1}/{n_scan}] cov@p90  {msg}')

    print('\n' + '=' * 62)
    print(f'VALIDATION GATE — aggregate coverage over {n_scan} scenes '
          f'({n_points} points)')
    print('=' * 62)
    print(f'  {"arm":8s} ' + ' '.join(f'p{int(p*100):02d}' for p in LEVELS))
    for name in arms:
        cov = totals[name] / max(n_points, 1)
        calib_err = float(np.mean(np.abs(cov - np.array(LEVELS))))
        print(f'  {name:8s} ' + ' '.join(f'{c:.3f}' for c in cov)
              + f'   calib_err={calib_err:.4f}')
    print('\n  Paper (HiVT-32, full val) cov@p90: noKD 0.903 | v1 0.711 | v2 0.909'
          ' | teacher 0.894')
    if n_scan < len(ds):
        print(f'  Scanned {n_scan}/{len(ds)} scenes -- a subset, so expect '
              f'close-but-not-identical.\n')
    else:
        print('  Full val set: these should match the paper EXACTLY. If v1 reads '
              '~0.754,\n  the wrong v1 checkpoint is wired in -- see '
              'docs/CHECKPOINTS.md. Do not render.\n')

    # ---------------- Phase B: geometry for the selected scenes -------------
    # Selection is a percentile WINDOW, not the top-k. The extreme tail of the
    # mode-spread distribution is where the model has no idea at all: scale
    # explodes (bands of 17x20 m over a 6 m trajectory), best-of-6 error
    # explodes with it, and the v1-vs-v2 contrast collapses because neither
    # covers a 12 m error. Measured on the rank-1 scene of the previous run,
    # the from-scratch baseline scored 54/60 against v1's 34 and v2's 39 --
    # the frame argued against the paper. The window keeps scenes that are
    # genuinely multi-modal but still correctly predicted, which is the regime
    # where only calibration differs.
    rank_score = scores[args.rank_by]
    lo, hi = np.percentile(rank_score, [args.pct_lo, args.pct_hi])
    window = np.where((rank_score >= lo) & (rank_score <= hi))[0]
    window = window[np.argsort(-rank_score[window])]
    print(f'Phase B: {len(window)} candidates in the '
          f'{args.pct_lo:.0f}-{args.pct_hi:.0f}th percentile of {args.rank_by} '
          f'([{lo:.1f}, {hi:.1f}] m); keeping {args.keep} with '
          f'best-of-6 FDE <= {args.max_fde} m ...')
    avm = ArgoverseMap()

    scenes = []
    rejected = 0
    for idx in window:
        if len(scenes) >= args.keep:
            break
        rank = len(scenes)
        base = ds[int(idx)]
        data0 = Batch.from_data_list([base.clone()])

        recs = {}
        for name, model in models.items():
            recs[name] = run_one(model, Batch.from_data_list([base.clone()]), device)

        # Reject scenes the models simply get wrong -- there the band
        # comparison is swamped by the prediction error.
        _r = recs['v2']
        _m = _r['mask'].astype(bool)
        _fde = np.linalg.norm(_r['mu'][:, _m][:, -1] - _r['y'][_m][-1], axis=-1).min()
        if _fde > args.max_fde:
            rejected += 1
            continue
        # Illustrative selection on the SYSTEMATIC pathology (scale shrinkage),
        # never on the stochastic one (whether this scene happens to miss).
        _bb, _cc = {}, {}
        for _k in ('noKD', 'v1', 'v2'):
            _rk = recs[_k]
            _mk = _rk['mask'].astype(bool)
            _bb[_k] = float(_rk['b'][_rk['best']][_mk].mean())
            _cc[_k], _nn = tally(_rk)
        _ratio = _bb['v1'] / max(_bb['v2'], 1e-9)
        _gap = (int(_cc['v2'][-1]) - int(_cc['v1'][-1])) / max(_nn, 1)
        if not (args.b_ratio_lo <= _ratio <= args.b_ratio_hi
                and args.cov_gap_lo <= _gap <= args.cov_gap_hi):
            rejected += 1
            continue

        ai = recs['v2']['agent_index']
        R_inv = recs['v2']['rotate_mat'].T
        positions = np.asarray(data0['positions'])          # [N, 50, 2] scene frame
        anchor = positions[ai, 19]                          # [2]
        pmask = np.asarray(data0['padding_mask'])

        entry = {
            'scene_idx': int(idx),
            'seq_id': int(data0['seq_id'][0]) if np.ndim(data0['seq_id']) else int(data0['seq_id']),
            'rank': rank,
            'rank_score': float(rank_score[idx]),
            # Selection diagnostics. Compare the mean of b_ratio across the
            # kept scenes against the global 0.648: if it drifted well below,
            # the tail was selected and the frames overstate the effect.
            'b_ratio': float(_ratio),
            'cov_gap': float(_gap),
            'mean_b_arm': {k: float(v) for k, v in _bb.items()},
            'lanes': get_lanes(data0, avm, args.map_radius, anchor),
            'history': positions[ai, :20],                  # [20, 2] scene frame
            'hist_valid': ~pmask[ai, :20],
            'gt': to_scene(recs['v2']['y'], R_inv, anchor),  # [30, 2] scene frame
            'gt_valid': recs['v2']['mask'].copy(),
            'models': {},
        }

        for name, rec in recs.items():
            c, n = tally(rec)
            entry['models'][name] = {
                'modes': np.stack([to_scene(rec['mu'][f], R_inv, anchor)
                                   for f in range(rec['mu'].shape[0])]),   # [F, 30, 2]
                'best': rec['best'],
                'probs': rec['probs'],
                'band': band_corners(rec, args.level, R_inv, anchor),        # [30, 4, 2]
                'counts': c,
                'n': n,
                'mean_b': float(rec['b'][rec['best']][rec['mask']].mean()),
            }
        scenes.append(entry)
        print(f'  [{rank+1}/{args.keep}] scene {idx}  '
              f'spread={rank_score[idx]:.1f}m  best-of-6 FDE={_fde:.2f}m  '
              f'b_ratio={_ratio:.3f}  cov_gap={_gap:+.3f}')

    print(f'  ({rejected} candidate(s) rejected by the FDE / b_ratio / cov_gap '
          f'filters)')
    if scenes:
        _mr = float(np.mean([s['b_ratio'] for s in scenes]))
        _mg = float(np.mean([s['cov_gap'] for s in scenes]))
        print(f'  selected mean b_ratio={_mr:.3f} (global 0.648)   '
              f'mean cov_gap={_mg:+.3f} (global +0.198)')
        if _mr < 0.55 or _mg > 0.28:
            print('  WARNING: selection drifted off the global values -- these '
                  'scenes\n           overstate the v1 pathology. Widen the '
                  'windows.')

    payload = {
        'levels': LEVELS,
        'band_level': args.level,
        'arms': arms,
        'rank_by': args.rank_by,
        'select': {'pct_lo': args.pct_lo, 'pct_hi': args.pct_hi,
                   'max_fde': args.max_fde, 'window': [float(lo), float(hi)],
                   'rejected': rejected},
        'n_scan': n_scan,
        'n_points': n_points,
        'totals': {k: v for k, v in totals.items()},
        'scores': scores,
        'scenes': scenes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'wb') as fh:
        pickle.dump(payload, fh)
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
