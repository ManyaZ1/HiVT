"""Render pass for the ICRA video's coverage montage.

Reads the pickle written by precompute_montage.py and draws one frame per
scene: three panels (from-scratch / v1 / v2) over the same scene with the same
axis limits, each showing the focal agent's min-FDE mode and its 90% Laplace
band, plus a per-scene point tally and a running coverage figure.

No torch, no Argoverse map API -- this is deliberately fast so the visual can
be iterated on without reloading models. See docs/handoff_video_preparation.md.

Colour: categorical slots 1-3 of the data-viz reference palette, unmodified
(blue / orange / aqua). Colour carries model identity only; the ground truth
and history wear ink and grey, so nothing competes with the arm hues. The
three-slot set is documented as passing the all-pairs CVD and normal-vision
floors in both light and dark modes.
"""
from argparse import ArgumentParser
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from shapely.geometry import Polygon, MultiPolygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

# --- design tokens ----------------------------------------------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
LANE = '#d2d2cf'

# History steps to draw/frame. The full 2 s history is about as long as the
# 3 s future; showing all of it halves the scale of the part that matters.
HIST_TAIL = 10

ARMS = [
    ('noKD', 'from-scratch HiVT-32', '#2a78d6'),   # categorical slot 1 (blue)
    ('v1',   'v1  mean-target KD',   '#eb6834'),   # categorical slot 2 (orange)
    ('v2',   'v2  distribution matching', '#1baf7a'),  # categorical slot 3 (aqua)
]


def geom_to_patch(geom, **kw):
    """Convert a shapely (Multi)Polygon to a single matplotlib PathPatch.

    One patch rather than N overlapping ones: overlapping translucent patches
    composite into darker blotches where they intersect, which would make the
    band look structured where it isn't.
    """
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    verts, codes = [], []
    for poly in polys:
        if poly.is_empty:
            continue
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            if len(xy) < 3:
                continue
            verts.extend(xy)
            codes.extend([MplPath.MOVETO]
                         + [MplPath.LINETO] * (len(xy) - 2)
                         + [MplPath.CLOSEPOLY])
    if not verts:
        return None
    return PathPatch(MplPath(np.asarray(verts), codes), **kw)


def band_geom(band, valid, connect=False):
    """Union the per-timestep band rectangles into one region.

    band:  [30, 4, 2] rotated rectangle corners in the scene frame
    valid: [30] bool
    connect: also union the convex hull of each consecutive rectangle pair, so
        a fast-moving agent whose boxes do not overlap still reads as one tube.
        This inflates the drawn region slightly beyond the literal per-timestep
        rectangles; the tally is computed numerically and is unaffected.
    """
    idx = np.where(valid)[0]
    boxes = [Polygon(band[t]) for t in idx]
    boxes = [b for b in boxes if b.is_valid and not b.is_empty]
    if not boxes:
        return None
    parts = list(boxes)
    if connect:
        for a, b in zip(boxes[:-1], boxes[1:]):
            parts.append(MultiPolygon([a, b]).convex_hull)
    return unary_union(parts)


def draw_panel(ax, scene, arm_key, color, level, running):
    m = scene['models'][arm_key]
    valid = scene['gt_valid'].astype(bool)

    for cl in scene['lanes']:
        ax.plot(cl[:, 0], cl[:, 1], color=LANE, lw=0.9, zorder=1,
                solid_capstyle='round')

    # non-selected modes: present so multimodality is visible, recessive so
    # they do not compete with the mode the coverage statistic is about.
    for f in range(m['modes'].shape[0]):
        if f == m['best']:
            continue
        tr = m['modes'][f]
        ax.plot(tr[:, 0], tr[:, 1], color=color, lw=1.0, alpha=0.28, zorder=3)

    geom = band_geom(m['band'], valid, connect=scene.get('_connect', False))
    if geom is not None:
        patch = geom_to_patch(geom, facecolor=color, alpha=0.20,
                              edgecolor=color, linewidth=0.8, zorder=2)
        if patch is not None:
            ax.add_patch(patch)

    hist = scene['history'][scene['hist_valid'].astype(bool)][-HIST_TAIL:]
    ax.plot(hist[:, 0], hist[:, 1], color=INK_2, lw=2.0, zorder=4,
            solid_capstyle='round')
    ax.plot(hist[-1, 0], hist[-1, 1], 'o', color=INK_2, ms=5, zorder=6)

    gt = scene['gt'][valid]
    ax.plot(gt[:, 0], gt[:, 1], color=INK, lw=2.0, ls=(0, (4, 2.5)), zorder=5)
    ax.plot(gt[-1, 0], gt[-1, 1], '*', color=INK, ms=11, zorder=6)

    best = m['modes'][m['best']]
    ax.plot(best[:, 0], best[:, 1], color=color, lw=2.4, zorder=5,
            solid_capstyle='round')

    lvl_i = -1  # p90 is the last level
    k, n = int(m['counts'][lvl_i]), int(m['n'])
    ax.set_title(scene['_titles'][arm_key], fontsize=11, color=INK,
                 fontweight='medium', pad=8)
    ax.set_xlabel(
        f'{int(level*100)}% band contains {k}/{n} points\n'
        f'running coverage  {running:.3f}',
        fontsize=10.5, color=INK_2, labelpad=8)

    # datalim (not the default 'box') lets the shorter axis expand to fill the
    # panel at the same metres-per-pixel. The expansion depends only on the
    # axes-box shape, which is identical across the three panels, so they stay
    # directly comparable.
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def scene_limits(scene, pad=3.0):
    """Shared limits across all three panels, framed on the FUTURE.

    Different limits per panel would make the band-width comparison -- the
    entire point of the shot -- meaningless, so one box is computed over the
    union of all three arms and applied to all.

    Only the last HIST_TAIL steps of history are included. The full 2 s of
    history is roughly as long as the 3 s future, so including it doubles the
    extent and shrinks the bands (2-5 m) to sub-pixel width. Combined with
    aspect='equal', adjustable='datalim' on the axes, this fills the panel
    instead of forcing a square around a horizontal trajectory.
    """
    hist = scene['history'][scene['hist_valid'].astype(bool)][-HIST_TAIL:]
    pts = [hist, scene['gt'][scene['gt_valid'].astype(bool)]]
    for key in ('noKD', 'v1', 'v2'):
        m = scene['models'][key]
        pts.append(m['modes'].reshape(-1, 2))
        pts.append(m['band'].reshape(-1, 2))
    allp = np.concatenate(pts, axis=0)
    return ((allp[:, 0].min() - pad, allp[:, 0].max() + pad),
            (allp[:, 1].min() - pad, allp[:, 1].max() + pad))


def main():
    ap = ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='pickle from precompute')
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--hold', type=float, default=1.5, help='seconds per scene')
    ap.add_argument('--dpi', type=int, default=100)
    ap.add_argument('--scene', type=int, default=None,
                    help='render only this rank (quick iteration)')
    ap.add_argument('--no-connect', dest='connect', action='store_false',
                    help='draw the bare union of per-timestep band boxes; without '
                         'bridging, a fast agent whose boxes do not overlap renders '
                         'as a dashed chain rather than a tube')
    ap.set_defaults(connect=True)
    args = ap.parse_args()

    with open(args.data, 'rb') as fh:
        payload = pickle.load(fh)

    level = payload['band_level']
    scenes = payload['scenes']
    if args.scene is not None:
        scenes = [scenes[args.scene]]

    os.makedirs(args.outdir, exist_ok=True)
    titles = {k: t for k, t, _ in ARMS}

    # Running coverage accumulates in display order, so the counters converge
    # on screen the way the aggregate statistic does.
    run_k = {k: 0 for k, _, _ in ARMS}
    run_n = {k: 0 for k, _, _ in ARMS}

    disconnected = 0
    written = []

    for i, scene in enumerate(scenes):
        scene['_titles'] = titles
        scene['_connect'] = args.connect

        for k, _, _ in ARMS:
            m = scene['models'][k]
            run_k[k] += int(m['counts'][-1])
            run_n[k] += int(m['n'])

        fig = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi, facecolor=SURFACE)
        gs = fig.add_gridspec(1, 3, left=0.02, right=0.98, top=0.83, bottom=0.20,
                              wspace=0.05)
        xlim, ylim = scene_limits(scene)

        for j, (key, _, color) in enumerate(ARMS):
            ax = fig.add_subplot(gs[0, j])
            running = run_k[key] / max(run_n[key], 1)
            draw_panel(ax, scene, key, color, level, running)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            geom = band_geom(scene['models'][key]['band'],
                             scene['gt_valid'].astype(bool), connect=args.connect)
            if isinstance(geom, MultiPolygon) and len(geom.geoms) > 1:
                disconnected += 1

        fig.suptitle(
            f'Argoverse 1 val — scene {scene["seq_id"]}   '
            f'(selected by {payload["rank_by"].replace("_", " ")}, '
            f'rank {scene["rank"]+1}/{len(payload["scenes"])})',
            fontsize=12.5, color=INK, fontweight='semibold', y=0.955)

        handles = [
            Line2D([], [], color=INK_2, lw=2.0, label='observed history (2 s)'),
            Line2D([], [], color=INK, lw=2.0, ls=(0, (4, 2.5)),
                   label='ground-truth future (3 s)'),
            Line2D([], [], color=INK_2, lw=2.4, label='min-FDE mode'),
            Patch(facecolor=INK_2, alpha=0.20, edgecolor=INK_2,
                  label=f'{int(level*100)}% Laplace band'),
            Line2D([], [], color=INK_2, lw=1.0, alpha=0.4, label='other 5 modes'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
                   fontsize=10, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.015))

        fp = os.path.join(args.outdir, f'frame_{i:03d}.png')
        fig.savefig(fp, facecolor=SURFACE)
        plt.close(fig)
        written.append(fp)
        print(f'  [{i+1}/{len(scenes)}] {fp}')

    if disconnected:
        print(f'\n  NOTE: {disconnected} panel(s) produced a disconnected band '
              f'(fast-moving agent, boxes do not overlap).\n'
              f'  Re-run with --connect to bridge consecutive boxes.')

    concat = os.path.join(args.outdir, 'concat.txt')
    with open(concat, 'w') as fh:
        for fp in written:
            fh.write(f"file '{os.path.basename(fp)}'\nduration {args.hold}\n")
        fh.write(f"file '{os.path.basename(written[-1])}'\n")  # last frame needs a repeat

    print(f'\nwrote {len(written)} frames + {concat}')
    print('\nEncode with:')
    print(f'  ffmpeg -y -f concat -safe 0 -i {concat} \\')
    print(f'    -vf "fps={args.fps},format=yuv420p" -c:v libx264 -crf 20 '
          f'-preset slow \\')
    print(f'    {os.path.join(args.outdir, "montage.mp4")}')
    print(f'\n  ({len(written)} scenes x {args.hold}s = '
          f'{len(written)*args.hold:.1f}s of runtime)')


if __name__ == '__main__':
    main()
