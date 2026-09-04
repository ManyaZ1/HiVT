"""
shots.py — the ten shots of the ICRA submission video.

Pure matplotlib and numpy: no torch, no dataset, no map. Everything these
shots draw was precomputed by

  * KD/kd_mode_permutation_test.py --dump_per_scene  -> docs/video/perm_trace500.npz
        (Beat 1: per-scene teacher/student mode geometry, cost matrices,
         optimal permutations. Reproduces the paper's Fig. 2 exactly.)
  * KD/video/precompute_montage.py                   -> docs/video/montage_illustrative.pkl
        (Beat 2: the three-arm coverage montage, gate-verified at
         0.903 / 0.711 / 0.909 / 0.894.)
  * eval.py                                          -> docs/figures/fulldata_reliability.json
        (the reliability curves)

so the whole video re-renders in about a minute and can be iterated on freely.
Numbers quoted in captions that are NOT read from those files are hard-coded
from the paper tables, each with the table it came from named in a comment.

Every shot writes into a FrameWriter and is responsible for its own duration;
build_video.py owns the timeline and checks the total.
"""

import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from KD.video.vstyle import (
    SURFACE, BLUE, ORANGE, AQUA, INK, INK_2, INK_3, HAIRLINE, CMAP_BLUE,
    ARM_COLOR, ARM_LABEL, BAND_TOP,
    new_frame, bare, title, subtitle, caption, legend_chips,
)

PAPER_TITLE = ('Permutation-Invariant Knowledge Distillation\n'
               'for Motion Prediction in Autonomous Vehicles')

# --- from paper Table I (from-scratch baselines) --------------------------- #
BASELINE = [  # (label, params, minFDE)
    ('HiVT-128\n(teacher)', 2_559_993, 0.969),
    ('HiVT-64', 653_369, 1.030),
    ('HiVT-32', 170_073, 1.157),
    ('HiVT-16', 45_929, 1.605),
]
# --- from paper Table II (HiVT-32, lambda=0.5) and Table III (HiVT-16 seeds)  #
DISTILLED = [('HiVT-32', 170_073, 1.050), ('HiVT-16', 45_929, 1.224)]

TABLE2 = {  # paper Table II
    'rows_geom': [('minADE', 0.736, 0.696, 0.698, 0.661),
                  ('minFDE', 1.157, 1.050, 1.050, 0.969),
                  ('MR', 0.122, 0.105, 0.106, 0.092)],
    'rows_calib': [('mixNLL', 26.6, 38.0, 24.2, 21.1),
                   ('calib. err', 0.033, 0.184, 0.028, 0.047),
                   ('mean b', 0.418, 0.268, 0.413, 0.358),
                   ('cov@p90', 0.903, 0.711, 0.909, 0.894)],
}
TABLE3 = [  # paper Table III, HiVT-16 3-seed mean +/- sd
    ('minADE', '0.875 ± 0.021', '0.771 ± 0.005', '-11.8%'),
    ('minFDE', '1.562 ± 0.080', '1.224 ± 0.018', '-21.6%'),
    ('MR', '0.189 ± 0.016', '0.130 ± 0.002', '-31.0%'),
    ('mixNLL', '34.9 ± 1.0', '29.4 ± 0.3', '-15.7%'),
    ('calib. err', '0.0283 ± 0.0010', '0.0229 ± 0.0004', '-19.1%'),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_perm(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_montage(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def load_reliability(path):
    with open(path) as fh:
        return json.load(fh)


def load_efficiency(dirpath='kd_ckpt/_triage_logs'):
    """Measured latency/memory, read from the profiling artefacts rather than
    transcribed. Same files the presentation's efficiency slide was built from
    (KD/profile_efficiency.py wrote them), so the video cannot drift from it.

    Returns {device: {batch: {model: (scene_ms_median, peak_mem_mb)}}}.
    """
    import glob
    import re
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, 'efficiency_*_bs*.json'))):
        m = re.search(r'efficiency_(cpu|gpu)_bs(\d+)\.json$', f)
        if not m:
            continue
        with open(f) as fh:
            d = json.load(fh)
        out.setdefault(m.group(1), {})[int(m.group(2))] = {
            k: (v['latency']['scene_ms_median'], v['latency']['peak_mem_mb'])
            for k, v in d.items()}
    return out


def pick_perm_scene(tr):
    """Representative scene for the pairing shot, by a stated criterion.

    Chosen as: (a) its optimal permutation is the MODAL one, so the frame shows
    the typical relabelling rather than a special case; (b) its identity/optimal
    cost ratio is within 15% of the dataset MEAN, which is the number the
    caption puts on screen -- deliberately NOT the maximum ratio, since
    selecting on the extreme of a quality statistic is what produced a
    misleading frame once already in this project; (c) its student mode spread
    is in the 70-90th percentile, so the modes are legible without being an
    outlier.

    Note on what these modes look like: in the agent-local rotated frame the six
    modes fan out almost entirely LONGITUDINALLY (measured over these 500
    scenes, the largest angular spread between mode endpoints is only ~2 deg).
    They differ in how far the agent travels, not in which way it turns. The
    endpoints therefore string out along the heading axis, which is legible in a
    16:9 window only if the window is sized to the axes box rather than forced
    square -- see _modes_axes.
    """
    import collections
    perm, ci, co = tr['perm'], tr['cost_identity'], tr['cost_optimal']
    ratio = ci / co
    modal = collections.Counter(map(tuple, perm)).most_common(1)[0][0]
    ep = tr['s_loc'][:, :, -1]
    spread = np.linalg.norm(ep[:, :, None] - ep[:, None, :], axis=-1).sum((1, 2)) / 30
    lo, hi = np.percentile(spread, [70, 90])
    ok = np.array([tuple(p) == modal for p in perm]) & (spread >= lo) & (spread <= hi)
    idx = np.where(ok)[0]
    return int(idx[np.argmin(np.abs(ratio[idx] - ratio.mean()))]), np.array(modal)


# --------------------------------------------------------------------------- #
# Small drawing helpers
# --------------------------------------------------------------------------- #
def _modes_axes(fig, rect, tloc, sloc, pad=4.0, focus='all'):
    """Axes framed on the mode geometry.

    focus='ends' frames on the terminal fan of endpoints rather than the whole
    trajectory. The pairing argument is entirely about which endpoint goes with
    which, and a 3-second, 30-metre trajectory renders that fan at a few pixels
    if the window is sized to the trajectory -- the same framing mistake that
    made the montage's uncertainty bands sub-pixel. Trajectories then run in
    from outside the frame, which reads correctly as "arriving from the past".
    """
    ax = fig.add_axes(rect)
    bare(ax)
    if focus == 'ends':
        pts = np.concatenate([tloc[:, -1], sloc[:, -1]])
        margin, pad = 0.16, 1.0
    else:
        pts = np.concatenate([tloc.reshape(-1, 2), sloc.reshape(-1, 2)])
        margin, pad = 0.05, 2.0
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = np.maximum((hi - lo) / 2, 0.5) * (1 + margin) + pad

    # Expand the SHORTER data axis until the window matches the axes box's
    # pixel aspect. Scale then stays equal in both directions with no wasted
    # margin -- forcing a square window instead wastes most of a 16:9 box and
    # is what shrank the mode fan to a smudge.
    box_aspect = (rect[2] * 1280.0) / (rect[3] * 720.0)
    if half[0] / half[1] < box_aspect:
        half[0] = half[1] * box_aspect
    else:
        half[1] = half[0] / box_aspect
    ax.set_xlim(c[0] - half[0], c[0] + half[0])
    ax.set_ylim(c[1] - half[1], c[1] + half[1])
    ax.set_aspect('equal', adjustable='box')
    return ax


def _draw_modes(ax, loc, color, lw=1.6, alpha=1.0, numbers=True, zorder=3,
                nudge=0.0, fontsize=11):
    """Six mode trajectories from the agent origin, with numbered endpoints."""
    for k, tr in enumerate(loc):
        p = np.vstack([[0, 0], tr])
        ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=zorder)
        if numbers:
            e = tr[-1]
            ax.plot(*e, 'o', ms=13, color=color, alpha=alpha,
                    mec=SURFACE, mew=1.6, zorder=zorder + 1)
            ax.text(e[0], e[1] + nudge, str(k), fontsize=fontsize, color='white',
                    ha='center', va='center', fontweight='bold', zorder=zorder + 2)


def _pair_lines(ax, tloc, sloc, pairing, color, lw=2.0, alpha=0.9, ls='-'):
    """Connectors from student mode k to its paired teacher mode."""
    for k, f in enumerate(pairing):
        a, b = sloc[k, -1], tloc[int(f), -1]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, alpha=alpha,
                ls=ls, zorder=2, solid_capstyle='round')


def _best_perm(C):
    """Min-cost one-to-one assignment by brute force over 6! = 720 permutations.

    Mirrors `KD.kd_mode_permutation_test.best_permutation`; duplicated here only
    so this module stays importable without torch, the dataset, or ArgoverseMap.
    """
    import itertools
    rows = np.arange(C.shape[0])
    perms = np.array(list(itertools.permutations(range(C.shape[0]))))
    return perms[int(C[rows[None, :], perms].sum(axis=1).argmin())]


def _table_axes(fig, rect):
    ax = fig.add_axes(rect)
    bare(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


# --------------------------------------------------------------------------- #
# 1. Title  (10 s)   -- anonymous: no authors, no affiliation, no repo
# --------------------------------------------------------------------------- #
def shot01_title(fw):
    def frame(step):
        fig = new_frame()
        fig.text(0.5, 0.60, PAPER_TITLE, fontsize=32, color=INK,
                 fontweight='semibold', ha='center', va='center', linespacing=1.45)
        if step >= 1:
            fig.text(0.5, 0.40, 'Distilling a mixture-density predictor is a structurally '
                                'different problem\nfrom distilling a classifier and needs'
                                ' a different approach.',
                     fontsize=16, color=INK_2, ha='center', va='center', linespacing=1.6)
        if step >= 2:
            fig.text(0.5, 0.22, 'Argoverse 1  ·  HiVT  ·  2.56 M-parameter teacher → 46 k-parameter student',
                     fontsize=13.5, color=INK_3, ha='center', va='center')
        return fig

    fw.add(frame(0), 3.0)
    fw.add(frame(1), 3.0)
    fw.add(frame(2), 4.0)


# --------------------------------------------------------------------------- #
# 2. Why distillation  (20 s)
# --------------------------------------------------------------------------- #
def shot02_why_kd(fw):
    # --- 2a: the capacity cost (10 s) --------------------------------------
    def frontier(n_pts, annotate=False):
        fig = new_frame()
        title(fig, 'Accuracy degrades with smaller models')
        ax = fig.add_axes([0.10, 0.30, 0.82, 0.50])
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(HAIRLINE)
        ax.tick_params(colors=INK_3, labelsize=11)
        ax.grid(True, color=HAIRLINE, lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xscale('log')
        ax.set_xlabel('parameters (log scale)', fontsize=12.5, color=INK_2)
        ax.set_ylabel('minFDE  (m, lower is better)', fontsize=12.5, color=INK_2)
        ax.set_xlim(3e4, 4e6)
        ax.set_ylim(0.88, 1.72)

        pts = BASELINE[:n_pts]
        if len(pts) > 1:
            ax.plot([p[1] for p in pts], [p[2] for p in pts], color=INK_3,
                    lw=1.6, alpha=0.55, zorder=2)
        for lab, pr, fde in pts:
            ax.plot(pr, fde, 'o', ms=13, color=BLUE, mec=SURFACE, mew=2, zorder=4)
            ax.annotate(f'{lab}\n{fde:.3f} m', (pr, fde), textcoords='offset points',
                        xytext=(0, 20), ha='center', fontsize=11.5, color=INK_2,
                        linespacing=1.3)
        if annotate:
            ax.annotate('', xy=(45_929, 1.605), xytext=(170_073, 1.157),
                        arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2.2))
            ax.text(9.5e4, 1.42, '+0.45 m', fontsize=14, color=ORANGE,
                    fontweight='semibold', ha='center')
            ax.annotate('', xy=(170_073, 1.157), xytext=(653_369, 1.030),
                        arrowprops=dict(arrowstyle='<->', color=INK_3, lw=1.8))
            ax.text(3.5e5, 1.06, '+0.13 m', fontsize=12.5, color=INK_3, ha='center')
        caption(fig, 'Smaller models demand less memory',
                highlight=None if not annotate else
                'Accuracy falls faster than linearly.')
        return fig

    for n in (1, 2, 3, 4):
        fw.add(frontier(n), 1.1)
    fw.add(frontier(4, annotate=True), 5.6)

    # --- 2b: the same behaviour lives in a different slot (10 s) -----------
    TEACHER = [('mode 1', 'turn left'), ('mode 2', 'go straight'), ('mode 3', 'turn right')]
    STUDENT = [('mode 1', 'go straight'), ('mode 2', 'turn right'), ('mode 3', 'turn left')]
    BY_MEANING = [(0, 2), (1, 0), (2, 1)]      # teacher row -> student row, same behaviour
    ROW_Y = [0.80, 0.50, 0.20]
    BOX_H, LX, RX, BW = 0.20, 0.03, 0.63, 0.34

    def slots(stage):
        fig = new_frame()
        title(fig, 'Both models learn the same behaviours — in a different order')
        ax = fig.add_axes([0.08, 0.235, 0.84, 0.56])
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.text(LX + BW / 2, 1.02, 'Teacher   HiVT-128', fontsize=14, color=ORANGE,
                ha='center', va='bottom', fontweight='semibold')
        ax.text(RX + BW / 2, 1.02, 'Student   HiVT-32', fontsize=14, color=BLUE,
                ha='center', va='bottom', fontweight='semibold')

        for (col, x, items) in ((ORANGE, LX, TEACHER), (BLUE, RX, STUDENT)):
            for r, (slot, behaviour) in enumerate(items):
                y = ROW_Y[r]
                ax.add_patch(plt.Rectangle((x, y - BOX_H / 2), BW, BOX_H,
                                           facecolor=col, edgecolor='none', zorder=3))
                ax.text(x + BW / 2, y + 0.035, slot, fontsize=13.5, color='white',
                        ha='center', va='center', fontweight='bold', zorder=4)
                ax.text(x + BW / 2, y - 0.038, behaviour, fontsize=12.5, color='white',
                        ha='center', va='center', zorder=4)

        if stage >= 1:                       # what index-aligned KD pairs
            for r in range(3):
                ax.annotate('', xy=(RX - 0.005, ROW_Y[r]),
                            xytext=(LX + BW + 0.005, ROW_Y[r]),
                            arrowprops=dict(arrowstyle='-|>', color=INK_3, lw=2.0,
                                            ls=(0, (5, 3)), shrinkA=0, shrinkB=0))
        if stage >= 2:                       # what actually corresponds
            for tr_, sr in BY_MEANING:
                ax.annotate('', xy=(RX - 0.005, ROW_Y[sr]),
                            xytext=(LX + BW + 0.005, ROW_Y[tr_]),
                            arrowprops=dict(arrowstyle='-|>', color=AQUA, lw=2.6,
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle='arc3,rad=-0.18'))

        chips = [(ORANGE, 'teacher slots'), (BLUE, 'student slots')]
        if stage >= 1:
            chips.append((INK_3, 'paired by index'))
        if stage >= 2:
            chips.append((AQUA, 'the same behaviour'))
        legend_chips(fig, chips, y=0.185, gap=0.20)

        if stage == 0:
            caption(fig, ['Distillation recovers that gap for free — a training-time loss only, '
                          'with the teacher cached offline.',
                          'But both models were trained winner-takes-all, so nothing ever fixed '
                          'which slot holds which behaviour.'])
        elif stage == 1:
            caption(fig, ['Standard KD pairs teacher slot k with student slot k.'],
                    highlight='It is matching “turn left” against “go straight”.')
        else:
            caption(fig, ['The same behaviour is there — it just lives in a different slot.'],
                    highlight='Match by meaning, not by index.')
        return fig

    fw.add(slots(0), 3.4)
    fw.add(slots(1), 3.2)
    fw.add(slots(2), 3.4)


# --------------------------------------------------------------------------- #
# 3. BEAT 1 — the mode-permutation problem  (34 s)
# --------------------------------------------------------------------------- #
def shot03_permutation(fw, tr):
    """One slide. The real per-scene cost matrix -- the matrix a Hungarian
    matcher would actually be handed -- with the identity pairing and the
    optimal pairing marked on it, and the validation-set statistics beside it.

    Earlier this was three animated sub-shots (winner-takes-all, the pairing,
    the matrix accumulating over 500 scenes). The accumulation animation was
    cut: it spent 12 s establishing a number the caption states in one line,
    and the conceptual setup now happens in shot 2b.
    """
    import collections

    sc, modal = pick_perm_scene(tr)
    C = tr['cost'][sc]
    ci, co = float(tr['cost_identity'][sc]), float(tr['cost_optimal'][sc])

    perm = tr['perm']
    n = len(perm)
    n_identity = int(sum(tuple(p) == tuple(range(6)) for p in perm))
    modal_share = collections.Counter(map(tuple, perm)).most_common(1)[0][1] / n
    mean_id = float(tr['cost_identity'].mean())
    mean_opt = float(tr['cost_optimal'].mean())
    ratio = float((tr['cost_identity'] / tr['cost_optimal']).mean())

    def frame(stage):
        fig = new_frame()
        title(fig, 'The pairing standard KD assumes is never the right one')

        axm = fig.add_axes([0.075, 0.275, 0.34, 0.50])
        axm.set_facecolor(SURFACE)
        axm.imshow(C, cmap=CMAP_BLUE, zorder=1)
        axm.set_xticks(range(6))
        axm.set_yticks(range(6))
        axm.tick_params(colors=INK_2, labelsize=10.5, length=0)
        axm.set_xlabel('teacher mode', fontsize=12.5, color=INK_2, labelpad=6)
        axm.set_ylabel('student mode', fontsize=12.5, color=INK_2, labelpad=6)
        axm.set_title('teacher/student mode distance  (m)', fontsize=12.5,
                      color=INK_2, pad=10)
        for sp in axm.spines.values():
            sp.set_color(HAIRLINE)
        for (r, c_) in np.ndindex(C.shape):
            axm.text(c_, r, f'{C[r, c_]:.0f}', ha='center', va='center',
                     fontsize=10, zorder=3,
                     color='white' if C[r, c_] > 0.55 * C.max() else INK_2)
        for k in range(6):
            axm.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False,
                                        ec=INK, lw=2.2, zorder=4))
        if stage >= 1:
            for k, f in enumerate(modal):
                axm.add_patch(plt.Rectangle((int(f) - .5, k - .5), 1, 1, fill=False,
                                            ec=AQUA, lw=2.8, zorder=5))

        x = 0.50
        fig.text(x, 0.735, 'on this scene', fontsize=12.5, color=INK_3, ha='left')
        fig.text(x, 0.665, 'pairing by index', fontsize=14, color=INK_2, ha='left')
        fig.text(0.945, 0.665, f'{ci:.2f} m', fontsize=20, color=INK, ha='right',
                 fontweight='semibold')
        if stage >= 1:
            fig.text(x, 0.585, 'optimal pairing', fontsize=14, color=INK_2, ha='left')
            fig.text(0.945, 0.585, f'{co:.2f} m', fontsize=20, color=AQUA, ha='right',
                     fontweight='semibold')
            fig.add_artist(plt.Line2D([x, 0.945], [0.525, 0.525], color=HAIRLINE,
                                      lw=1.0, transform=fig.transFigure))
            fig.text(x, 0.465, f'over {n} validation scenes', fontsize=12.5,
                     color=INK_3, ha='left')
            rows = [
                ('identity pairing optimal in', f'{n_identity} scenes'),
                ('one non-identity permutation', f'{modal_share * 100:.1f}%'),
                ('mean distance, index vs optimal', f'{mean_id:.2f} → {mean_opt:.2f} m'),
                ('index pairing is worse by', f'{ratio:.2f}×'),
            ]
            for i, (lab, val) in enumerate(rows):
                y = 0.395 - i * 0.062
                fig.text(x, y, lab, fontsize=13, color=INK_2, ha='left')
                fig.text(0.945, y, val, fontsize=13.5, color=INK, ha='right',
                         fontweight='semibold')

        legend_chips(fig, [(INK, 'paired by index'), (AQUA, 'optimal pairing')]
                     if stage >= 1 else [(INK, 'paired by index')], y=0.195, gap=0.22)

        if stage == 0:
            caption(fig, ['Index-aligned KD reads straight down the diagonal — and picks some '
                          'of the worst cells in the matrix.'])
        else:
            caption(fig, ['Solving the assignment properly gives a different permutation on '
                          'essentially every scene.'],
                    highlight='The identity pairing is optimal in 0% of 500 validation scenes.')
        return fig

    fw.add(frame(0), 5.0)
    fw.add(frame(1), 9.0)


# --------------------------------------------------------------------------- #
# 4. The fix  (18 s)
# --------------------------------------------------------------------------- #
def shot04_objective(fw, tr):
    """Both halves argue on the SAME cost matrix shot 3 introduced, because the
    three approaches are three different readings of it:

        index-aligned   -> the diagonal cells
        Hungarian       -> one cell per row, and it moves as training moves
        ours            -> a whole column: no cell is ever selected

    That last reading is what "never forms a pairing" looks like, and it is far
    clearer on the matrix than on the trajectories (whose modes fan
    longitudinally and render as an unreadable streak -- see pick_perm_scene).
    """
    sc, modal = pick_perm_scene(tr)
    C = tr['cost'][sc]

    # Two training steps: the student's modes shift slightly, so the cost matrix
    # shifts and the argmin genuinely lands on a different permutation. The
    # perturbation is fixed, not random, so the build stays reproducible.
    nudge = np.zeros_like(C)
    nudge[0, 0] = nudge[3, 2] = 2.6
    STEPS = [(C, _best_perm(C)), (C + nudge, _best_perm(C + nudge))]

    def matrix_axes(fig, mat):
        ax = fig.add_axes([0.075, 0.275, 0.34, 0.50])
        ax.set_facecolor(SURFACE)
        ax.imshow(mat, cmap=CMAP_BLUE, vmin=C.min(), vmax=C.max(), zorder=1)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.tick_params(colors=INK_2, labelsize=10.5, length=0)
        ax.set_xlabel('teacher mode', fontsize=12.5, color=INK_2, labelpad=6)
        ax.set_ylabel('student mode', fontsize=12.5, color=INK_2, labelpad=6)
        ax.set_title('teacher/student mode distance  (m)', fontsize=12.5,
                     color=INK_2, pad=10)
        for sp in ax.spines.values():
            sp.set_color(HAIRLINE)
        for (r, c_) in np.ndindex(mat.shape):
            ax.text(c_, r, f'{mat[r, c_]:.0f}', ha='center', va='center',
                    fontsize=10, zorder=3,
                    color='white' if mat[r, c_] > 0.55 * C.max() else INK_2)
        return ax

    # --- 4a: why not Hungarian (7 s) ---------------------------------------
    def hungarian(which):
        mat, perm = STEPS[which % 2]
        fig = new_frame()
        title(fig, 'Why not just solve the assignment problem?')
        ax = matrix_axes(fig, mat)
        for k, f in enumerate(perm):
            ax.add_patch(plt.Rectangle((int(f) - .5, k - .5), 1, 1, fill=False,
                                       ec=INK, lw=2.8, zorder=5))
        fig.text(0.50, 0.665, f'training step {2100 + which % 2}', fontsize=15,
                 color=INK_3, ha='left')
        fig.text(0.50, 0.575, 'chosen permutation', fontsize=13.5, color=INK_2, ha='left')
        fig.text(0.50, 0.495, '  '.join(str(int(f)) for f in perm), fontsize=25,
                 color=INK, ha='left', fontweight='semibold')
        fig.text(0.50, 0.395, 'the student’s modes move a little,\nand the argmin jumps to a\n'
                              'different assignment entirely.',
                 fontsize=13, color=INK_2, ha='left', va='top', linespacing=1.7)
        legend_chips(fig, [(INK, 'the assignment chosen this step')], y=0.195)
        caption(fig, ['Hungarian matching re-solves a hard, non-differentiable assignment '
                      'every step, and is undefined when the two models have different mode '
                      'counts.'],
                highlight='The pairing also flips discontinuously between steps.')
        return fig

    for i in range(6):
        fw.add(hungarian(i), 0.55)
    fw.add(hungarian(0), 3.7)

    # --- 4b: score against the whole mixture (11 s) ------------------------
    f_star = int(np.argmax(tr['t_probs'][sc]))

    def mixture(stage):
        fig = new_frame()
        title(fig, 'Our objective never forms a pairing at all')
        ax = matrix_axes(fig, C)
        if stage >= 1:
            # a whole column: teacher mode f is scored against EVERY student mode
            ax.add_patch(plt.Rectangle((f_star - .5, -.5), 1, 6, fill=False,
                                       ec=AQUA, lw=3.0, zorder=5))
        fig.text(0.50, 0.700, 'score each teacher mode under the\nstudent’s ENTIRE mixture density',
                 fontsize=15, color=INK, ha='left', va='center', linespacing=1.6)
        fig.text(0.50, 0.545,
                 r'$\mathcal{L} = -\sum_f \pi^T_f \, \log \sum_k \pi^S_k \, '
                 r'\mathrm{Lap}\left(\mu^T_f \mid \mu^S_k,\, b^S_k\right)$',
                 fontsize=18, color=INK, ha='left', va='center')
        if stage >= 1:
            fig.text(0.50, 0.395, 'No cell is ever selected. Every student mode\n'
                                  'contributes, weighted by how well it explains\n'
                                  'the teacher mode — by value, never by index.',
                     fontsize=13, color=INK_2, ha='left', va='top', linespacing=1.7)
        legend_chips(fig, [(AQUA, 'one teacher mode vs the whole student mixture')]
                     if stage >= 1 else [], y=0.195)
        caption(fig, ['The student enters only through its mixture density, so reordering '
                      'either model’s modes changes nothing.'],
                highlight='Permutation-invariant by construction — and defined for any mode counts.')
        return fig

    fw.add(mixture(0), 3.6)
    fw.add(mixture(1), 7.4)


# --------------------------------------------------------------------------- #
# 5. The trap: what the obvious target does  (12 s)
# --------------------------------------------------------------------------- #
def shot05_pathology(fw):
    def frame(stage):
        fig = new_frame()
        title(fig, 'Do that with the teacher’s mean trajectories, and geometry improves')

        ax = fig.add_axes([0.075, 0.40, 0.36, 0.38])
        bare(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.86, 'HiVT-32  minFDE', fontsize=13, color=INK_2, ha='center')
        ax.text(0.22, 0.52, '1.157', fontsize=30, color=INK_3, ha='center')
        ax.text(0.5, 0.52, '→', fontsize=26, color=INK_3, ha='center')
        ax.text(0.79, 0.52, '1.050', fontsize=30, color=AQUA, ha='center',
                fontweight='semibold')
        ax.text(0.5, 0.22, 'no KD                        v1', fontsize=12,
                color=INK_3, ha='center')

        if stage >= 1:
            fig.text(0.50, 0.66, 'But the stationary scale it converges to is',
                     fontsize=14, color=INK_2, ha='left', va='center')
            fig.text(0.50, 0.525,
                     r'$b^{S\star} \;=\; \frac{\sum_f \pi^T_f\, r_{fk}\,'
                     r'\left| \mu^T_f - \mu^S_k \right|}{\sum_f \pi^T_f\, r_{fk}}$',
                     fontsize=23, color=INK, ha='left', va='center')
        if stage >= 2:
            fig.text(0.50, 0.375, 'a weighted average of the student’s own residuals.\n'
                                  'The teacher’s spread $b^T_f$ appears nowhere.',
                     fontsize=14, color=ORANGE, ha='left', va='center', linespacing=1.7)

        if stage == 0:
            caption(fig, 'Scoring the teacher’s mean treats it as a zero-width point mass.')
        elif stage == 1:
            caption(fig, 'The location term drives those residuals toward zero — and drags the '
                         'scale down with them.')
        else:
            caption(fig, ['v1 and v2 tie on minADE, minFDE and MR.'],
                    highlight='A field that reports only best-of-K geometry cannot see this failure at all.')
        return fig

    fw.add(frame(0), 3.0)
    fw.add(frame(1), 4.0)
    fw.add(frame(2), 5.0)


# --------------------------------------------------------------------------- #
# 6. BEAT 2 — the coverage montage  (26 s)
# --------------------------------------------------------------------------- #
def shot06_montage(fw, mont):
    from KD.video.render_montage import draw_panel, scene_limits

    arms = [('noKD', ARM_LABEL['noKD'], BLUE),
            ('v1', ARM_LABEL['v1'], ORANGE),
            ('v2', ARM_LABEL['v2'], AQUA)]
    scenes = mont['scenes']
    level = mont['band_level']
    n_pts = mont['n_points']
    totals = mont['totals']

    def lead():
        fig = new_frame()
        title(fig, 'So what does the obvious objective actually cost?')
        fig.text(0.5, 0.56, 'v1’s nominal 90% intervals still contain the truth\n'
                            'on roughly 7 of every 10 scenes.',
                 fontsize=19, color=INK, ha='center', va='center', linespacing=1.7)
        fig.text(0.5, 0.36, 'No single scene settles this. Watch the running total.',
                 fontsize=15, color=ORANGE, ha='center', va='center')
        caption(fig, ['16 validation scenes, selected by highest mode spread (the paper’s own '
                      'diversity measure) — not by eye.',
                      'Counting every (step, coordinate) point inside each model’s own 90% '
                      'Laplace band: 60 points per scene.'])
        return fig

    fw.add(lead(), 2.5)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    li = list(mont['levels']).index(level)
    run_k = {k: 0 for k, _, _ in arms}
    run_n = {k: 0 for k, _, _ in arms}

    handles = [
        Line2D([], [], color=INK_2, lw=2.0, label='observed history'),
        Line2D([], [], color=INK, lw=2.0, ls=(0, (4, 2.5)), label='ground-truth future'),
        Line2D([], [], color=INK_2, lw=2.4, label='min-FDE mode'),
        Patch(facecolor=INK_2, alpha=0.20, edgecolor=INK_2, label='90% Laplace band'),
        Line2D([], [], color=INK_2, lw=1.0, alpha=0.4, label='other 5 modes'),
    ]

    for si, scene in enumerate(scenes):
        # draw_panel reads its own per-arm title and the band-bridging flag off
        # the scene dict, exactly as render_montage.main sets them up.
        scene['_titles'] = {k: lbl for k, lbl, _ in arms}
        scene['_connect'] = True
        for key, _, _ in arms:
            run_k[key] += int(scene['models'][key]['counts'][li])
            run_n[key] += int(scene['models'][key]['n'])

        fig = new_frame()
        title(fig, f'Argoverse 1 validation — scene {scene["seq_id"]}', y=0.955, size=15)
        subtitle(fig, f'selected by mode spread — {si + 1} of {len(scenes)}',
                 y=0.907, size=12)
        xlim, ylim = scene_limits(scene)
        for ai, (key, label, color) in enumerate(arms):
            ax = fig.add_axes([0.035 + ai * 0.322, 0.315, 0.29, 0.525])
            draw_panel(ax, scene, key, color, level, run_k[key] / max(run_n[key], 1))
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
                   fontsize=10.5, labelcolor=INK_2, bbox_to_anchor=(0.5, BAND_TOP + 0.015))
        caption(fig, 'Every point outside a model’s own 90% interval is a point where it told '
                     'the planner it was more certain than it turned out to be.')
        fw.add(fig, 1.8)

    # land on the paper's numbers
    fig = new_frame()
    title(fig, 'The running totals converge on the paper’s coverage')
    ax = fig.add_axes([0.14, 0.30, 0.72, 0.48])
    bare(ax)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    for i, (key, label, color) in enumerate(arms):
        cov = totals[key][li] / n_pts
        ax.text(i + 0.5, 0.78, label.split('  ')[0], fontsize=15, color=INK,
                ha='center', fontweight='semibold')
        ax.text(i + 0.5, 0.62, label.split('  ')[-1] if '  ' in label else '',
                fontsize=12, color=INK_2, ha='center')
        ax.text(i + 0.5, 0.32, f'{cov:.3f}', fontsize=46, color=color,
                ha='center', fontweight='semibold')
        ax.text(i + 0.5, 0.13, 'cov@p90', fontsize=12.5, color=INK_3, ha='center')
    caption(fig, ['Full validation set, all 39,472 scenes. Nominal level: 90%.'],
            highlight='v1 keeps every bit of the geometry — and under-covers by 19 points.')
    fw.add(fig, 4.1)


# --------------------------------------------------------------------------- #
# 7. Reliability curves  (12 s)
# --------------------------------------------------------------------------- #
def shot07_reliability(fw, rel):
    key_of = {'noKD': 'nokd', 'v1': 'v1', 'v2': 'v2'}
    nominal = np.arange(1, 10) / 10.0

    def curves(upto, final=False):
        fig = new_frame()
        title(fig, 'Reliability: is the predicted uncertainty honest?')
        ax = fig.add_axes([0.34, 0.285, 0.35, 0.545])
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(HAIRLINE)
        ax.grid(True, color=HAIRLINE, lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_3, labelsize=10.5)
        ax.plot([0, 1], [0, 1], color=INK_3, lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.text(0.62, 0.52, 'perfectly calibrated', fontsize=10.5, color=INK_3,
                rotation=39, ha='center', va='center')
        ax.set_xlabel('nominal coverage', fontsize=12, color=INK_2)
        ax.set_ylabel('empirical coverage', fontsize=12, color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # noKD and v2 very nearly coincide -- that agreement is the result, but
        # drawn at equal weight the aqua simply hides the blue and the viewer
        # sees two curves against a three-item legend. Draw the baseline as a
        # wide soft band underneath and v2 as a thin line on top, so both stay
        # visible AND the coincidence is what you notice.
        style = {'noKD': dict(lw=7.0, alpha=0.38, marker=None, zorder=3),
                 'v1': dict(lw=2.6, marker='o', ms=6, zorder=5),
                 'v2': dict(lw=2.4, marker='o', ms=5, zorder=6)}
        for arm in ('noKD', 'v1', 'v2'):
            emp = np.array([rel[key_of[arm]][f'cov_p{int(round(p * 100))}']
                            for p in nominal])
            ax.plot(nominal[:upto], emp[:upto], color=ARM_COLOR[arm],
                    mec=SURFACE, mew=1.4, solid_capstyle='round', **style[arm])
        if final:
            e90 = rel['v1']['cov_p90']
            ax.annotate(f'{e90:.3f}', xy=(0.9, e90), xytext=(0.5, 0.20),
                        fontsize=15, color=ORANGE, fontweight='semibold',
                        arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.8))
        legend_chips(fig, [(BLUE, ARM_LABEL['noKD']), (ORANGE, ARM_LABEL['v1']),
                           (AQUA, ARM_LABEL['v2'])], y=0.185, gap=0.245)
        if final:
            fig.text(0.945, 0.60, 'v2 sits on top of\nthe no-KD baseline',
                     fontsize=11.5, color=INK_3, ha='right', va='center',
                     linespacing=1.5)
        if not final:
            caption(fig, 'A curve below the diagonal means the predicted intervals are too narrow.')
        else:
            caption(fig, ['v1 under-covers at every nominal level; v2 lies on the from-scratch '
                          'reference (0.909 vs 0.903).'],
                    highlight='Removes the calibration penalty; gives up none of the geometry.')
        return fig

    for u in range(1, 10):
        fw.add(curves(u), 0.42)
    fw.add(curves(9), 3.2)
    fw.add(curves(9, final=True), 5.0)


# --------------------------------------------------------------------------- #
# 8. Results  (30 s)
# --------------------------------------------------------------------------- #
def _table_header(ax, cols, x0, dx, y, size=13.5):
    for i, c in enumerate(cols):
        ax.text(x0 + i * dx, y, c, fontsize=size, color=INK, ha='center',
                fontweight='semibold')


def shot08_results(fw):
    # --- 8a: Table II (12 s) -----------------------------------------------
    def table2(stage):
        fig = new_frame()
        title(fig, 'HiVT-32, λ = 0.5 — same geometry, opposite honesty')
        ax = _table_axes(fig, [0.16, 0.20, 0.70, 0.63])
        x0, dx = 0.34, 0.185
        _table_header(ax, ['no KD', 'v1', 'v2', 'teacher'], x0, dx, 0.95)
        ax.plot([0.02, 0.98], [0.90, 0.90], color=HAIRLINE, lw=1.2)

        y = 0.80
        for name, *vals in TABLE2['rows_geom']:
            ax.text(0.02, y, name, fontsize=13.5, color=INK_2, ha='left')
            for i, v in enumerate(vals):
                hot = stage >= 1 and i in (1, 2)
                ax.text(x0 + i * dx, y, f'{v:.3f}', fontsize=13.5,
                        color=INK if hot else INK_2, ha='center',
                        fontweight='semibold' if hot else 'normal')
            y -= 0.105
        if stage >= 1:
            ax.text(0.99, 0.695, 'tie', fontsize=14, color=INK_3, ha='right', style='italic')

        if stage >= 2:
            ax.plot([0.02, 0.98], [y + 0.035, y + 0.035], color=HAIRLINE, lw=1.2)
            y -= 0.02
            for name, *vals in TABLE2['rows_calib']:
                ax.text(0.02, y, name, fontsize=13.5, color=INK_2, ha='left')
                for i, v in enumerate(vals):
                    hot = stage >= 3 and i in (1, 2)
                    col = (ORANGE if i == 1 else AQUA) if hot else INK_2
                    ax.text(x0 + i * dx, y, f'{v:.3f}' if v < 10 else f'{v:.1f}',
                            fontsize=13.5, color=col, ha='center',
                            fontweight='semibold' if hot else 'normal')
                y -= 0.105

        if stage == 0:
            caption(fig, 'Geometry first — the metrics a leaderboard reports.')
        elif stage == 1:
            caption(fig, 'Both variants close most of the gap to the teacher, and they tie.')
        elif stage == 2:
            caption(fig, 'Now the metrics that describe the predicted distribution.')
        else:
            caption(fig, ['v1: calibration error up 5.6×, mean scale contracted 36%, '
                          '90% intervals covering 71%.'],
                    highlight='v2 holds calibration below the from-scratch baseline — and below the teacher.')
        return fig

    for s, d in ((0, 2.4), (1, 3.0), (2, 2.4), (3, 4.2)):
        fw.add(table2(s), d)

    # --- 8b: Table III, HiVT-16 seeds (10 s) -------------------------------
    def table3(stage):
        fig = new_frame()
        title(fig, 'HiVT-16 — 46 k parameters, 55× smaller than the teacher')
        subtitle(fig, 'full data, 64 epochs, 3 independent seeds per arm (mean ± sd)')
        ax = _table_axes(fig, [0.10, 0.235, 0.80, 0.55])
        _table_header(ax, ['no KD', 'v2,  λ = 1.0', 'Δ'], 0.40, 0.245, 0.95)
        ax.plot([0.0, 1.0], [0.89, 0.89], color=HAIRLINE, lw=1.2)
        y = 0.76
        for name, a, b, dlt in TABLE3:
            ax.text(0.0, y, name, fontsize=13.5, color=INK_2, ha='left')
            ax.text(0.40, y, a, fontsize=13.5, color=INK_2, ha='center')
            ax.text(0.645, y, b, fontsize=13.5, color=AQUA if stage >= 1 else INK_2,
                    ha='center', fontweight='semibold' if stage >= 1 else 'normal')
            ax.text(0.89, y, dlt, fontsize=13.5,
                    color=INK if stage >= 1 else INK_3, ha='center',
                    fontweight='semibold' if stage >= 1 else 'normal')
            y -= 0.148
        if stage == 0:
            caption(fig, 'The smallest student has the most to recover — and gains the most.')
        else:
            caption(fig, ['Geometry and calibration improve together, and the seed clouds do '
                          'not overlap:', 'the worst v2 run (1.24) still beats the best no-KD run (1.47).'],
                    highlight='Not seed noise.')
        return fig

    fw.add(table3(0), 4.0)
    fw.add(table3(1), 6.0)

    # --- 8c: the frontier (8 s) --------------------------------------------
    def frontier():
        fig = new_frame()
        title(fig, 'Distillation buys back most of a size class')
        ax = fig.add_axes([0.10, 0.29, 0.82, 0.50])
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(HAIRLINE)
        ax.grid(True, color=HAIRLINE, lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_3, labelsize=11)
        ax.set_xscale('log')
        ax.set_xlim(3e4, 4e6)
        ax.set_ylim(0.88, 1.72)
        ax.set_xlabel('parameters (log scale)', fontsize=12.5, color=INK_2)
        ax.set_ylabel('minFDE  (m, lower is better)', fontsize=12.5, color=INK_2)
        ax.plot([p[1] for p in BASELINE], [p[2] for p in BASELINE], color=INK_3,
                lw=1.6, alpha=0.5, zorder=2)
        for lab, pr, fde in BASELINE:
            ax.plot(pr, fde, 'o', ms=12, color=BLUE, mec=SURFACE, mew=2, zorder=4)
            ax.annotate(lab.replace('\n', ' '), (pr, fde), textcoords='offset points',
                        xytext=(0, 17), ha='center', fontsize=11, color=INK_2)
        for lab, pr, fde in DISTILLED:
            ax.plot(pr, fde, 'o', ms=13, color=AQUA, mec=SURFACE, mew=2, zorder=6)
            ax.annotate(f'{lab} + v2\n{fde:.3f} m', (pr, fde), textcoords='offset points',
                        xytext=(6, -34), ha='left', fontsize=11.5, color=AQUA,
                        linespacing=1.3, fontweight='semibold')
        for (lab, pr, base) in [('HiVT-32', 170_073, 1.157), ('HiVT-16', 45_929, 1.605)]:
            tgt = dict(DISTILLED and {d[0]: d[2] for d in DISTILLED})[lab]
            ax.annotate('', xy=(pr, tgt + 0.012), xytext=(pr, base - 0.012),
                        arrowprops=dict(arrowstyle='-|>', color=AQUA, lw=2.4))
        ax.annotate('', xy=(45_929, 1.224), xytext=(170_073, 1.157),
                    arrowprops=dict(arrowstyle='-', color=INK_3, lw=1.2, ls=':'))
        legend_chips(fig, [(BLUE, 'from scratch'), (AQUA, 'distilled with v2')],
                     y=0.175, gap=0.22)
        caption(fig, ['A 46 k-parameter student lands within 5.8% of an un-distilled model '
                      'with 3.7× the parameters.'],
                highlight='~83% of a size class, recovered — at no inference cost.')
        return fig

    fw.add(frontier(), 7.5)


# --------------------------------------------------------------------------- #
# 9. Takeaway  (14 s)  + 10. anonymous reprise (4 s)
# --------------------------------------------------------------------------- #
def shot09_takeaway(fw):
    lines = [
        (BLUE, 'Diagnosis: index-aligned KD is ill-posed \n',
         'Winner-takes-all training leaves the ordering arbitrary \n'
         '(identity pairing is optimal in 0% of scenes)'),
        (ORANGE, 'A permutation-invariant, matching-free mixture objective corrects it',
        ),
        (AQUA, 'The output distributions should be matched, not the means.\n'
        'v1 breaks calibration by scale shrinkage \n'
         'v2 = forward mode-covering KL fixes it'),
         ( INK, 'Distillation recovers most of a size class at no inference cost \n')
    ]

    def frame(n):
        fig = new_frame()
        title(fig, 'Takeaway', y=0.90, size=22)
        y = 0.73
        for i in range(n):
            # entries may be (colour, head) or (colour, head, body); a
            # head-only entry carrying newlines is split so its first line is
            # the heading and the rest reads as body, matching the others.
            color, head, body = (tuple(lines[i]) + ('',))[:3]
            if not body.strip() and '\n' in head.strip():
                head, _, body = head.strip().partition('\n')
            fig.patches.append(plt.Rectangle((0.055, y - 0.055), 0.006, 0.095,
                                             transform=fig.transFigure,
                                             facecolor=color, edgecolor='none'))
            fig.text(0.085, y + 0.020, head.strip(), fontsize=16, color=INK,
                     fontweight='semibold', ha='left', va='center')
            if body.strip():
                fig.text(0.085, y - 0.040, body.strip(), fontsize=12.5, color=INK_2,
                         ha='left', va='top', linespacing=1.55)
            y -= 0.16
        return fig

    for i, hold in enumerate((3.0, 3.0, 3.0, 5.0)):
        fw.add(frame(i + 1), hold)


def shot10_reprise(fw):
    fig = new_frame()
    fig.text(0.5, 0.56, PAPER_TITLE, fontsize=27, color=INK, fontweight='semibold',
             ha='center', va='center', linespacing=1.45)
    fig.text(0.5, 0.36, 'Code and trained students released on acceptance.',
             fontsize=14, color=INK_3, ha='center', va='center')
    fw.add(fig, 3.5)


# --------------------------------------------------------------------------- #
# 11. Deployment footprint  (9 s)
# --------------------------------------------------------------------------- #
def shot11_efficiency(fw, eff):
    """Answers the reviewer question the handoff flags as the main vulnerability:
    the teacher is only 2.56 M parameters, so who needs to compress it?

    Every number is read from the profiling JSONs. The honest bottom line is on
    screen too -- "55x smaller" is a memory claim, not a speed claim, and a
    reviewer who catches that oversell is worse than the caveat itself.
    """
    T, S16, S32 = 'teacher-128', 'student-16', 'student-32'
    PARAM = {T: 2_559_993, S32: 170_073, S16: 45_929}      # exact, architecture-determined

    gpu = eff['gpu']
    cpu = eff['cpu']
    batches = sorted(gpu)
    mem_ratio = [gpu[b][T][1] / gpu[b][S16][1] for b in batches]
    cpu_best_b = max(cpu, key=lambda b: cpu[b][T][0] / cpu[b][S16][0])
    cpu_speed = cpu[cpu_best_b][T][0] / cpu[cpu_best_b][S16][0]
    gpu_lo = min(gpu[b][T][0] / gpu[b][S16][0] for b in batches)
    gpu_hi = max(gpu[b][T][0] / gpu[b][S16][0] for b in batches)

    tiles = [
        (f'{PARAM[T] // PARAM[S16]}×', 'fewer parameters', '2.56 M → 46 k', BLUE),
        (f'{max(mem_ratio):.1f}×', 'less GPU memory', f'at batch {batches[-1]}', AQUA),
        (f'{cpu_speed:.1f}×', 'faster on CPU', f'at batch {cpu_best_b}', ORANGE),
    ]

    def frame(stage):
        fig = new_frame()
        title(fig, 'What the 46 k-parameter student actually costs to run')

        for i, (big, lab, sub, col) in enumerate(tiles):
            x = 0.16 + i * 0.28
            fig.text(x, 0.70, big, fontsize=44, color=col, ha='center',
                     va='center', fontweight='semibold')
            fig.text(x, 0.605, lab, fontsize=14.5, color=INK, ha='center', va='center')
            fig.text(x, 0.555, sub, fontsize=12, color=INK_3, ha='center', va='center')

        if stage >= 1:
            ax = fig.add_axes([0.16, 0.265, 0.30, 0.20])
            ax.set_facecolor(SURFACE)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(HAIRLINE)
            ax.grid(True, color=HAIRLINE, lw=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK_3, labelsize=10)
            ax.plot(range(len(batches)), mem_ratio, color=AQUA, lw=2.4, marker='o',
                    ms=6, mec=SURFACE, mew=1.4)
            ax.set_xticks(range(len(batches)))
            ax.set_xticklabels(batches)
            ax.set_xlabel('batch size', fontsize=11.5, color=INK_2)
            ax.set_ylabel('memory saved (×)', fontsize=11.5, color=INK_2)
            ax.set_title('the memory advantage widens with batch size',
                         fontsize=12, color=INK_2, pad=8)

            fig.text(0.53, 0.44, 'On GPU the latency advantage nearly vanishes\n'
                                 f'({gpu_lo:.1f}–{gpu_hi:.1f}×): the students are too small to\n'
                                 'saturate the device, and width-independent\n'
                                 'overhead sets the runtime.',
                     fontsize=13, color=INK_2, ha='left', va='top', linespacing=1.7)

        if stage == 0:
            caption(fig, ['v2 adds a training-time loss and changes nothing at inference — the '
                          'student is an ordinary HiVT.'])
        else:
            caption(fig, ['Measured on one RTX 5060; latency is hardware-dependent.'],
                    highlight='“55× smaller” is a memory claim, not a speed claim.')
        return fig

    fw.add(frame(0), 3.6)
    fw.add(frame(1), 5.4)
