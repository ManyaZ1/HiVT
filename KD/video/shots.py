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
            fig.text(0.5, 0.40, 'Distilling a mixture-density forecaster is a structurally '
                                'different problem\nfrom distilling a classifier — and the '
                                'obvious objective is quietly wrong.',
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
        title(fig, 'Small models are the point — and they cost accuracy')
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
        caption(fig, 'On-vehicle forecasting wants the smallest model that still works.',
                highlight=None if not annotate else
                'Accuracy falls faster than linearly: the last halving costs 3.5× more.')
        return fig

    for n in (1, 2, 3, 4):
        fw.add(frontier(n), 1.1)
    fw.add(frontier(4, annotate=True), 5.6)

    # --- 2b: classifier vs mixture (10 s) ----------------------------------
    def contrast(step):
        fig = new_frame()
        title(fig, 'Distillation recovers that gap — but its playbook is a classifier’s')

        axl = fig.add_axes([0.075, 0.32, 0.36, 0.44])
        bare(axl, keep_frame=False)
        axl.set_title('a classifier’s output', fontsize=13.5, color=INK, pad=12)
        probs = [0.62, 0.18, 0.09, 0.06, 0.03, 0.02]
        names = ['car', 'truck', 'bus', 'bike', 'ped', 'other']
        for i, (p, nm) in enumerate(zip(probs, names)):
            axl.barh(5 - i, p, height=0.62, color=BLUE, alpha=0.85, zorder=3)
            axl.text(-0.02, 5 - i, nm, ha='right', va='center', fontsize=11, color=INK_2)
        axl.set_xlim(0, 0.78)
        axl.set_ylim(-0.7, 5.7)

        axr = fig.add_axes([0.545, 0.32, 0.38, 0.44])
        bare(axr)
        rng = np.random.default_rng(7)
        t = np.linspace(0, 1, 30)
        for k, ang in enumerate(np.linspace(-0.85, 0.85, 6)):
            x = 26 * t * np.sin(ang * t)
            y = 26 * t * np.cos(ang * t)
            axr.plot(x, y, color=BLUE, lw=2.0, alpha=0.9, solid_capstyle='round')
            axr.fill_betweenx(y, x - 1.1 - 2.6 * t, x + 1.1 + 2.6 * t,
                              color=BLUE, alpha=0.10, lw=0)
            axr.plot(x[-1], y[-1], 'o', ms=12, color=BLUE, mec=SURFACE, mew=1.6)
            axr.text(x[-1], y[-1], str(k), fontsize=10.5, color='white',
                     ha='center', va='center', fontweight='bold')
        axr.set_aspect('equal')
        axr.set_title('a forecaster’s output: a mixture of 6 trajectories',
                      fontsize=13.5, color=INK, pad=12)

        if step == 0:
            caption(fig, ['Knowledge distillation is free at inference: a training-time loss only,',
                          'with the teacher’s predictions cached offline.'])
        else:
            caption(fig, ['For a classifier, output slot k means “class k” in every model ever trained.',
                          'A forecaster’s slots are trained winner-takes-all.'],
                    highlight='Slot k means nothing.')
        return fig

    fw.add(contrast(0), 4.6)
    fw.add(contrast(1), 5.4)


# --------------------------------------------------------------------------- #
# 3. BEAT 1 — the mode-permutation problem  (34 s)
# --------------------------------------------------------------------------- #
def shot03_permutation(fw, tr):
    sc, modal = pick_perm_scene(tr)
    tloc, sloc = tr['t_loc'][sc], tr['s_loc'][sc]
    ci, co = float(tr['cost_identity'][sc]), float(tr['cost_optimal'][sc])

    # --- 3a: winner-takes-all makes the index meaningless (10 s) -----------
    gt_all = tr['gt']
    win = np.linalg.norm(tr['s_loc'][:, :, -1] - gt_all[:, None, -1], axis=-1).argmin(1)
    # three scenes whose winning slot differs, so the shot shows the index moving
    picks, seen = [], set()
    for i in range(len(win)):
        if win[i] not in seen and np.isfinite(gt_all[i]).all():
            seen.add(int(win[i]))
            picks.append(i)
        if len(picks) == 3:
            break

    def wta(i, reveal):
        fig = new_frame()
        title(fig, 'HiVT is trained winner-takes-all')
        s, g = tr['s_loc'][i], gt_all[i]
        ax = _modes_axes(fig, [0.075, 0.255, 0.85, 0.55], g[None], s)
        w = int(win[i])
        for k in range(6):
            hot = reveal and k == w
            _draw_modes(ax, s[k:k + 1], BLUE if hot else INK_3,
                        lw=2.6 if hot else 1.4, alpha=1.0 if hot else 0.45,
                        zorder=6 if hot else 3)
            ax.texts[-1].set_text(str(k))
        if reveal:
            p = np.vstack([[0, 0], g])
            ax.plot(p[:, 0], p[:, 1], color=INK, lw=2.6, ls=(0, (5, 3)), zorder=8)
            ax.plot(*g[-1], '*', ms=20, color=INK, zorder=9)
        legend_chips(fig, [(INK, 'ground truth'), (BLUE, 'the mode that wins'),
                           (INK_3, 'the five that get no gradient')], y=0.20)
        if not reveal:
            caption(fig, 'Six modes. Only one of them will be trained on this scene.')
        else:
            caption(fig, [f'Here slot {w} happened to be closest, so slot {w} got the gradient '
                          f'and the other five got nothing.',
                          'Nothing ever tells slot 3 to be “the left turn” — the ordering is an '
                          'accident of initialisation.'])
        return fig

    fw.add(wta(picks[0], False), 2.0)
    for j, i in enumerate(picks):
        fw.add(wta(i, True), 2.6 if j == 0 else 2.7)

    # --- 3b: the hero pairing shot (12 s) ----------------------------------
    C = tr['cost'][sc]

    def pairing(stage):
        fig = new_frame()
        title(fig, 'Index-aligned distillation matches teacher mode k to student mode k')

        # left: the real endpoints, cropped to the fan so the twelve modes separate
        ax = _modes_axes(fig, [0.045, 0.30, 0.44, 0.44], tloc, sloc, focus='ends')
        _draw_modes(ax, tloc, ORANGE, lw=1.5, alpha=0.85, fontsize=10)
        _draw_modes(ax, sloc, BLUE, lw=1.5, alpha=0.85, fontsize=10)
        ax.set_title('where the six modes actually end up', fontsize=12.5,
                     color=INK_2, pad=10)

        # right: the cost matrix the pairing is chosen in
        axm = fig.add_axes([0.585, 0.285, 0.30, 0.47])
        axm.set_facecolor(SURFACE)
        # NOT reversed: dark = large distance = a bad pairing. Keeping the same
        # light->dark = more convention as the aggregate matrix in 3c means the
        # argument reads off the colour alone -- the identity diagonal lands on
        # dark cells, the optimal assignment on pale ones.
        axm.imshow(C, cmap=CMAP_BLUE, zorder=1)
        axm.set_xticks(range(6))
        axm.set_yticks(range(6))
        axm.tick_params(colors=INK_2, labelsize=10.5, length=0)
        axm.set_xlabel('teacher mode', fontsize=12, color=INK_2, labelpad=6)
        axm.set_ylabel('student mode', fontsize=12, color=INK_2, labelpad=6)
        axm.set_title('endpoint distance between every pair  (m)',
                      fontsize=12.5, color=INK_2, pad=10)
        for s in axm.spines.values():
            s.set_color(HAIRLINE)
        for (r, c_) in np.ndindex(C.shape):
            axm.text(c_, r, f'{C[r, c_]:.0f}', ha='center', va='center',
                     fontsize=9.5, zorder=3,
                     color='white' if C[r, c_] > 0.55 * C.max() else INK_2)
        if stage >= 1:
            for k in range(6):
                axm.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, fill=False,
                                            ec=INK, lw=2.2, zorder=4))
        if stage >= 2:
            for k, f in enumerate(modal):
                axm.add_patch(plt.Rectangle((int(f) - .5, k - .5), 1, 1, fill=False,
                                            ec=AQUA, lw=2.8, zorder=5))

        chips = [(ORANGE, 'teacher modes 0–5'), (BLUE, 'student modes 0–5')]
        if stage == 1:
            chips.append((INK, 'pairing by index'))
        if stage >= 2:
            chips.append((AQUA, 'optimal pairing'))
        legend_chips(fig, chips, y=0.205)

        if stage == 0:
            caption(fig, ['Both models emit six modes for the same scene, and both were '
                          'trained winner-takes-all,',
                          'so neither ever had a reason to order them the same way.'])
        elif stage == 1:
            caption(fig, ['Pairing by index reads straight down the diagonal — and picks '
                          'some of the worst cells in the matrix.'],
                    highlight=f'Mean pairing distance: {ci:.2f} m')
        else:
            caption(fig, [f'The optimal assignment takes a different permutation entirely: '
                          f'{co:.2f} m, {ci / co:.2f}× closer.'],
                    highlight='Index-aligned KD is comparing unrelated trajectories.')
        return fig

    fw.add(pairing(0), 2.6)
    fw.add(pairing(1), 4.2)
    fw.add(pairing(2), 5.2)

    # --- 3c: the matrix accumulating over 500 scenes (12 s) ----------------
    perm = tr['perm']
    n_total = len(perm)
    steps = list(range(20, n_total + 1, 20))

    def matrix(n, final=False):
        fig = new_frame()
        title(fig, 'Aggregated over the validation set')
        M = np.zeros((6, 6), dtype=int)
        for p in perm[:n]:
            M[np.arange(6), p] += 1

        ax = fig.add_axes([0.335, 0.265, 0.33, 0.565])
        ax.set_facecolor(SURFACE)
        ax.imshow(M, cmap=CMAP_BLUE, vmin=0, vmax=max(M.max(), 1), zorder=2)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.tick_params(colors=INK_2, labelsize=11, length=0)
        ax.set_xlabel('teacher mode index', fontsize=12.5, color=INK_2, labelpad=8)
        ax.set_ylabel('student mode index', fontsize=12.5, color=INK_2, labelpad=8)
        for s in ax.spines.values():
            s.set_color(HAIRLINE)
        for (r, c), v in np.ndenumerate(M):
            if v:
                ax.text(c, r, str(v), ha='center', va='center', fontsize=10.5,
                        color='white' if v > M.max() * 0.55 else INK_2, zorder=4)
        # the identity diagonal, outlined so the viewer can watch mass NOT land on it
        for k in range(6):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                       ec=ORANGE, lw=2.0, ls=(0, (4, 2)), zorder=5))
        ax.text(0.5, 1.045, 'dashed = the identity pairing', transform=ax.transAxes,
                ha='center', fontsize=11.5, color=ORANGE)

        fig.text(0.735, 0.63, f'scenes\n{n} / {n_total}', fontsize=17, color=INK_2,
                 ha='left', va='center', linespacing=1.5)
        fig.text(0.735, 0.45, 'identity was\noptimal in', fontsize=13.5, color=INK_2,
                 ha='left', va='center', linespacing=1.5)
        fig.text(0.735, 0.355, f'{int(np.array([tuple(p) == tuple(range(6)) for p in perm[:n]]).sum())}'
                               '  scenes', fontsize=22, color=INK, ha='left',
                 va='center', fontweight='semibold')
        if not final:
            caption(fig, 'Cell (i, j) counts the scenes whose optimal assignment maps '
                         'teacher mode j to student mode i.')
        else:
            caption(fig, ['The mass lands on a consistent non-identity permutation — the same one '
                          'in 84.4% of scenes.'],
                    highlight='The identity pairing is optimal in 0% of 500 validation scenes.')
        return fig

    per = 5.4 / len(steps)
    for n in steps:
        fw.add(matrix(n), per)
    fw.add(matrix(n_total, final=True), 6.0)


# --------------------------------------------------------------------------- #
# 4. The fix  (18 s)
# --------------------------------------------------------------------------- #
def shot04_objective(fw, tr):
    sc, modal = pick_perm_scene(tr)
    tloc, sloc = tr['t_loc'][sc], tr['s_loc'][sc]

    # --- 4a: why not Hungarian (8 s) ---------------------------------------
    alt = modal.copy()
    alt[[0, 3]] = alt[[3, 0]]          # a plausible neighbouring assignment

    def hungarian(which, show_caption=True):
        fig = new_frame()
        title(fig, 'Why not just solve the assignment problem?')
        ax = _modes_axes(fig, [0.20, 0.235, 0.60, 0.60], tloc, sloc, focus='ends')
        _draw_modes(ax, tloc, ORANGE, lw=1.5, alpha=0.85)
        _draw_modes(ax, sloc, BLUE, lw=1.5, alpha=0.85)
        _pair_lines(ax, tloc, sloc, modal if which == 0 else alt, INK, lw=2.4, alpha=0.9)
        fig.text(0.72, 0.30, f'training step  {2100 + which}', fontsize=13,
                 color=INK_3, ha='left')
        if show_caption:
            caption(fig, ['Hungarian matching re-solves a hard, non-differentiable assignment '
                          'every step.',
                          'It is undefined when the two models have different mode counts,'],
                    highlight='and the pairing flips discontinuously between steps.')
        return fig

    for i in range(6):                  # visible flicker between two assignments
        fw.add(hungarian(i % 2), 0.55)
    fw.add(hungarian(0), 4.7)

    # --- 4b: score against the whole mixture (10 s) ------------------------
    def mixture(stage):
        fig = new_frame()
        title(fig, 'Our objective never forms a pairing at all')
        ax = _modes_axes(fig, [0.045, 0.235, 0.44, 0.60], tloc, sloc, focus='ends')
        _draw_modes(ax, sloc, BLUE, lw=1.5, alpha=0.75)
        f = int(np.argmax(tr['t_probs'][sc]))
        _draw_modes(ax, tloc[f:f + 1], ORANGE, lw=2.6, alpha=1.0)
        ax.texts[-1].set_text(str(f))
        if stage >= 1:
            # soft responsibilities: every student mode is linked, weighted by
            # how well it explains this teacher mode. No winner is chosen.
            d = np.linalg.norm(sloc[:, -1] - tloc[f, -1], axis=-1)
            w = np.exp(-d / max(d.mean(), 1e-6))
            w /= w.max()
            for k in range(6):
                a, b = sloc[k, -1], tloc[f, -1]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=AQUA,
                        lw=0.8 + 4.2 * w[k], alpha=0.30 + 0.6 * w[k], zorder=2,
                        solid_capstyle='round')

        fig.text(0.55, 0.66, 'score each teacher mode under the\nstudent’s '
                             'ENTIRE mixture density',
                 fontsize=15.5, color=INK, ha='left', va='center', linespacing=1.6)
        fig.text(0.55, 0.47,
                 r'$\mathcal{L} = -\sum_f \pi^T_f \, \log \sum_k \pi^S_k \, '
                 r'\mathrm{Lap}\left(\mu^T_f \mid \mu^S_k,\, b^S_k\right)$',
                 fontsize=19, color=INK, ha='left', va='center')
        if stage >= 1:
            fig.text(0.55, 0.345, 'the log-sum-exp is a smooth maximum: teacher mode f is\n'
                                  'explained by whichever student modes lie near it —\n'
                                  'selected by value, never by index.',
                     fontsize=12.5, color=INK_2, ha='left', va='center', linespacing=1.7)
        caption(fig, ['The student enters only through its mixture density, so reordering '
                      'either model’s modes changes nothing.'],
                highlight='Permutation-invariant by construction — and defined for any mode counts.')
        return fig

    fw.add(mixture(0), 3.8)
    fw.add(mixture(1), 6.2)


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

    fw.add(lead(), 2.4)

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
        fw.add(fig, 1.35)

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
    fw.add(fig, 2.0)


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
        (BLUE, 'Mode index is meaningless.',
         'Winner-takes-all training leaves the ordering arbitrary — the identity\n'
         'pairing is optimal in 0% of scenes, so index-aligned KD is ill-posed.'),
        (ORANGE, 'Match the distribution, not the means.',
         'Scoring the teacher’s mean trajectories improves every geometric metric\n'
         'while quietly destroying calibration. Best-of-K cannot see it.'),
        (AQUA, 'Score teacher modes under the student’s whole mixture.',
         'No matching, any number of modes, permutation-invariant by construction —\n'
         'and calibration survives compression for free.'),
    ]

    def frame(n):
        fig = new_frame()
        title(fig, 'Three things to take away', y=0.90, size=22)
        y = 0.68
        for i in range(n):
            color, head, body = lines[i]
            fig.patches.append(plt.Rectangle((0.055, y - 0.075), 0.006, 0.115,
                                             transform=fig.transFigure,
                                             facecolor=color, edgecolor='none'))
            fig.text(0.085, y + 0.018, head, fontsize=17.5, color=INK,
                     fontweight='semibold', ha='left', va='center')
            fig.text(0.085, y - 0.045, body, fontsize=13, color=INK_2,
                     ha='left', va='center', linespacing=1.6)
            y -= 0.185
        if n == 3:
            caption(fig, [], highlight='If your forecaster emits a WTA mixture, index-aligned KD '
                                       'is wrong for it too.')
        return fig

    fw.add(frame(1), 3.6)
    fw.add(frame(2), 3.8)
    fw.add(frame(3), 6.6)


def shot10_reprise(fw):
    fig = new_frame()
    fig.text(0.5, 0.56, PAPER_TITLE, fontsize=27, color=INK, fontweight='semibold',
             ha='center', va='center', linespacing=1.45)
    fig.text(0.5, 0.36, 'Code and trained students released on acceptance.',
             fontsize=14, color=INK_3, ha='center', va='center')
    fw.add(fig, 3.5)
