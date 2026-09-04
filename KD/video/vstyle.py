"""
vstyle.py — shared visual language for the ICRA submission video.

Every shot is a 1280x720 matplotlib figure on a flat light surface, with a
fixed caption band across the bottom. The video is assumed to be watched
MUTED, so no fact may be carried by audio: whatever the shot argues has to be
readable in that band.

Colour is load-bearing and constant across the whole video:

    blue   #2a78d6   from-scratch student / the model you would ship today
    orange #eb6834   v1, mean-target KD  (and the teacher-as-a-point-target)
    aqua   #1baf7a   v2, distribution matching  (ours)

These are slots 1-3 of the data-viz reference palette, used unmodified, on that
palette's own light surface (#fcfcfb). The reference records this triple as
clearing all-pairs CVD and normal-vision separation in both modes, so it is not
re-stepped here. Two consequences are load-bearing and must be preserved by any
shot added later:

  * aqua sits below 3:1 contrast on the light surface, so the "relief rule"
    applies -- v2 always carries a visible direct label and is never
    identified by colour alone.
  * the assignment-matrix heatmap uses the palette's SEQUENTIAL blue ramp
    (one hue, light->dark), not a rainbow map: magnitude is a magnitude.

Frames are emitted as PNGs plus an ffmpeg concat manifest with per-frame
durations, so a 4-second hold costs one file rather than 100. See build_video.py.
"""

import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
SURFACE = '#fcfcfb'

# Categorical slots 1-3, unmodified. Keys match the montage renderer's arm keys.
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

ARM_COLOR = {'noKD': BLUE, 'v1': ORANGE, 'v2': AQUA, 'teacher': '#6b6b66'}
ARM_LABEL = {
    'noKD': 'from-scratch HiVT-32',
    'v1': 'v1  mean-target KD',
    'v2': 'v2  distribution matching',
    'teacher': 'HiVT-128 teacher',
}

# Ink. Text never wears a series colour; a coloured mark beside it carries identity.
INK = '#1a1a19'        # primary
INK_2 = '#55554f'      # secondary
INK_3 = '#8a8a82'      # muted / recessive grid, axes, de-emphasised marks
HAIRLINE = '#dedbd4'

# Sequential blue ramp, palette steps 100..700 (light -> dark).
SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
            '#0d366b']
CMAP_BLUE = LinearSegmentedColormap.from_list('seq_blue', [SURFACE] + SEQ_BLUE)


# --------------------------------------------------------------------------- #
# Canvas
# --------------------------------------------------------------------------- #
W, H, DPI = 1280, 720, 100

# The caption band occupies the bottom of every frame at a fixed height, so the
# viewer's eye never has to hunt for it between shots.
BAND_H = 0.155          # fraction of figure height
BAND_TOP = BAND_H       # content axes must stay above this


def new_frame():
    """A blank 1280x720 figure on the surface colour."""
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=SURFACE)
    return fig


def bare(ax, keep_frame=False):
    """Strip an axes to its data: no ticks, no spines unless asked."""
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(keep_frame)
        if keep_frame:
            s.set_color(HAIRLINE)
            s.set_linewidth(1.0)
    return ax


def title(fig, text, y=0.945, size=19, color=None, weight='semibold'):
    """Shot title, top-left-anchored to a consistent x across all shots."""
    return fig.text(0.055, y, text, fontsize=size, color=color or INK,
                    fontweight=weight, ha='left', va='center')


def subtitle(fig, text, y=0.885, size=13, color=None):
    return fig.text(0.055, y, text, fontsize=size, color=color or INK_2,
                    ha='left', va='center')


def caption(fig, lines, highlight=None, size=15):
    """The burned-in caption band.

    lines: str or list of str, rendered as consecutive lines in the band.
    highlight: optional str rendered in a heavier weight beneath them -- used
        for the sentence the shot exists to land.
    """
    if isinstance(lines, str):
        lines = [lines]

    # Wrap to the printable width. Captions carry the whole argument for a muted
    # viewer, so a line running off the right edge silently deletes the point of
    # the shot -- wrap rather than trust the author to count characters.
    wrapped = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, width=112) or [''])

    # A hairline rule separates the band from the content without boxing it in.
    fig.add_artist(plt.Line2D([0.055, 0.945], [BAND_TOP, BAND_TOP],
                              color=HAIRLINE, linewidth=1.0,
                              transform=fig.transFigure))

    y = BAND_TOP - 0.042
    for ln in wrapped:
        fig.text(0.055, y, ln, fontsize=size, color=INK_2, ha='left', va='center')
        y -= 0.040
    if highlight:
        for i, ln in enumerate(textwrap.wrap(highlight, width=78)):
            fig.text(0.055, y - 0.004 - i * 0.044, ln, fontsize=size + 2.5,
                     color=INK, fontweight='semibold', ha='left', va='center')
    return fig


def legend_chips(fig, entries, y=0.055, x0=0.055, size=12, gap=0.185):
    """Inline colour chips + labels. Identity is never colour-alone, so every
    chip is directly labelled -- this is also what satisfies the relief rule
    for aqua on the light surface."""
    for i, (color, label) in enumerate(entries):
        x = x0 + i * gap
        fig.patches.append(plt.Rectangle(
            (x, y - 0.008), 0.018, 0.016, transform=fig.transFigure,
            facecolor=color, edgecolor='none', zorder=5))
        fig.text(x + 0.026, y, label, fontsize=size, color=INK_2,
                 ha='left', va='center')


# --------------------------------------------------------------------------- #
# Frame emission
# --------------------------------------------------------------------------- #
class FrameWriter:
    """Writes numbered PNGs and an ffmpeg concat manifest with per-frame holds.

    Animated passages emit one frame per tick at FPS; a static beat emits ONE
    PNG that the manifest then references once per tick. H.264 spends almost
    nothing on a repeated frame, so a 6-second hold still costs one file.

    Durations are quantised to whole 1/FPS ticks and the manifest emits one
    fixed-length entry per tick, rather than one variable-length entry per
    frame. Variable `duration` directives are rounded up independently by the
    concat demuxer, which silently added 2.6 s across 95 frames and pushed a
    179.98 s timeline to 182.6 s on encode -- over a hard submission cap that
    nothing downstream would have caught.
    """

    FPS = 25

    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.n = 0
        self.entries = []       # (filename, duration_seconds)
        self.overflows = []     # (frame, text, bbox) for anything off-frame

    def add(self, fig, seconds):
        """Save `fig` and hold it for `seconds`. Closes the figure."""
        self._check_overflow(fig)
        name = f'f{self.n:05d}.png'
        fig.savefig(os.path.join(self.outdir, name), dpi=DPI,
                    facecolor=SURFACE, edgecolor='none')
        plt.close(fig)
        self.entries.append((name, max(1, int(round(seconds * self.FPS)))))
        self.n += 1
        return name

    def _check_overflow(self, fig):
        """Warn if any text runs outside the frame.

        A caption clipped at the right edge deletes the claim the shot exists to
        make, and on a muted video there is no second channel carrying it. This
        catches it at render time instead of on playback.
        """
        fig.canvas.draw()
        for t in fig.texts:
            bb = t.get_window_extent().transformed(fig.transFigure.inverted())
            if bb.x1 > 0.995 or bb.x0 < 0.005 or bb.y1 > 1.0 or bb.y0 < 0.0:
                self.overflows.append(
                    (self.n, repr(t.get_text()[:60]),
                     f'x=[{bb.x0:.3f},{bb.x1:.3f}] y=[{bb.y0:.3f},{bb.y1:.3f}]'))

    def add_copy(self, seconds):
        """Hold the previously written frame for a further `seconds` without
        re-rendering it (concat can list the same file twice)."""
        if not self.entries:
            raise RuntimeError('add_copy() before any frame was written')
        self.entries.append((self.entries[-1][0],
                             max(1, int(round(seconds * self.FPS)))))

    @property
    def ticks(self):
        return sum(t for _, t in self.entries)

    @property
    def duration(self):
        """Exact encoded duration: whole ticks at FPS, no rounding surprises."""
        return self.ticks / self.FPS

    def write_manifest(self, path=None):
        path = path or os.path.join(self.outdir, 'concat.txt')
        tick = 1.0 / self.FPS
        with open(path, 'w') as fh:
            for name, n in self.entries:
                for _ in range(n):
                    fh.write(f"file '{name}'\nduration {tick:.6f}\n")
            # The concat demuxer drops the final entry's duration unless the
            # last file is repeated without one.
            fh.write(f"file '{self.entries[-1][0]}'\n")
        return path
