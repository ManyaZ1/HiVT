"""Reliability (empirical vs nominal coverage) figure: v1 breaks calibration, v2 does not.

Two panels, both plotting the empirical coverage of the best mode's per-coordinate
Laplace central interval against its nominal level p:

  left  -- the 25%-data lambda triage. v1 (mean-target KD) bows progressively below
           the diagonal as lambda grows; v2 (distribution-matching KD) stays on it at
           every lambda. Same protocol for both families: 15-epoch triage checkpoints
           evaluated on the FULL val set by KD/eval_lambda_sweep.py.
  right -- the three full-data models of the thesis A/B table (no KD / v1 / v2) at
           lambda=0.5, so the plotted cov@p90 values are exactly the table's.

Colours: hue = KD variant (identity), shade + marker + linestyle = lambda, so lambda
is never encoded by colour alone. Palette checked for CVD separation (min pair dE
~11 OKLab x100 under deuteranopia, all normal-vision pairs >= 18).

Usage:
    python visualisation_other_tests/plot_reliability.py \
        --v1 docs/figures/lambda_sweep.json \
        --v2 docs/figures/lambda_sweep_v2.json \
        --full docs/figures/fulldata_reliability.json \
        --out docs/figures/reliability_v1_v2.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Nominal levels logged by metrics/calibration.py (val_cov_p10 ... val_cov_p90).
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

NOKD = "#1A1A1A"
V1_SHADES = {"0.25": "#FDAE6B", "0.5": "#E6550D", "1.0": "#7F2704"}
V2_SHADES = {"0.25": "#6BAED6", "0.5": "#2171B5", "1.0": "#08306B"}
LAMBDA_STYLE = {"0.25": (":", "v"), "0.5": ("--", "^"), "1.0": ("-", "D")}


def curve(entry):
    """Pull the 9-point coverage curve out of one eval-sweep record."""
    return [entry["cov_p%02d" % round(p * 100)] for p in LEVELS]


def diagonal(ax):
    ax.plot([0, 1], [0, 1], ls=(0, (6, 4)), color="#8C8C8C", lw=1.2, zorder=1,
            label="ideal (calibrated)")


def decorate(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("nominal coverage $p$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="v1 lambda-sweep JSON (has the shared lambda=0 point)")
    ap.add_argument("--v2", required=True, help="v2 lambda-sweep JSON")
    ap.add_argument("--full", required=True, help="full-data JSON with keys nokd/v1/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=("both", "full"), default="both",
                    help="both = two panels (thesis, full text width); full = the "
                         "full-data panel alone, sized for one IEEE column")
    args = ap.parse_args()

    with open(args.v1) as f:
        v1 = json.load(f)
    with open(args.v2) as f:
        v2 = json.load(f)
    with open(args.full) as f:
        full = json.load(f)

    single = args.layout == "full"
    if single:
        # One IEEE column is 3.5 in wide, so draw at 1:1 and keep type >= 8 pt.
        plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9,
                             "axes.labelsize": 8.5, "xtick.labelsize": 8,
                             "ytick.labelsize": 8})
        fig, axR = plt.subplots(1, 1, figsize=(3.5, 3.5))
        axL = None
    else:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.2))

    def baseline(ax, ys, label):
        """no-KD drawn as a wide reference band: the v2 curves land on top of it."""
        ax.plot(LEVELS, ys, color=NOKD, lw=5.0, alpha=0.22, solid_capstyle="round",
                zorder=2, label=label)

    # ---- left: the lambda dose-response, v1 vs v2 -------------------------------
    if axL is not None:
        diagonal(axL)
        baseline(axL, curve(v1["0.0"]), r"no KD ($\lambda=0$, reference)")
        for shades, tag in ((V1_SHADES, "v1"), (V2_SHADES, "v2")):
            for lam in ("0.25", "0.5", "1.0"):
                ls, mk = LAMBDA_STYLE[lam]
                axL.plot(LEVELS, curve((v1 if tag == "v1" else v2)[lam]), color=shades[lam],
                         lw=1.9, ls=ls, marker=mk, ms=6, mec="white", mew=0.6, zorder=3,
                         label=rf"{tag} $\lambda={lam}$")
        decorate(axL, "(a)  $\\lambda$-sweep (25% data): v1 drifts off, v2 does not")
        axL.set_ylabel("empirical coverage")
        axL.legend(fontsize=8.5, loc="upper left", frameon=True, framealpha=0.9,
                   borderpad=0.6, labelspacing=0.35)
        axL.text(0.97, 0.11, "below the diagonal =\nintervals too narrow\n(over-confident)",
                 fontsize=8.5, color="#5A5A5A", ha="right", va="bottom", linespacing=1.4)

    # ---- right: the full-data A/B, with cov@p90 labelled directly ---------------
    diagonal(axR)
    baseline(axR, curve(full["nokd"]), "no KD (reference)")
    series = [("v1", "v1 (mean-target)" if single else r"v1 (mean-target, $\lambda=0.5$)",
               V1_SHADES["0.5"], "--", "^"),
              ("v2", "v2 (dist-matching)" if single else r"v2 (dist-matching, $\lambda=0.5$)",
               V2_SHADES["0.5"], "-", "o")]
    for key, label, color, ls, mk in series:
        axR.plot(LEVELS, curve(full[key]), color=color, lw=2.0, ls=ls, marker=mk,
                 ms=5 if single else 6, mec="white", mew=0.6, zorder=3, label=label)
    if single:
        # Print figure: no in-figure title (the caption carries it) and one compact
        # cov@p90 block in the empty corner below the v1 curve.
        cov = {k: curve(full[k])[-1] for k in ("nokd", "v1", "v2")}
        axR.text(0.97, 0.04,
                 "cov@p90:  no KD %.3f\nv1 %.3f      v2 %.3f"
                 % (cov["nokd"], cov["v1"], cov["v2"]),
                 fontsize=7.5, color="#3D3D3D", ha="right", va="bottom",
                 linespacing=1.5)
        decorate(axR, "")
        axR.set_ylabel("empirical coverage")
    else:
        # Direct cov@p90 labels, stacked so they cannot collide (three series only).
        for key, tag, dy in (("v2", "v2", 12), ("nokd", "no KD", -13), ("v1", "v1", -12)):
            y = curve(full[key])[-1]
            axR.annotate(f"{tag}  {y:.3f}", xy=(0.9, y), xytext=(-8, dy),
                         textcoords="offset points", ha="right",
                         va="bottom" if dy > 0 else "top", fontsize=9, color="#3D3D3D")
        axR.text(0.97, 0.05, "labels: empirical coverage at $p=0.9$",
                 fontsize=9, color="#5A5A5A", ha="right", va="bottom")
        decorate(axR, "(b)  Full data, $\\lambda=0.5$: v2 repairs the calibration")
    axR.legend(fontsize=7.5 if single else 9, loc="upper left", frameon=True,
               framealpha=0.9, borderpad=0.5, labelspacing=0.3, handlelength=1.8)

    fig.tight_layout()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=300 if single else 200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
