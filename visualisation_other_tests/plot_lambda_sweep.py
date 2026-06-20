"""Multi-panel KD lambda dose-response figure.

Plots, against lambda_kl on the x-axis, the metrics that together prove the
"mean-target KD = scale shrinkage" mechanism at every distillation strength:
geometry improves (minFDE down), full-distribution calibration degrades
(mixNLL up, calib_err up) via sharpening (b_scale down), while mode-weight
entropy stays flat (no mode collapse).

Reads the numbers from a JSON produced by the eval sweep (see
docs/kd_emb32_kl_comparison.md). Usage:
    python visualisation_other_tests/plot_lambda_sweep.py \
        --data /tmp/lambda_sweep.json --out docs/figures/kd_lambda_sweep.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (key, title, y-label, expected direction annotation)
PANELS = [
    ("minFDE", "Geometry: minFDE", "minFDE (m)", "↓ improves"),
    ("mixNLL", "Calibration: mixNLL", "mixNLL", "↑ degrades"),
    ("b_scale", "Sharpness: mean Laplace b", "b_scale (m)", "↓ sharper"),
    ("calib_err", "Reliability: calibration error", "calib_err", "↑ degrades"),
    ("pi_entropy", "Diversity: pi entropy", "pi_entropy (nats)", "≈ flat"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSON: {lambda: {metric: value}}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    lambdas = sorted(float(k) for k in data)
    rows = [data[str(l)] if str(l) in data else data[("%g" % l)] for l in lambdas]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 4))
    for ax, (key, title, ylab, ann) in zip(axes, PANELS):
        ys = [r[key] for r in rows]
        ax.plot(lambdas, ys, "o-", color="tab:blue", lw=2, ms=7)
        ax.set_title(f"{title}\n({ann})", fontsize=11)
        ax.set_xlabel(r"$\lambda_{KL}$")
        ax.set_ylabel(ylab)
        ax.set_xticks(lambdas)
        ax.grid(alpha=0.3)
        # Annotate each point with its value.
        for x, y in zip(lambdas, ys):
            ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
    fig.suptitle(
        "KD lambda dose-response (emb32 triage: 25% data, 15 ep, lr=3e-3) — "
        "mechanism = scale shrinkage at all strengths", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()