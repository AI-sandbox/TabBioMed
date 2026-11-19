"""Calibration figures."""

from __future__ import annotations
from tabbiomed.analysis.downloads import ensure_results
import argparse
import json
from pathlib import Path
import numpy as np
import tabbiomed.analysis.calibration as cs
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve

OUT = cs.DEFAULT_OUT / "paper_figures"
SERIES = "#2a78d6"
REFERENCE = "#6f6e69"
HIST = "#8a8984"
GRID = "#e4e3df"
INK = "#0b0b0b"


def paper_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 21,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 21,
            "axes.linewidth": 1.0,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def reliability(dataset: str, model: str, n_bins: int, strategy: str, variant: str):
    loaded = cs.load_pair(cs.DEFAULT_RESULTS / model / dataset, variant)
    if loaded is None:
        raise FileNotFoundError(f"no predictions for {model} on {dataset}")
    y, proba, var, _, _ = loaded
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2 and proba.shape[1] > 2:
        m = cs.compute_metrics_multiclass(y, proba)
        p = proba.max(1)
        target = (proba.argmax(1) == np.asarray(y).astype(int)).astype(int)
        multiclass = True
    else:
        target = (y == 1).astype(int)
        p = cs._positive(proba)
        m = cs.compute_metrics(target, p)
        multiclass = False
    frac, mean_pred = calibration_curve(target, p, n_bins=n_bins, strategy=strategy)
    m.update(model=model, variant_used=var)
    return (mean_pred, frac, p, multiclass, m)


def style_axes(ax):
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=4, width=0.8)


HIST_FRACTION = 0.35
HIST_ALPHA = 0.28


def count_axis(ax, log_max: float):
    ax2 = ax.twinx()
    ax2.set_ylim(-0.02, 1.02)
    max_count = 10**log_max - 1
    ticks = [0] + [10**k for k in range(1, int(np.floor(np.log10(max_count))) + 1)]
    pos = [HIST_FRACTION * np.log10(1 + c) / log_max for c in ticks]
    ax2.set_yticks(pos)
    ax2.set_yticklabels([f"{c:,}" for c in ticks])
    ax2.spines["right"].set_bounds(0, HIST_FRACTION)
    ax2.spines["right"].set_color(REFERENCE)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)
    ax2.tick_params(
        axis="y", labelsize=13, length=3, width=0.8, colors=REFERENCE, pad=2
    )
    ax2.grid(False)


def make_figure(
    name: str,
    dataset: str,
    models: list[str],
    n_bins: int,
    strategy: str,
    hist: bool,
    variant: str,
    out: Path,
):
    ncols = min(2, len(models))
    nrows = (len(models) + ncols - 1) // ncols
    width, height = (11.0, 3.1 * nrows + 2.4)
    left = 0.135
    bottom_in, top_in = (0.99, 1.4)
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        nrows,
        ncols,
        left=left,
        right=1 - left,
        bottom=bottom_in / height,
        top=1 - top_in / height,
        wspace=0.3 if hist else 0.2,
        hspace=0.34,
    )
    panels = [
        reliability(dataset, model, n_bins, strategy, variant) for model in models
    ]
    edges = np.linspace(0, 1, (n_bins if strategy == "uniform" else 20) + 1)
    counts = [np.histogram(p, bins=edges)[0] for _, _, p, _, _ in panels]
    log_max = np.log10(1 + max((c.max() for c in counts)))
    multiclass = panels[0][3]
    for i, ((mean_pred, frac, p, _, m), cnt) in enumerate(zip(panels, counts)):
        ax = fig.add_subplot(grid[i // ncols, i % ncols])
        if hist:
            heights = HIST_FRACTION * np.log10(1 + cnt) / log_max
            ax.bar(
                edges[:-1],
                heights,
                width=np.diff(edges),
                align="edge",
                bottom=0,
                color=HIST,
                alpha=HIST_ALPHA,
                edgecolor="white",
                linewidth=1.0,
                zorder=1,
            )
            count_axis(ax, log_max)
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle=(0, (5, 4)),
            color=REFERENCE,
            linewidth=1.6,
            zorder=2,
        )
        ax.plot(
            mean_pred,
            frac,
            color=SERIES,
            linewidth=2.2,
            solid_joinstyle="round",
            solid_capstyle="round",
            marker="o",
            markersize=7,
            markerfacecolor=SERIES,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
        style_axes(ax)
        ax.set_title(m["model"], pad=10)
        if i % ncols:
            ax.tick_params(labelleft=False)
        if i + ncols < len(models):
            ax.tick_params(labelbottom=False)
    if multiclass:
        xlabel, ylabel = ("Confidence (top-label predicted probability)", "Accuracy")
    else:
        xlabel, ylabel = ("Mean predicted probability", "Fraction of positives")
    centre = 0.5
    fig.text(
        centre,
        0.03,
        xlabel,
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    fig.text(
        0.045,
        (grid.top + grid.bottom) / 2,
        ylabel,
        rotation=90,
        ha="left",
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    handles = [
        Line2D(
            [],
            [],
            color=SERIES,
            linewidth=2.2,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="Model",
        ),
        Line2D(
            [],
            [],
            color=REFERENCE,
            linewidth=1.6,
            linestyle=(0, (5, 4)),
            label="Perfect calibration",
        ),
    ]
    if hist:
        handles.append(
            Patch(
                facecolor=HIST,
                alpha=HIST_ALPHA,
                edgecolor="none",
                label="Test samples per bin (right axis, log)",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(centre, 0.995),
        ncol=len(handles),
        frameon=False,
        handlelength=2.2 if hist else 2.6,
        columnspacing=1.6 if hist else 2.2,
        fontsize=18 if hist else None,
    )
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"calibration_{name}_{('hist' if hist else 'nohist')}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    keep = [
        "model",
        "variant_used",
        "n_test",
        "roc_auc_pooled",
        "accuracy_pooled",
        "ece_uniform",
        "ici",
        "cal_slope",
        "cal_intercept",
        "bias",
    ]
    summary = [{k: m[k] for k in keep if k in m} for *_, m in panels]
    stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "variant": variant,
                "n_bins": n_bins,
                "strategy": strategy,
                "histogram": hist,
                "histogram_bins": len(edges) - 1,
                "histogram_counts": {
                    m["model"]: c.tolist() for (*_, m), c in zip(panels, counts)
                },
                "panels": summary,
            },
            indent=2,
        )
    )
    print(f"wrote {stem.with_suffix('.pdf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sexcheck_chrX_50k")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["TabPFNv2.5", "RealTabPFNv2.5", "RealMLP", "ElasticNet"],
    )
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--strategy", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument("--name", help="Output filename stem, defaults to the dataset name")
    ap.add_argument("--variant", default="best", choices=["best", "default"])
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    if args.n_bins < 1:
        ap.error("--n-bins must be positive")
    ensure_results()
    paper_style()
    for hist in (False, True):
        make_figure(
            args.name or args.dataset,
            args.dataset,
            args.models,
            args.n_bins,
            args.strategy,
            hist,
            args.variant,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
