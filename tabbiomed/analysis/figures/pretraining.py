"""Pretraining figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BLUE = "#376FA3"
GREY = "#858B94"
INK = "#283443"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "medium",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def comparison_legend(
    fig, full="Before exclusion", retained="After exclusion", title_axes=None
):
    layout = dict(loc="upper center", bbox_to_anchor=(0.54, 1), ncol=2)
    if title_axes is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            ax.title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            for ax in title_axes
        ]
        title_x = np.mean([(box.x0 + box.x1) / 2 for box in boxes])
        title_y = np.mean([(box.y0 + box.y1) / 2 for box in boxes])
        layout = dict(loc="center", bbox_to_anchor=(title_x, title_y), ncol=1)
    return fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markeredgecolor=GREY,
                markerfacecolor="white",
                markersize=7,
                label=full,
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markeredgecolor=BLUE,
                markerfacecolor=BLUE,
                markersize=6,
                label=retained,
            ),
        ],
        frameon=False,
        **layout,
    )


def paired_axis(ax, table, before, after, title, xlabel, limits=None, rank_axis=False):
    table = table.sort_values(["full_rank_position"], kind="stable")
    y = np.arange(len(table))
    a, b = (table[before].to_numpy(), table[after].to_numpy())
    ax.hlines(y, a, b, color="#B5BBC2", linewidth=1.35, zorder=1)
    ax.scatter(a, y, s=47, facecolor="white", edgecolor=GREY, linewidth=1.1, zorder=2)
    ax.scatter(b, y, s=23, facecolor=BLUE, edgecolor=BLUE, linewidth=0.6, zorder=3)
    ax.set_yticks(y, table.index)
    ax.set_ylim(len(table) - 0.5, -0.65)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.grid(axis="x", color="#E6E8EC", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if limits is not None:
        ax.set_xlim(*limits)
    if rank_axis:
        ax.set_xlim(0.5, len(table) + 0.6)
        ax.set_xticks(
            sorted({tick for tick in (1, 5, 10, 15, len(table)) if tick <= len(table)})
        )


def main_rank_shifts(tables, summary):
    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 17,
            "legend.fontsize": 13,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 8.5))
        count = f"{summary['n_datasets']} → {summary['n_retained_datasets']} datasets"
        for ax, key, title in zip(
            axes,
            ["rf_elo", "mean_rank"],
            ["RF-imputed Elo leaderboard", "Mean-rank leaderboard"],
        ):
            paired_axis(
                ax,
                tables[key],
                "full_rank_position",
                "excluded_rank_position",
                f"{title}\n{count}",
                "Leaderboard position (1 = best)",
                rank_axis=True,
            )
        fig.subplots_adjust(left=0.19, right=0.99, bottom=0.11, top=0.89, wspace=0.66)
        comparison_legend(fig, title_axes=axes)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_OUTPUT_ROOT / "pretraining")
    parser.add_argument(
        "--output", type=Path, default=_OUTPUT_ROOT / "pretraining_figures"
    )
    args = parser.parse_args()
    summary = json.loads((args.input / "summary.json").read_text())
    tables = {
        "rf_elo": pd.read_csv(
            args.input / "rf_imputed/elo_models_comparison.csv"
        ).set_index("model"),
        "mean_rank": pd.read_csv(
            args.input / "all_datasets_mean_rank_comparison.csv"
        ).set_index("model"),
    }
    if any(
        (
            len(frame) != summary["n_models"] or frame.index.has_duplicates
            for frame in tables.values()
        )
    ):
        raise ValueError("Invalid leaderboard comparison")
    configure_style()
    figure = main_rank_shifts(tables, summary)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output / "02_main_rank_shifts.pdf", bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
