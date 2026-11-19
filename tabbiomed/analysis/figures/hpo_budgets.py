"""Hpo budgets figures."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

CAPS = (25, 50, 100, 200)
BLUE, ORANGE = ("#376FA3", "#D68936")


def binary_gaps(frame):
    rows = []
    for model, group in frame.loc[frame.task.eq("binary")].groupby("model"):
        val = group.pivot(index="dataset", columns="cap", values="validation").reindex(
            columns=CAPS
        )
        test = group.pivot(index="dataset", columns="cap", values="test").reindex(
            columns=CAPS
        )
        complete = val.notna().all(axis=1) & test.notna().all(axis=1)
        for cap in CAPS:
            rows.append(
                dict(
                    model=model,
                    cap=cap,
                    n=int(complete.sum()),
                    gap_pp=100
                    * (val.loc[complete, cap] - test.loc[complete, cap]).mean(),
                )
            )
    return pd.DataFrame(rows)


def draw_curves(curves, gaps, output):
    curves = curves.loc[~curves.model.eq("iLTM")]
    gaps = gaps.loc[~gaps.model.eq("iLTM")]
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    models = sorted(curves.model.unique())
    ncols = 3
    nrows = max(1, (len(models) + ncols - 1) // ncols)
    for kind in ("budget", "gap"):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(12.6, max(3.6, 1.8 * nrows)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        values = (
            curves[["validation_net_win", "test_net_win"]].to_numpy()
            if kind == "budget"
            else gaps.gap_pp.to_numpy()
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            lo = min(0, np.floor(finite.min() / 5) * 5) - 5
            hi = (
                100 if kind == "budget" else max(0, np.ceil(finite.max() / 5) * 5)
            ) + 5
        else:
            lo, hi = (-5, 105 if kind == "budget" else 5)
        for ax, model in zip(axes.ravel(), models):
            data = (
                (curves if kind == "budget" else gaps)
                .loc[lambda x: x.model.eq(model)]
                .sort_values("cap")
            )
            n = int(data.n.iloc[0]) if len(data) else 0
            if n:
                series = (
                    [
                        ("validation_net_win", BLUE, "Validation"),
                        ("test_net_win", ORANGE, "Test"),
                    ]
                    if kind == "budget"
                    else [("gap_pp", BLUE, "Validation - test")]
                )
                for column, color, label in series:
                    ax.plot(
                        data.cap,
                        data[column],
                        color=color,
                        linewidth=1.8,
                        marker="o",
                        markersize=11,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        label=label,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No complete comparison",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.axhline(0, color="#999999", linewidth=0.7, zorder=0)
            ax.set_title(f"{model}  (n = {n})", loc="left", pad=5)
            ax.set_xscale("log", base=2)
            ax.set_xticks(CAPS, [str(cap) for cap in CAPS])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlim(22, 225)
            ax.set_ylim(lo, hi)
            ax.grid(axis="y", alpha=0.2, linewidth=0.6)
        for ax in axes.ravel()[len(models) :]:
            ax.set_axis_off()
        if kind == "budget":
            handles_labels = next(
                (
                    ax.get_legend_handles_labels()
                    for ax in axes.ravel()
                    if ax.get_legend_handles_labels()[0]
                ),
                ([], []),
            )
            if handles_labels[0]:
                fig.legend(
                    *handles_labels,
                    loc="upper center",
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=(0.52, 1.0),
                    fontsize=16,
                )
        fig.supxlabel("Attempted-configuration cap", fontsize=17, y=0.012)
        fig.supylabel(
            "Net improvement relative to cap 25 (%)"
            if kind == "budget"
            else "Validation - test ROC-AUC (percentage points)",
            fontsize=17,
            x=0.006,
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.99,
            bottom=0.055,
            top=0.95 if kind == "budget" else 0.975,
            hspace=0.59,
            wspace=0.15,
        )
        stem = (
            "hpo_budget_comparison" if kind == "budget" else "hpo_validation_test_gaps"
        )
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)


def counts_and_rates(searches, pairs):
    by_dataset = searches.groupby(["model", "dataset"])[["attempted", "scored"]].agg(
        lambda x: x.mean() if x.notna().all() else np.nan
    )
    rows = []
    for model, group in by_dataset.groupby(level="model"):
        row = dict(model=model, n_datasets=len(group))
        for column in ("attempted", "scored"):
            for q, label in ((0.25, "q1"), (0.5, "median"), (0.75, "q3")):
                row[f"{column}_{label}"] = float(group[column].quantile(q))
        for cap in (100, 200):
            subset = pairs.loc[pairs.model.eq(model) & pairs.cap.eq(cap)]
            row[f"n_{cap}"] = len(subset)
            row[f"overtuning_{cap}"] = int(subset.overtuning.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def table_rows(summary):
    def number(value):
        if value is None or not np.isfinite(value):
            return "--"
        return str(int(value)) if np.isclose(value, round(value)) else f"{value:.1f}"

    def count(row, key):
        return f"{number(row[key + '_median'])} [{number(row[key + '_q1'])}, {number(row[key + '_q3'])}]"

    def rate(row, cap):
        n, k = (row[f"n_{cap}"], row[f"overtuning_{cap}"])
        return f"{int(k)}/{int(n)} ({100 * k / n:.1f}\\%)" if n else "--"

    lines = []
    for _, row in summary.iterrows():
        lines.append(
            f"{row.model} & {count(row, 'attempted')} & {count(row, 'scored')} & {rate(row, 100)} & {rate(row, 200)} "
            + "\\\\"
        )
    return "\n".join(lines)


def tables_latex(summary, tables=""):
    lines = [
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}\\toprule",
        "Model & Attempted & Scored & Cap 100 vs 25 & Cap 200 vs 25 \\\\",
        "\\midrule",
        table_rows(summary),
        "\\bottomrule\\end{tabular}}",
    ]
    return "\n".join(lines) + "\n" + tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    output = args.input
    searches = pd.read_csv(
        output / "realized_searches.csv", float_precision="round_trip"
    )
    frame = pd.read_csv(
        output / "milestones_by_dataset.csv", float_precision="round_trip"
    )
    curves = pd.read_csv(output / "budget_curves.csv", float_precision="round_trip")
    pairs = pd.read_csv(output / "overtuning_pairs.csv")
    summary = counts_and_rates(searches, pairs)
    gaps = binary_gaps(frame)
    summary.to_csv(output / "model_summary.csv", index=False)
    gaps.to_csv(output / "binary_validation_test_gaps.csv", index=False)
    supplement = output / "supplement.tex"
    (output / "appendix.tex").write_text(
        tables_latex(summary, supplement.read_text() if supplement.exists() else "")
    )
    draw_curves(curves, gaps, output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
