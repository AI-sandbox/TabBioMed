"""Hpo sensitivity analysis."""

from __future__ import annotations
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from threadpoolctl import threadpool_limits
from tabbiomed.analysis.results import BENCHMARK_PLAN, build_results
from tabbiomed.analysis.common import (
    load_benchmark_models_config,
)
from tabbiomed.analysis.elo.missing import apply_missing_policy
from tabbiomed.analysis.elo.plotting import plot_bars
from tabbiomed.analysis.elo.splitwise import _HIGHER_IS_BETTER_METRICS, rate

CAPS = (25, 50, 100, 200)
BLUE, ORANGE = ("#376FA3", "#D68936")


def default_comparisons(selected):
    paired = selected.loc[
        selected.hpo_enabled
        & selected.default_metric.notna()
        & selected.tuned_metric.notna()
    ].copy()
    maximize = (
        paired.primary_metric.astype(str).str.lower().isin(_HIGHER_IS_BETTER_METRICS)
        if "primary_metric" in paired.columns
        else paired.task.eq("binary")
    )
    paired["degradation"] = np.where(
        maximize,
        paired.default_metric - paired.tuned_metric,
        paired.tuned_metric - paired.default_metric,
    )
    paired["outcome"] = np.select(
        [paired.degradation > 1e-12, paired.degradation < -1e-12],
        ["default_better", "tuned_better"],
        default="tie",
    )
    paired["magnitude"] = paired.degradation
    paired.loc[paired.primary_metric.eq("roc_auc"), "magnitude"] *= 100
    mse = paired.primary_metric.eq("mse")
    denominator = paired.default_metric.abs().where(paired.default_metric.abs() > 0)
    paired.loc[mse, "magnitude"] = (
        100 * paired.loc[mse, "degradation"] / denominator[mse]
    )
    frequencies = pd.crosstab(paired.model, paired.outcome).reindex(
        columns=["tuned_better", "tie", "default_better"], fill_value=0
    )
    frequencies["n"] = frequencies.sum(axis=1)
    magnitudes = []
    for (model, metric), group in paired.groupby(["model", "primary_metric"]):
        bad = group.loc[group.outcome.eq("default_better"), "magnitude"].dropna()
        magnitudes.append(
            dict(
                model=model,
                metric=metric,
                n=len(group),
                n_worse=int(group.outcome.eq("default_better").sum()),
                n_defined=len(bad),
                q1=bad.quantile(0.25),
                median=bad.median(),
                q3=bad.quantile(0.75),
                maximum=bad.max(),
            )
        )
    return (paired, frequencies, pd.DataFrame(magnitudes))


def logloss_gaps(frame):
    rows = []
    for (model, task), group in frame.loc[
        frame.task.isin(["binary", "classification"])
    ].groupby(["model", "task"]):
        validation = group.pivot(
            index="dataset", columns="cap", values="validation_log_loss"
        ).reindex(columns=CAPS)
        test = group.pivot(
            index="dataset", columns="cap", values="test_log_loss"
        ).reindex(columns=CAPS)
        common = validation.notna().all(axis=1) & test.notna().all(axis=1)
        for cap in CAPS:
            gap = test.loc[common, cap] - validation.loc[common, cap]
            rows.append(
                dict(
                    model=model,
                    task=task,
                    cap=cap,
                    n=int(common.sum()),
                    mean_gap=gap.mean(),
                    median_gap=gap.median(),
                )
            )
    return pd.DataFrame(rows)


def plot_logloss(gaps, output):
    gaps = gaps.loc[~gaps.model.eq("iLTM")]
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(7, 3, figsize=(12.6, 12.6), sharex=True, sharey=True)
    finite = gaps.mean_gap.dropna()
    low, high = (min(0, finite.min()), max(0, finite.max()))
    margin = max(0.01, (high - low) * 0.1)
    for ax, model in zip(axes.ravel(), sorted(gaps.model.unique())):
        counts = []
        for task, color in [("binary", BLUE), ("classification", ORANGE)]:
            data = gaps.loc[gaps.model.eq(model) & gaps.task.eq(task)].sort_values(
                "cap"
            )
            n = int(data.n.iloc[0]) if len(data) else 0
            counts.append(n)
            if n:
                ax.plot(
                    data.cap,
                    data.mean_gap,
                    color=color,
                    linewidth=1.8,
                    marker="o",
                    markersize=11,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                )
        ax.set_title(f"{model}  ({counts[0]}/{counts[1]})", loc="left", pad=5)
        ax.axhline(0, color="#999999", linewidth=0.7, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CAPS, [str(cap) for cap in CAPS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(22, 225)
        ax.set_ylim(low - margin, high + margin)
        ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    for ax in axes.ravel()[gaps.model.nunique() :]:
        ax.set_axis_off()
    fig.legend(
        handles=[
            Line2D([], [], color=color, marker="o", markersize=9, label=name)
            for name, color in [("Binary", BLUE), ("Multiclass", ORANGE)]
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=16,
    )
    fig.supxlabel("Attempted-configuration cap", fontsize=17, y=0.012)
    fig.supylabel("Test - validation log loss", fontsize=17, x=0.006)
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.055, top=0.95, hspace=0.59, wspace=0.15
    )
    for ext in ("pdf", "png"):
        fig.savefig(
            output / f"hpo_logloss_validation_test_gaps.{ext}",
            bbox_inches="tight",
            dpi=150,
        )
    plt.close(fig)


def elo_inputs(selected, milestones, models, cap):
    tuned = milestones.loc[milestones.cap.eq(cap)].set_index(["model", "dataset"])
    rows = []
    for row in selected.loc[selected.model.isin(models)].itertuples():
        for variant in ("default", "tuned"):
            if variant == "tuned" and not row.hpo_enabled:
                continue
            value = row.default_metric
            if variant == "tuned":
                key = (row.model, row.dataset)
                value = tuned.at[key, "test"] if key in tuned.index else np.nan
            rows.append(
                dict(
                    model=row.model,
                    hpo_enabled=row.hpo_enabled,
                    dataset=row.dataset,
                    task=row.task,
                    variant=variant,
                    competitor=f"{row.model}::{variant}",
                    fold=0,
                    metric_value=value,
                )
            )
    return apply_missing_policy(pd.DataFrame(rows), "rf_imputed")


def verify_official(selected, milestones, models):
    actual = milestones.loc[milestones.cap.eq(100)].set_index(["model", "dataset"]).test
    public = (
        selected.loc[selected.hpo_enabled].set_index(["model", "dataset"]).tuned_metric
    )
    actual = actual.reindex(public.index)
    if not actual.isna().equals(public.isna()):
        raise ValueError("Cap 100 coverage differs from consolidated official results")
    np.testing.assert_allclose(actual.dropna(), public.dropna(), rtol=0, atol=1e-12)
    frame, _ = elo_inputs(selected, milestones, models, 100)
    table, _, _ = rate(frame, bootstrap=2)
    old = pd.read_csv(
        _OUTPUT_ROOT / "elo" / "rf_imputed/elo_all_entries.csv",
        float_precision="round_trip",
    )
    np.testing.assert_allclose(
        table.set_index("competitor").elo.reindex(old.competitor),
        old.elo,
        rtol=0,
        atol=1e-07,
    )
    return int(public.notna().sum())


def tables_latex(paired, frequencies):
    lines = [
        "\\small",
        "\\begin{tabular}{lrrrr}\\toprule",
        "Model & Paired & Tuned better & Tie & Default better \\\\",
        "\\midrule",
    ]
    for model, row in frequencies.iterrows():
        cells = [
            f"{int(row[key])} ({100 * row[key] / row.n:.1f}\\%)"
            for key in ("tuned_better", "tie", "default_better")
        ]
        lines.append(f"{model} & {int(row.n)} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule\\end{tabular}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}\\toprule",
        "Primary metric & Paired & Tuned worse & Median [Q1, Q3] & Maximum \\\\",
        "\\midrule",
    ]
    for metric, name in [
        ("roc_auc", "ROC-AUC decrease (pp)"),
        ("log_loss", "Multiclass log-loss increase"),
        ("mse", "Relative MSE increase (\\%)"),
    ]:
        subset = paired.loc[paired.primary_metric.eq(metric)]
        worse = subset.loc[subset.outcome.eq("default_better")]
        values = worse.magnitude.dropna()
        lines.append(
            f"{name} & {len(subset)} & {len(worse)} ({100 * len(worse) / len(subset):.1f}\\%) & {values.median():.3g} [{values.quantile(0.25):.3g}, {values.quantile(0.75):.3g}] & {values.max():.3g}"
            + " \\\\"
        )
    lines += ["\\bottomrule\\end{tabular}}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    ensure_results()
    output = args.input
    milestones = pd.read_csv(
        output / "milestones_by_dataset.csv", float_precision="round_trip"
    )
    config = load_benchmark_models_config()
    models = list(config.expected_models_regression)
    selected = build_results(load_benchmark(BENCHMARK_PLAN)["datasets"], config)
    selected.to_csv(output / "main_results.csv", index=False)
    paired, frequencies, magnitudes = default_comparisons(selected)
    paired.to_csv(output / "default_tuned_pairs.csv", index=False)
    frequencies.to_csv(output / "default_tuned_frequencies.csv")
    magnitudes.to_csv(
        output / "default_tuned_degradation_by_model_metric.csv", index=False
    )
    gaps = logloss_gaps(milestones)
    gaps.to_csv(output / "logloss_validation_test_gaps.csv", index=False)
    plot_logloss(gaps, output)
    with threadpool_limits(limits=1):
        checked = verify_official(selected, milestones, models)
        frame, missing = elo_inputs(selected, milestones, models, 200)
        table, elo_gaps, summary = rate(
            frame,
            bootstrap=200,
            seed=42,
            progress=lambda i, n: print(f"Elo bootstrap {i}/{n}", flush=True),
        )
    table.to_csv(output / "hpo_200_elo.csv", index=False)
    elo_gaps.to_csv(output / "hpo_200_tuned_default_elo_gaps.csv", index=False)
    frame.to_csv(output / "hpo_200_elo_inputs.csv", index=False)
    plot_bars(table, frame, models, output / "hpo_200_elo")
    summary.update(missing)
    summary.update(
        cap=200,
        verified_official_test_scores=checked,
        main_report_sha256=dict(zip(selected.report_path, selected.report_sha256)),
        milestone_sha256=hashlib.sha256(
            (output / "milestones_by_dataset.csv").read_bytes()
        ).hexdigest(),
    )
    (output / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output / "supplement.tex").write_text(tables_latex(paired, frequencies))
    print(frequencies.to_string())
    print("Verified official Tuned scores:", checked)
    print("Log-loss curve cohort counts:", gaps.loc[gaps.cap.eq(25)].to_dict("records"))


if __name__ == "__main__":
    main()
