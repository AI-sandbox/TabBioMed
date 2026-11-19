"""Failures analysis."""

from __future__ import annotations
from tabbiomed.analysis.paths import PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.datasets import dataset_directory
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from tabbiomed.analysis.results import (
    BENCHMARK_PLAN,
    RESULTS,
    _test_metric,
    build_results,
)
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.common import (
    analysis_dataset_metadata,
    load_benchmark_models_config,
)

CAUSES = (
    "gpu_oom",
    "host_ram_oom",
    "time_limit",
    "segmentation_fault",
    "mixed",
    "unknown",
)
DOMAINS = {
    "clinical & healthcare": "Clinical and healthcare",
    "drug discovery": "Drug discovery",
    "genomics": "Genomics",
    "metabolomics": "Metabolomics",
    "proteomics": "Proteomics",
    "single-cell": "Single-cell",
    "systems biology": "Systems biology",
    "transcriptomics": "Transcriptomics",
}


def reason_codes(unit, variant):
    block = unit.get(variant) or {}
    candidates = [block]
    if variant == "best":
        candidates.append(unit.get("tuning") or {})
    candidates.append(unit)
    for candidate in candidates:
        reason = (candidate.get("failure") or {}).get("reason")
        if reason == "segmentation_fault_and_gpu_oom":
            return {"segmentation_fault", "gpu_oom"}
        if reason:
            return {reason if reason in CAUSES else "unknown"}
    return {"unknown"}


def variant_failure(report, variant, metric, n_folds):
    if n_folds:
        indexed = {fold["fold_id"]: fold for fold in report.get("cv_fold_metrics", [])}
        units = [(i, indexed.get(i, {})) for i in range(n_folds)]
    else:
        units = [(0, report)]
    codes, failed_folds = (set(), [])
    for fold_id, unit in units:
        if _test_metric(unit.get(variant), metric) is None:
            failed_folds.append(fold_id)
            codes.update(reason_codes(unit, variant))
    if not failed_folds:
        raise ValueError("A complete variant was classified as failed")
    cause = next(iter(codes)) if len(codes) == 1 else "mixed"
    return (cause, sorted(codes), failed_folds)


def rank_sensitivity(raw, tasks):
    normalized = normalized_scores_with_failures(raw, tasks)
    baseline = normalized.rank(axis=1, ascending=False, method="average")
    strict = normalized.where(raw.notna(), -1).rank(
        axis=1, ascending=False, method="average"
    )
    summary = pd.DataFrame(
        {
            "main_mean_rank": baseline.mean(),
            "failure_as_worst_mean_rank": strict.mean(),
            "n_completed": raw.notna().sum(),
            "completion_rate": raw.notna().mean(),
            "conditional_normalized_score": normalized.where(raw.notna()).mean(),
        }
    )
    summary["main_position"] = summary.main_mean_rank.rank(method="min")
    summary["failure_as_worst_position"] = summary.failure_as_worst_mean_rank.rank(
        method="min"
    )
    summary.index.name = "model"
    return (summary.sort_values(["failure_as_worst_mean_rank", "model"]), strict)


def pareto_flags(points):
    values = points[["completion_rate", "conditional_normalized_score"]].to_numpy()
    return np.array(
        [
            not np.any(np.all(values >= point, axis=1) & np.any(values > point, axis=1))
            for point in values
        ]
    )


def plot_completion_frontier(points, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.4, 6.5))
    frontier = points.loc[points.on_pareto_frontier].sort_values("completion_rate")
    ax.plot(
        100 * frontier.completion_rate,
        frontier.conditional_normalized_score,
        color="#376FA3",
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )
    for on_frontier, color, label in [
        (False, "#9299A1", "Other models"),
        (True, "#376FA3", "Pareto frontier"),
    ]:
        part = points.loc[points.on_pareto_frontier.eq(on_frontier)]
        ax.scatter(
            100 * part.completion_rate,
            part.conditional_normalized_score,
            color=color,
            s=65,
            edgecolors="white",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
    full = points.loc[points.completion_rate.eq(1)].sort_values(
        "conditional_normalized_score"
    )
    ys = full.conditional_normalized_score.to_numpy().copy()
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.029)
    ys[-1] = min(ys[-1], 0.91)
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - 0.029)
    for (model, row), y in zip(full.iterrows(), ys):
        ax.annotate(
            model,
            xy=(100, row.conditional_normalized_score),
            xytext=(101.1, y),
            ha="left",
            va="center",
            fontsize=11,
            arrowprops=dict(arrowstyle="-", color="#A5A9AE", linewidth=0.6),
        )
    offsets = {
        "RealTabPFNv2.5": (-6, 17),
        "TabPFNv2.5": (-8, -19),
        "TabICLv2": (-20, -16),
        "CatBoost": (-20, -39),
        "xRFM": (-20, 28),
    }
    for model, row in points.loc[points.completion_rate.lt(1)].iterrows():
        ax.annotate(
            model,
            xy=(100 * row.completion_rate, row.conditional_normalized_score),
            xytext=offsets.get(model, (-8, 15)),
            textcoords="offset points",
            ha="right",
            fontsize=11,
            arrowprops=dict(arrowstyle="-", color="#A5A9AE", linewidth=0.6),
        )
    ax.set(
        xlabel="Dataset completion (%)",
        ylabel="Mean normalized score on completed datasets",
        xlim=(85, 105.1),
        ylim=(0.24, 0.95),
        xticks=[85, 90, 95, 100],
    )
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            output / f"accuracy_completion_pareto.{ext}", bbox_inches="tight", dpi=180
        )
    plt.close(fig)


def collect(datasets, root, config):
    selected = build_results(datasets, config, root=root)
    metadata_hashes, domains = ({}, {})
    for dataset, info in datasets.items():
        path = dataset_directory(dataset) / "metadata.json"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        metadata = analysis_dataset_metadata(dataset, info, PROCESSED)
        metadata_hashes[dataset] = digest
        domains[dataset] = DOMAINS[metadata["label"].lower()]
    selected["domain"] = selected.dataset.map(domains)

    def read_variants(row):
        payload = Path(row.report_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row.report_sha256:
            raise ValueError(
                f"Report changed during analysis: {row.model}/{row.dataset}"
            )
        report = json.loads(payload)
        entries = []
        for variant, field in (("default", "default_metric"), ("best", "tuned_metric")):
            if variant == "best" and not row.hpo_enabled:
                continue
            completed = pd.notna(getattr(row, field))
            cause, codes, folds = (
                ("completed", [], [])
                if completed
                else variant_failure(
                    report,
                    variant,
                    row.primary_metric,
                    datasets[row.dataset]["n_folds"],
                )
            )
            entries.append(
                dict(
                    model=row.model,
                    hpo_enabled=row.hpo_enabled,
                    dataset=row.dataset,
                    domain=row.domain,
                    variant="default" if variant == "default" else "tuned",
                    completed=completed,
                    cause=cause,
                    recorded_causes="|".join(codes),
                    failed_outer_folds="|".join(map(str, folds)),
                )
            )
        return entries

    with ThreadPoolExecutor(max_workers=8) as pool:
        variants = pd.DataFrame(
            [
                entry
                for entries in pool.map(read_variants, selected.itertuples(index=False))
                for entry in entries
            ]
        )
    return (selected, variants, metadata_hashes)


def failure_table(variants, key):
    expected = variants.groupby(key).size().rename("n_expected")
    failed = variants.loc[~variants.completed]
    counts = pd.crosstab(failed[key], failed.cause).reindex(
        index=expected.index, columns=CAUSES, fill_value=0
    )
    counts = counts.join(expected)
    counts["n_failed"] = counts[list(CAUSES)].sum(axis=1)
    return counts.reset_index()


def tables_latex(selected, variants, ranks, by_domain):
    failed = variants.loc[~variants.completed]
    counts = pd.crosstab(failed.model, failed.variant).reindex(
        columns=["default", "tuned"], fill_value=0
    )
    models = list(ranks.index) + sorted(set(selected.model) - set(ranks.index))
    lines = [
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & \\makecell{Missing\\\\Default} & \\makecell{Missing\\\\Tuned} & \\makecell{Usable\\\\results} & \\makecell{Failure-as-worst\\\\mean rank} \\\\",
        "\\midrule",
    ]
    for model in models:
        rows = selected.loc[selected.model == model]
        d = int(counts.loc[model, "default"]) if model in counts.index else 0
        t = int(counts.loc[model, "tuned"]) if model in counts.index else 0
        t_text = "--" if not rows.hpo_enabled.any() else str(t)
        rank = (
            f"{ranks.loc[model, 'failure_as_worst_mean_rank']:.3f}"
            if model in ranks.index
            else "--"
        )
        if model == "HyperFast":
            lines.append("\\midrule")
        lines.append(
            f"{model} & {d} & {t_text} & {rows.metric_value.notna().sum()}/{len(rows)} & {rank} "
            + "\\\\"
        )
    lines.extend(
        [
            "\\bottomrule\\end{tabular}",
            "\\small",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Domain & \\makecell{Missing/\\\\applicable} & \\makecell{GPU\\\\OOM} & \\makecell{RAM\\\\OOM} & \\makecell{Time\\\\limit} & Segfault & Mixed \\\\",
            "\\midrule",
        ]
    )
    assert (
        by_domain.unknown.sum() == 0
    ), "Add unknown-cause counts to the manuscript table"
    for row in by_domain.itertuples(index=False):
        lines.append(
            f"{row.domain} & {row.n_failed}/{row.n_expected} & {row.gpu_oom} & {row.host_ram_oom} & {row.time_limit} & {row.segmentation_fault} & {row.mixed} "
            + "\\\\"
        )
    totals = by_domain.select_dtypes(include="number").sum()
    lines.extend(
        [
            "\\midrule",
            f"Total & {totals.n_failed}/{totals.n_expected} & {totals.gpu_oom} & {totals.host_ram_oom} & {totals.time_limit} & {totals.segmentation_fault} & {totals.mixed} "
            + "\\\\",
            "\\bottomrule\\end{tabular}",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=_OUTPUT_ROOT / "failures")
    args = parser.parse_args()
    ensure_results(args.results_root)
    config = load_benchmark_models_config()
    datasets = load_benchmark(BENCHMARK_PLAN)["datasets"]
    selected, variants, metadata_hashes = collect(datasets, args.results_root, config)
    scope = selected.loc[selected.model.isin(config.expected_models_regression)]
    raw = scope.pivot(index="dataset", columns="model", values="metric_value").reindex(
        columns=config.expected_models_regression
    )
    ranks, strict = rank_sensitivity(raw, scope.groupby("dataset").task.first())
    ranks["on_pareto_frontier"] = pareto_flags(ranks)
    by_model, by_domain = (failure_table(variants, key) for key in ("model", "domain"))
    summary = dict(
        n_datasets=len(datasets),
        n_models=selected.model.nunique(),
        n_applicable_variants=len(variants),
        n_failed_variants=int((~variants.completed).sum()),
        cause_counts=variants.loc[~variants.completed, "cause"]
        .value_counts()
        .to_dict(),
        kendall_tau=float(
            kendalltau(ranks.main_mean_rank, ranks.failure_as_worst_mean_rank).statistic
        ),
        pareto_frontier=list(ranks.index[ranks.on_pareto_frontier]),
        n_changed_positions=int(
            (ranks.main_position != ranks.failure_as_worst_position).sum()
        ),
        metadata_sha256=metadata_hashes,
    )
    assert int(by_domain.n_failed.sum()) == summary["n_failed_variants"]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plot_completion_frontier(ranks, output)
    selected.to_csv(output / "selected_results.csv", index=False)
    variants.to_csv(output / "variant_completion.csv", index=False)
    variants.loc[~variants.completed].to_csv(
        output / "failed_variants.csv", index=False
    )
    by_model.to_csv(output / "failures_by_model.csv", index=False)
    by_domain.to_csv(output / "failures_by_domain.csv", index=False)
    ranks.to_csv(output / "rank_and_completion.csv")
    strict.to_csv(output / "failure_as_worst_dataset_ranks.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "appendix.tex").write_text(
        tables_latex(selected, variants, ranks, by_domain)
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "metadata_sha256"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
