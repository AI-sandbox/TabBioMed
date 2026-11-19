"""Elo features figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-feature-terciles-mpl")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from threadpoolctl import threadpool_limits
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS
from tabbiomed.analysis.common import (
    analysis_dataset_metadata,
    load_benchmark_models_config,
)
from tabbiomed.analysis.figures.elo_regimes import (
    collect_model_inputs,
    digest,
    models_for_case,
    plot_cleveland_quadrants_elo,
    prepare_case,
)
from tabbiomed.analysis.elo.splitwise import rate

LABELS = ("Low-feature", "Medium-feature", "High-feature")
OUTPUT = _OUTPUT_ROOT / "elo_features"


def assign_terciles(metadata):
    if metadata.dataset.duplicated().any() or len(metadata) < 3:
        raise ValueError("Feature terciles require at least three unique datasets")
    out = metadata.sort_values("dataset").copy()
    widths = pd.to_numeric(out.num_features)
    if not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("Feature counts must be positive and finite")
    out["feature_tercile"] = pd.qcut(
        widths.rank(method="first"), q=3, labels=LABELS
    ).astype(str)
    return out


def summarize(inputs, models, bootstrap, seed):
    summaries, details = ([], {})
    for label in LABELS:
        frame = inputs[inputs.feature_tercile.eq(label)]
        if frame.empty or any(
            (set(g.model) != set(models) for _, g in frame.groupby("dataset"))
        ):
            raise ValueError(f"Incomplete model grid in {label}")
        print(
            f"{label}: {frame.dataset.nunique()} tasks, {len(models)} models, {int(frame.imputed.sum())} RF fills",
            flush=True,
        )
        ratings, _, meta = rate(
            frame, bootstrap=bootstrap, seed=seed, reference="RandomForest"
        )
        if set(ratings.competitor) != set(models):
            raise ValueError("Elo omitted a requested model")
        ratings = ratings.rename(
            columns={
                "competitor": "model",
                "elo": "elo_calibrated",
                "elo_lo_95": "elo_cal_lo_95",
                "elo_hi_95": "elo_cal_hi_95",
            }
        )
        counts = frame.groupby("model").imputed.sum()
        ratings["feature_tercile"] = label
        ratings["n_datasets"] = frame.dataset.nunique()
        ratings["n_imputed"] = ratings.model.map(counts).astype(int)
        ratings["n_success"] = ratings.n_datasets - ratings.n_imputed
        ratings["completion_pct"] = ratings.n_success / ratings.n_datasets * 100
        ratings["elo_rank"] = ratings.elo_calibrated.rank(
            ascending=False, method="average"
        )
        summaries.append(
            ratings.sort_values(["elo_calibrated", "model"], ascending=[False, True])
        )
        details[label] = {**meta, "n_imputed_cells": int(frame.imputed.sum())}
    return (pd.concat(summaries, ignore_index=True), details)


def rank_agreement(summary):
    rows = []
    for left, right in combinations(LABELS, 2):
        a = summary[summary.feature_tercile.eq(left)].set_index("model").elo_rank
        b = summary[summary.feature_tercile.eq(right)].set_index("model").elo_rank
        if set(a.index) != set(b.index):
            raise ValueError("Rank correlations require matching rosters")
        result = kendalltau(a.sort_index(), b.sort_index())
        rows.append(
            dict(
                left=left,
                right=right,
                n_models=len(a),
                kendall_tau=float(result.statistic),
                p_value=float(result.pvalue),
            )
        )
    return pd.DataFrame(rows)


def table_latex(summary):
    lines = [
        "\\small",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "\\textbf{Regime} & \\textbf{Model} & \\textbf{Elo rating} & \\textbf{Completion (\\%)} \\\\",
        "\\midrule",
    ]
    for label in LABELS:
        top = summary[summary.feature_tercile.eq(label)].nlargest(3, "elo_calibrated")
        for i, row in enumerate(top.itertuples()):
            lines.append(
                f"{(label if i == 0 else '')} & {row.model} & {row.elo_calibrated:.1f} & {row.completion_pct:.2f} "
                + "\\\\"
            )
        if label != LABELS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ensure_results()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    config = load_benchmark_models_config()
    models = models_for_case(
        config.expected_models_binary_classification, "all_tasks"
    )
    plan = load_benchmark(BENCHMARK_PLAN)
    if set(models) != set(config.expected_models_regression):
        raise ValueError("Feature-count inputs differ from the configured model scope")
    raw, metadata = collect_model_inputs(plan["datasets"], models, RESULTS, PROCESSED)
    for row in metadata.itertuples():
        analysis_dataset_metadata(row.dataset, plan["datasets"][row.dataset], PROCESSED)
    metadata = assign_terciles(metadata)
    inputs = prepare_case(raw, models, "all_tasks").merge(
        metadata[["dataset", "feature_tercile"]], on="dataset", validate="many_to_one"
    )
    if len(inputs) != len(plan["datasets"]) * len(models):
        raise ValueError("Expected exactly one score per model and dataset")
    with threadpool_limits(limits=1):
        summary, fits = summarize(inputs, models, args.bootstrap, args.seed)
    agreement = rank_agreement(summary)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "selected_results.csv", index=False)
    inputs.to_csv(output / "inputs.csv", index=False)
    metadata.to_csv(output / "dataset_terciles.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    agreement.to_csv(output / "rank_agreement.csv", index=False)
    (output / "table.tex").write_text(table_latex(summary))
    plot_cleveland_quadrants_elo(
        summary=summary.rename(columns={"feature_tercile": "quadrant"}),
        quad_to_ds_meta={
            label: metadata[metadata.feature_tercile.eq(label)].to_dict("records")
            for label in LABELS
        },
        out_pdf=output / "feature_tercile_cleveland.pdf",
        panel_labels=LABELS,
        size_inches=(13.5, 10.2),
    )
    ranges = {
        label: dict(
            n_datasets=len(g),
            min_features=int(g.num_features.min()),
            max_features=int(g.num_features.max()),
        )
        for label in LABELS
        for g in [metadata[metadata.feature_tercile.eq(label)]]
    }
    manifest = dict(
        results_root=str(RESULTS),
        n_datasets=len(plan["datasets"]),
        models=models,
        n_models=len(models),
        n_inputs=len(inputs),
        n_imputed_cells=int(inputs.imputed.sum()),
        bootstrap=args.bootstrap,
        seed=args.seed,
        terciles=ranges,
        fits=fits,
        score_policy="One competitor per model, complete Tuned test mean then own complete Default mean. LinearModel and NeoLsSVM use Default. Only pairs missing both own variants use the same task's complete RF Default mean. No mixing of variants across outer folds. Half-win ties.",
        selection_counts={
            str(k): int(v) for k, v in inputs.metric_source.value_counts().items()
        },
        calibration="RandomForest model competitor = 1000",
        dataset_assignment="Feature-count rank-based terciles, dataset-name order breaks equal-width ties",
        all_metadata_hashes_verified=True,
        benchmark_plan_sha256=digest(BENCHMARK_PLAN),
        input_sha256=digest(output / "inputs.csv"),
        script_sha256=digest(
            Path(__file__).resolve().parents[1] / "figures/elo_features.py"
        ),
        renderer_sha256=digest(
            Path(__file__).resolve().parents[1] / "figures/elo_regimes.py"
        ),
        figure_sha256=digest(output / "feature_tercile_cleveland.pdf"),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        summary.groupby("feature_tercile", sort=False)
        .head(3)[["feature_tercile", "model", "elo_calibrated", "completion_pct"]]
        .to_string(index=False)
    )
    print(agreement.to_string(index=False))


if __name__ == "__main__":
    main()
