"""Model summary tables."""

from __future__ import annotations
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.datasets import load_splits
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, read_result
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.common import load_benchmark_models_config


def timing_rates(block, sizes):
    train_field = "post_hpo_train_wall_time_seconds"
    if block.get(train_field) is None:
        train_field = "train_wall_time_seconds"
    train = block.get(train_field)
    inference = block.get("test_inference_wall_time_seconds")
    fitted = sizes["train"] + (
        sizes["val"] if block.get("train_val_merged_for_fit") else 0
    )
    for key in ("n_train_fitted", "n_fit_rows", "n_train_samples"):
        if block.get(key) is not None:
            fitted = int(block[key])
            break
    predicted = block.get("n_test_rows", block.get("n_test", sizes["test"]))

    def rate(seconds, rows):
        if seconds is None or not np.isfinite(seconds) or seconds < 0 or (not rows):
            return np.nan
        return float(seconds) * 1000 / rows

    train_rate = rate(train, fitted) if "train_val_merged_for_fit" in block else np.nan
    return dict(
        train_per_1k=train_rate,
        inference_per_1k=rate(inference, predicted),
        train_seconds=train,
        inference_seconds=inference,
        n_fitted=fitted,
        n_test=predicted,
        train_time_field=train_field,
    )


def split_sizes(dataset, info):
    splits = load_splits(dataset)
    folds = splits["cv_folds"] if info["n_folds"] else [splits]
    if len(folds) != (info["n_folds"] or 1):
        raise ValueError(f"Unexpected outer fold count: {dataset}")
    return [
        {key: len(fold.get(key, [])) for key in ("train", "val", "test")}
        for fold in folds
    ]


def selected_row(job):
    model, dataset, info, sizes = job
    row = read_result(model, dataset, info)
    row.update(train_per_1k=np.nan, inference_per_1k=np.nan)
    timing = []
    if row["selected_variant"] == "failed":
        return (row, timing)
    payload = Path(row["report_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["report_sha256"]:
        raise ValueError(f"Source report changed while reading: {row['report_path']}")
    report = json.loads(payload)
    units = (
        {fold["fold_id"]: fold for fold in report["cv_fold_metrics"]}
        if info["n_folds"]
        else {0: report}
    )
    key = "best" if row["selected_variant"] == "tuned" else "default"
    for fold, counts in enumerate(sizes):
        timing.append(
            dict(
                model=model,
                dataset=dataset,
                fold=fold,
                selected_variant=row["selected_variant"],
                **timing_rates(units[fold][key], counts),
            )
        )
    for metric in ("train_per_1k", "inference_per_1k"):
        values = [fold[metric] for fold in timing]
        if np.isfinite(values).all():
            row[metric] = float(np.mean(values))
    return (row, timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT_ROOT / "model_summary")
    args = parser.parse_args()
    ensure_results()
    plan = load_benchmark(BENCHMARK_PLAN)
    config = load_benchmark_models_config()
    models = config.expected_models_regression
    sizes = {
        dataset: split_sizes(dataset, info)
        for dataset, info in plan["datasets"].items()
    }
    jobs = [
        (model, dataset, info, sizes[dataset])
        for dataset, info in sorted(plan["datasets"].items())
        for model in models
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(selected_row, jobs))
    frame = pd.DataFrame([row for row, _ in results])
    raw = frame.pivot(index="dataset", columns="model", values="metric_value").reindex(
        columns=models
    )
    tasks = frame.groupby("dataset").task.first()
    normalized = normalized_scores_with_failures(raw, tasks)
    ranks = pd.DataFrame(
        rankdata(-normalized, axis=1), index=raw.index, columns=raw.columns
    )
    rows = []
    for model in models:
        selected = frame[frame.model == model]
        rows.append(
            dict(
                model=model,
                family=config.model_family[model],
                rank_min=ranks[model].min(),
                rank_max=ranks[model].max(),
                rank_mean=ranks[model].mean(),
                norm_mean=normalized[model].mean(),
                norm_median=normalized[model].median(),
                inference_per_1k=selected.inference_per_1k.median(),
                train_per_1k=selected.train_per_1k.median(),
                n_success=int(selected.metric_value.notna().sum()),
                n_datasets=len(raw),
                n_inference_times=int(selected.inference_per_1k.notna().sum()),
                n_train_times=int(selected.train_per_1k.notna().sum()),
            )
        )
    summary = pd.DataFrame(rows).sort_values(["rank_mean", "model"])
    summary = summary[
        [
            "model",
            "family",
            "rank_min",
            "rank_mean",
            "rank_max",
            "norm_mean",
            "norm_median",
            "inference_per_1k",
            "train_per_1k",
            "n_success",
            "n_datasets",
            "n_inference_times",
            "n_train_times",
        ]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "selected_results.csv", index=False)
    pd.DataFrame([row for _, rows in results for row in rows]).to_csv(
        args.output / "timing_by_split.csv", index=False
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    normalized.to_csv(args.output / "normalized_scores.csv")
    ranks.to_csv(args.output / "ranks.csv")
    latex = []
    for row in summary.itertuples(index=False):
        latex.append(
            f"{row.model} & {row.family} & {row.rank_min:.1f} & {row.rank_mean:.3f} & {row.rank_max:.1f} & {row.norm_mean:.3f} & {row.norm_median:.3f} & {row.inference_per_1k:.3f} & {row.train_per_1k:.3f} & {row.n_success}/{row.n_datasets} "
            + "\\\\"
        )
    (args.output / "table_rows.tex").write_text("\n".join(latex) + "\n")
    manifest = dict(
        results_root=str(RESULTS),
        datasets=sorted(plan["datasets"]),
        models=list(models),
        source_counts=frame.metric_source.value_counts().to_dict(),
        timing="Selected final fit only, post_hpo_train preferred. Per-fold seconds per 1000 actual fitted/test rows, then fold mean, then dataset median. Incomplete timings excluded.",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
