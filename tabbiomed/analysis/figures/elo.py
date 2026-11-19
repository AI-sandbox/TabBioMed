"""Elo figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import PLAN, RESULTS
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.common import PRIMARY_METRIC, load_benchmark_models_config
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabbiomed.analysis.elo.plotting import plot_bars
from tabbiomed.analysis.elo.api import (
    MISSING_POLICIES,
    apply_missing_policy,
    rate,
)

CLASSIFICATION_ONLY = load_benchmark_models_config().regression_optional_missing_models


def finite_metric(block, metric):
    if (
        not isinstance(block, dict)
        or not block.get("fixed_split_test_evaluated")
        or block.get("error")
        or block.get("post_hpo_test_error")
    ):
        return None
    value = block.get("test_metrics", {}).get(metric)
    return (
        float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None
    )


def read_consolidated_pair(root, job, info):
    path = root / job["model"] / job["dataset"] / "report.json"
    result = {
        "model": job["model"],
        "dataset": job["dataset"],
        "task": info["task"],
        "report_path": str(path),
        "default": None,
        "tuned": None,
        "split_results": {},
    }
    payload = path.read_bytes()
    report = json.loads(payload)
    if (
        report.get("artifact_type") != "evaluation_summary"
        or report.get("schema_version") != 1
        or str(report.get("model", "")).lower() != job["model"].lower()
        or (report.get("dataset") != job["dataset"])
    ):
        raise ValueError(f"Invalid consolidated Main report: {path}")
    metric = PRIMARY_METRIC[info["task"]]
    if report.get("primary_metric") != metric:
        raise ValueError(f"Unexpected primary metric: {path}")
    result["report_sha256"] = hashlib.sha256(payload).hexdigest()
    result["report_status"] = report.get("status")
    result["hpo_enabled"] = report["hpo_enabled"]
    variants = report.get("evaluated_variants", ["default", "best"])
    result["evaluated_variants"] = variants
    if info["n_folds"]:
        folds = report.get("cv_fold_metrics", [])
        indexed = {f["fold_id"]: f for f in folds}
        if len(indexed) != len(folds) or set(indexed) - set(range(info["n_folds"])):
            raise ValueError(f"Invalid outer folds: {path}")
        units = [indexed.get(i, {}) for i in range(info["n_folds"])]
    else:
        units = [report]
    for label, key in [("default", "default"), ("tuned", "best")]:
        if key not in variants or (label == "tuned" and not report["hpo_enabled"]):
            continue
        values = [finite_metric(unit.get(key), metric) for unit in units]
        result["split_results"][label] = values
        if not info["n_folds"]:
            result[label] = values[0]
    return result


def average_cv_results(frame):
    if frame.duplicated(["dataset", "fold", "competitor"]).any():
        raise ValueError("Duplicate model/split result")
    rows = []
    for _, group in frame.groupby(["dataset", "competitor"], sort=False):
        row = group.iloc[0].to_dict()
        values = pd.to_numeric(group.metric_value, errors="raise")
        row["metric_value"] = (
            float(values.mean()) if np.isfinite(values).all() else np.nan
        )
        row["fold"] = 0
        row["n_outer_splits"] = len(group)
        row["imputed"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def leaderboard(
    frame,
    competitors,
    output,
    bootstrap,
    seed,
    policy,
    cv_aggregation="splitwise",
):
    output.mkdir(parents=True, exist_ok=True)
    frame, missing_info = apply_missing_policy(frame, policy)
    if frame.empty:
        return dict(status="no_eligible_results", **missing_info)
    table, gaps, stats = rate(
        frame,
        bootstrap,
        seed,
        progress=lambda done, total: print(
            f"{policy}: bootstrap {done}/{total}", flush=True
        ),
    )
    if cv_aggregation == "mean":
        stats["split_weights"] = "1 per dataset after averaging all outer splits"
    table = table.sort_values("elo", ascending=False)
    table.to_csv(output / "elo_all_entries.csv", index=False)
    table.assign(
        model=table.competitor.str.split("::").str[0],
        variant=table.competitor.str.split("::").str[1],
    ).pivot(index="model", columns="variant", values="elo").to_csv(
        output / "elo_by_model.csv"
    )
    frame.to_csv(output / "metric_inputs.csv", index=False)
    gaps.to_csv(output / "tuned_default_gaps.csv", index=False)
    (output / "datasets.json").write_text(
        json.dumps(sorted(frame.dataset.unique()), indent=2) + "\n"
    )
    models = list(dict.fromkeys((c.split("::")[0] for c in competitors)))
    plot_bars(
        table,
        frame,
        models,
        output / "elo",
        transparent_bars=(policy == "complete_case"),
    )
    return dict(
        stats,
        **missing_info,
        task_counts=dict(Counter(frame.drop_duplicates("dataset").task)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--models", nargs="+", help="Model subset from the selected plan"
    )
    parser.add_argument(
        "--cv-aggregation",
        choices=["splitwise", "mean"],
        default="mean",
        help="Average all CV metrics before Elo and missing-data handling (default), or compare individual splits",
    )
    parser.add_argument(
        "--missing-policy",
        nargs="+",
        choices=MISSING_POLICIES,
        default=["complete_case", "rf_imputed"],
    )
    args = parser.parse_args()
    ensure_results()
    plan = load_benchmark(args.plan)
    excluded_models = sorted(set(plan["models"]) & CLASSIFICATION_ONLY)
    plan["models"] = {
        m: settings
        for m, settings in plan["models"].items()
        if m not in CLASSIFICATION_ONLY
    }
    if args.models is not None:
        if len(set(args.models)) != len(args.models):
            parser.error("Duplicate model selection")
        unknown = set(args.models) - set(plan["models"])
        if unknown:
            parser.error(f"Models outside the all-task plan: {sorted(unknown)}")
        plan["models"] = {model: plan["models"][model] for model in args.models}
    jobs = [
        {"model": model, "dataset": dataset}
        for model in plan["models"]
        for dataset in plan["datasets"]
    ]
    root = RESULTS
    if not root.is_dir():
        raise FileNotFoundError(root)
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(read_consolidated_pair, root, j, plan["datasets"][j["dataset"]])
            for j in jobs
        ]
        for i, future in enumerate(as_completed(futures), 1):
            pairs.append(future.result())
            if i % 200 == 0:
                print(f"Loaded {i}/{len(futures)} model/dataset reports", flush=True)
    pairs.sort(key=lambda p: (p["model"], p["dataset"]))
    (args.output / "selected_reports.json").write_text(
        json.dumps(pairs, indent=2) + "\n"
    )
    rows = []
    for p in pairs:
        if p["hpo_enabled"] != plan["models"][p["model"]]["hpo_enabled"]:
            raise ValueError(
                f"Report HPO policy differs from analysis plan: {p['report_path']}"
            )
        for variant in ("default", "tuned"):
            if variant == "tuned" and not p["hpo_enabled"]:
                continue
            key = "best" if variant == "tuned" else "default"
            if key not in p.get("evaluated_variants", ["default", "best"]):
                continue
            count = plan["datasets"][p["dataset"]]["n_folds"] or 1
            values = p["split_results"].get(variant, [None] * count)
            for fold, value in enumerate(values):
                rows.append(
                    {
                        "model": p["model"],
                        "hpo_enabled": p["hpo_enabled"],
                        "dataset": p["dataset"],
                        "task": p["task"],
                        "variant": variant,
                        "competitor": f"{p['model']}::{variant}",
                        "fold": fold,
                        "metric_value": value,
                        "imputed": False,
                        "report_path": p["report_path"],
                    }
                )
    original = pd.DataFrame(rows)
    if args.cv_aggregation == "mean":
        original.to_csv(args.output / "availability_outer_splits.csv", index=False)
        original = average_cv_results(original)
    original.to_csv(args.output / "availability.csv", index=False)
    competitors = [
        f"{m}::{v}"
        for m in plan["models"]
        for v in ("default", "tuned")
        if v == "default" or plan["models"][m]["hpo_enabled"]
    ]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(root),
        "plan": str(args.plan),
        "models": list(plan["models"]),
        "excluded_models": excluded_models,
        "policy": "Consolidated Main report.json files only. No private-run, PCA or A100 fallback. Compare stored outer splits with equal total dataset weight. Explicit missing policies, task-ineligible pairs omitted. Default and tuned share one Elo pool, no test-score clamping.",
        "elo": "Bradley-Terry MLE, ties half, tolerance 1e-10. Default RF anchored to 1000, dataset bootstrap intervals with point-fit calibration.",
        "bootstrap": args.bootstrap,
        "seed": 42,
    }
    summary["cv_aggregation"] = args.cv_aggregation
    if args.cv_aggregation == "mean":
        summary["policy"] = (
            "Consolidated Main report.json files only. No private-run, PCA or A100 fallback. Average all outer-test metrics per dataset and variant before pairwise comparisons. An incomplete CV evaluation has no mean. Apply missing policies after averaging, using the dataset mean of RF Default for imputation. Default and tuned share one Elo pool, no test-score clamping."
        )
    summary["report_count"] = len(pairs)
    summary["missing_reports"] = [p["report_path"] for p in pairs if p.get("error")]
    for name in args.missing_policy:
        summary[name] = leaderboard(
            original,
            competitors,
            args.output / name,
            args.bootstrap,
            42,
            name,
            cv_aggregation=args.cv_aggregation,
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
