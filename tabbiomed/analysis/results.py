from __future__ import annotations
from tabbiomed.analysis.paths import PLAN, RESULTS
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from tabbiomed.results.selection import test_available
from tabbiomed.analysis.common import (
    BenchmarkModelsConfig,
    PRIMARY_METRIC,
)

BENCHMARK_PLAN = PLAN


def valid_split_identity(timing, info):
    observed = timing.get("dataset_split_sha256")
    expected = (info.get("splits_sha256"), info.get("recorded_split_sha256"))
    return isinstance(observed, str) and observed in expected


def _test_metric(block: Any, metric: str) -> float | None:
    if not test_available(block, metric):
        return None
    return float(block["test_metrics"][metric])


def read_result(
    model: str, dataset: str, info: dict[str, Any], *, root: Path = RESULTS
) -> dict[str, Any]:
    path = root / model / dataset / "report.json"
    payload = path.read_bytes()
    report = json.loads(payload)
    metric = PRIMARY_METRIC[info["task"]]
    if (
        type(report.get("hpo_enabled")) is not bool
        or report.get("artifact_type") != "evaluation_summary"
        or report.get("schema_version") != 1
        or str(report.get("model", "")).lower() != model.lower()
        or (report.get("dataset") != dataset)
        or (report.get("primary_metric") != metric)
    ):
        raise ValueError(f"Invalid consolidated Main report: {path}")
    n_folds = int(info["n_folds"])
    if n_folds:
        folds = report.get("cv_fold_metrics", [])
        indexed = {fold["fold_id"]: fold for fold in folds}
        if len(indexed) != len(folds) or set(indexed) - set(range(n_folds)):
            raise ValueError(f"Invalid outer folds: {path}")
        units = [indexed.get(i, {}) for i in range(n_folds)]
    else:
        if report.get("cv_fold_metrics"):
            raise ValueError(f"Unexpected CV results in a fixed-split report: {path}")
        units = [report]
    variants = report.get("evaluated_variants", ["default", "best"])
    default_only = not report["hpo_enabled"]
    scores: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for variant in ("default", "best"):
        values = [
            (
                _test_metric(unit.get(variant), metric)
                if variant in variants and (not (variant == "best" and default_only))
                else None
            )
            for unit in units
        ]
        counts[variant] = sum((value is not None for value in values))
        scores[variant] = (
            float(np.mean(values)) if counts[variant] == len(units) else None
        )
    if scores["best"] is not None:
        value, source, variant = (scores["best"], "tuned_test", "tuned")
    elif scores["default"] is not None:
        value = scores["default"]
        source = "default_only" if default_only else "default_fallback"
        variant = "default"
    else:
        value, source, variant = (None, "failed", "failed")
    return {
        "model": model,
        "hpo_enabled": report["hpo_enabled"],
        "dataset": dataset,
        "task": info["task"],
        "primary_metric": metric,
        "metric_value": value,
        "metric_source": source,
        "selected_variant": variant,
        "default_metric": scores["default"],
        "tuned_metric": scores["best"],
        "n_outer_splits": len(units),
        "n_default_splits": counts["default"],
        "n_tuned_splits": counts["best"],
        "report_path": str(path),
        "report_sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_results(
    datasets: dict[str, dict[str, Any]],
    bm: BenchmarkModelsConfig,
    *,
    root: Path = RESULTS,
) -> pd.DataFrame:
    jobs = []
    for dataset, info in sorted(datasets.items()):
        models = (
            bm.expected_models_regression
            if info["task"] in ("regression", "multiregression")
            else bm.expected_models_binary_classification
        )
        jobs.extend(((model, dataset, info) for model in models))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda job: read_result(*job, root=root), jobs))
    return pd.DataFrame(rows)
