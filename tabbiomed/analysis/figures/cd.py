"""Cd figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, build_results
from tabbiomed.analysis.ranks import run_cd_pipeline
from tabbiomed.analysis.common import (
    load_benchmark_models_config,
    load_include_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, default=_OUTPUT_ROOT / "cd")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    ensure_results()
    plan_bytes = BENCHMARK_PLAN.read_bytes()
    plan = load_benchmark(BENCHMARK_PLAN)
    datasets = plan["datasets"]
    if set(datasets) != set(load_include_datasets()):
        raise ValueError("The comparison plan and dataset scope must agree")
    bm = load_benchmark_models_config()
    print(f"Loading consolidated Main reports from {RESULTS}", flush=True)
    frame = build_results(datasets, bm)
    if set(frame.dataset) != set(datasets):
        raise ValueError("CD inputs do not cover every benchmark dataset")
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.plots_dir / "metric_inputs.csv", index=False)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_root": str(RESULTS),
        "benchmark_plan": str(BENCHMARK_PLAN),
        "benchmark_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "n_datasets": len(datasets),
        "datasets": sorted(datasets),
        "global_models": list(bm.expected_models_regression),
        "n_reports": len(frame),
        "selection_rule": "complete Tuned, otherwise complete Default, otherwise failure",
        "cv_aggregation": "all expected outer folds averaged before variant fallback",
        "normalization": "min-max over successful scores within each scope, best 1 and worst 0",
        "constant_row_policy": "all scores 0",
        "failure_score": 0,
        "tie_method": "average",
    }
    (args.plots_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    run_cd_pipeline(
        frame,
        bm,
        plots_dir=args.plots_dir,
        output_json=args.output_json or args.plots_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
