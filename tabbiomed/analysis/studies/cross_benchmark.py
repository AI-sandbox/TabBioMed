"""Cross benchmark analysis."""

from __future__ import annotations
from tabbiomed.analysis.paths import CONFIG, PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.common import load_benchmark_models_config
from tabbiomed.analysis.figures.elo_regimes import collect_model_inputs
from tabbiomed.analysis.cross_benchmark import (
    DEFAULT_ALIASES,
    align_shared_models,
    load_aliases,
    load_tabarena_ranks,
    rank_correlations,
    tabbiomed_mean_ranks,
    top_k_overlap,
)

OUTPUT = _OUTPUT_ROOT / "cross_benchmark"
TABLE2 = _OUTPUT_ROOT / "model_summary"
TABARENA_REFERENCE = CONFIG / "tabarena_verified_20261108.csv"
SCOPES = {
    "overall": None,
    "binary": "binary",
    "multi": "classification",
    "regr": "regression",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_rows(inputs, models):
    if inputs.duplicated(["dataset", "model"]).any():
        raise ValueError("Duplicate model/dataset rows")
    if any((set(group.model) != set(models) for _, group in inputs.groupby("dataset"))):
        raise ValueError("Incomplete model roster")
    if any((group.task.nunique() != 1 for _, group in inputs.groupby("dataset"))):
        raise ValueError("Inconsistent dataset task metadata")
    raw = inputs.pivot(index="dataset", columns="model", values="metric_value").reindex(
        columns=models
    )
    tasks = inputs.groupby("dataset").task.first()
    normalized = normalized_scores_with_failures(raw, tasks)
    ranks = normalized.rank(axis=1, method="average", ascending=False)
    rows = inputs.merge(
        normalized.stack().rename("norm_metric").reset_index(),
        on=["dataset", "model"],
        validate="one_to_one",
    )
    rows = rows.merge(
        ranks.stack().rename("rank").reset_index(),
        on=["dataset", "model"],
        validate="one_to_one",
    )
    return (rows, normalized, ranks)


def compare_scopes(rows, arena, aliases):
    aligned = {}
    summary = []
    for scope, task in SCOPES.items():
        subset = rows if task is None else rows[rows.task.eq(task)]
        if subset.empty:
            continue
        means = tabbiomed_mean_ranks(subset)
        table = align_shared_models(
            aliases=aliases,
            tabarena=arena,
            tabbiomed=means,
            tabarena_metric=scope,
            drop_notes=True,
        )
        aligned[scope] = table
        summary.append(
            dict(
                scope=scope,
                task=task or "all",
                n_datasets=subset.dataset.nunique(),
                **rank_correlations(table),
                top_k=top_k_overlap(table, 5),
            )
        )
    return (aligned, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabarena-ranks", type=Path, default=TABARENA_REFERENCE)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--table2-dir", type=Path, default=TABLE2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ensure_results()
    plan = load_benchmark(BENCHMARK_PLAN)
    config = load_benchmark_models_config()
    models = sorted(config.expected_models_regression)
    inputs, metadata = collect_model_inputs(
        plan["datasets"], models, RESULTS, PROCESSED
    )
    inputs["model_family"] = inputs.model.map(config.model_family)
    rows, normalized, ranks = rank_rows(inputs, models)
    reference = pd.read_csv(
        args.table2_dir / "ranks.csv", index_col=0, float_precision="round_trip"
    )
    if set(reference.index) != set(ranks.index) or set(reference.columns) != set(
        models
    ):
        raise ValueError("Table 2 has a different dataset/model scope")
    np.testing.assert_array_equal(
        ranks.to_numpy(),
        reference.reindex(index=ranks.index, columns=models).to_numpy(),
    )
    aliases = load_aliases(args.aliases)
    arena = load_tabarena_ranks(args.tabarena_ranks)
    aligned, summary = compare_scopes(rows, arena, aliases)
    shared = set(aligned["overall"].model)
    if not shared or any(set(table.model) != shared for table in aligned.values()):
        raise ValueError("Comparison scopes must contain the same shared model versions")
    for record in inputs.itertuples():
        if digest(record.report_path) != record.report_sha256:
            raise ValueError(
                f"Main report changed during analysis: {record.report_path}"
            )
    for record in metadata.itertuples():
        if digest(record.metadata_path) != record.metadata_sha256:
            raise ValueError(
                f"Dataset metadata changed during analysis: {record.metadata_path}"
            )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "tabbiomed_rank_rows.csv", index=False)
    normalized.to_csv(out / "normalized_scores.csv")
    ranks.to_csv(out / "ranks.csv")
    metadata.to_csv(out / "datasets.csv", index=False)
    tabbiomed_mean_ranks(rows).to_csv(out / "tabbiomed_mean_ranks.csv", index=False)
    for scope, table in aligned.items():
        directory = out / scope
        directory.mkdir(exist_ok=True)
        table.to_csv(directory / "aligned_ranks.csv", index=False)
    (out / "agreement.json").write_text(json.dumps(summary, indent=2) + "\n")
    table_lines = []
    for r in aligned["overall"].itertuples():
        delta = f"{r.delta_rank:+.0f}" if r.delta_rank else "0"
        table_lines.append(
            f"{r.model} & {r.tabarena_rank:.0f} & {r.tabarena_avg_rank:.2f} & {r.tabbiomed_rank:.0f} & {r.tabbiomed_mean_rank:.2f} & {delta} "
            + "\\\\"
        )
    (out / "model_comparison_rows.tex").write_text("\n".join(table_lines) + "\n")
    task_lines = []
    labels = dict(
        overall="Overall",
        binary="Binary classification",
        multi="Multiclass classification",
        regr="Regression",
    )
    for r in summary:
        task_lines.append(
            f"{labels[r['scope']]} & {r['n_datasets']} & {r['kendall_tau']:.3f} & {r['spearman_rho']:.3f} & {r['top_k']['overlap']}/5 "
            + "\\\\"
        )
    (out / "task_agreement_rows.tex").write_text("\n".join(task_lines) + "\n")
    sources = [
        BENCHMARK_PLAN,
        args.tabarena_ranks,
        args.aliases,
        args.table2_dir / "ranks.csv",
        Path(__file__).resolve().parents[1] / "studies/cross_benchmark.py",
        Path(__file__).resolve().parents[1] / "cross_benchmark.py",
        Path(__file__).resolve().parents[1] / "results.py",
        Path(__file__).resolve().parents[1] / "ranks.py",
    ]
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        results_root=str(RESULTS.resolve()),
        models=models,
        n_datasets=len(plan["datasets"]),
        n_shared_models=len(shared),
        table2_ranks_identical=True,
        selection_counts=inputs.metric_source.value_counts().to_dict(),
        policy="Complete Tuned mean, then own complete Default mean. Min-max normalization over the configured models. Failures score zero and tie with the worst success. Shared-model filtering follows per-dataset ranking.",
        tabarena_source=str(args.tabarena_ranks.resolve()),
        version_mismatch_exclusions=[
            m for m, spec in aliases.items() if spec.get("note")
        ],
        source_hashes={str(p.resolve()): digest(p) for p in sources},
        outputs={
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "analysis.json"
        },
    )
    (out / "analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        aligned["overall"][
            ["model", "tabarena_rank", "tabbiomed_rank", "tabbiomed_mean_rank"]
        ].to_string(index=False)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
