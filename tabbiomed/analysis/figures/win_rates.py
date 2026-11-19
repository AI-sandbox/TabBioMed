"""Win rates figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import MODEL_FAMILIES, PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-domain-elo-mpl")
import numpy as np
import pandas as pd
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS
from tabbiomed.analysis.common import load_benchmark_models_config
from tabbiomed.analysis.figures.elo_regimes import (
    ALL_TASKS,
    CLASSIFICATION_ONLY_MODELS,
    CLASSIFICATION_TASKS,
    collect_model_inputs,
    digest,
)
from tabbiomed.analysis.elo.splitwise import _orient_scores
from tabbiomed.analysis.heatmap import plot_heatmap

FAMILIES = MODEL_FAMILIES
OUTPUT = _OUTPUT_ROOT / "win_rates"


def prepare_supported_inputs(raw, models):
    frame = raw.copy()
    if frame.empty or frame.duplicated(["dataset", "model"]).any():
        raise ValueError("Empty or duplicate model/dataset inputs")
    if set(frame.model) != set(models) or not frame.task.isin(ALL_TASKS).all():
        raise ValueError("Unexpected model roster or task type")
    if (
        frame.model.isin(CLASSIFICATION_ONLY_MODELS)
        & ~frame.task.isin(CLASSIFICATION_TASKS)
    ).any():
        raise ValueError(
            "Classification-only models cannot contribute regression scores"
        )
    for _, group in frame.groupby("dataset"):
        expected = set(models)
        if group.task.iloc[0] not in CLASSIFICATION_TASKS:
            expected -= CLASSIFICATION_ONLY_MODELS
        if group.task.nunique() != 1 or set(group.model) != expected:
            raise ValueError("Incomplete model roster or inconsistent task metadata")
    frame["metric_value"] = pd.to_numeric(frame.raw_metric)
    if np.isinf(frame.metric_value).any():
        raise ValueError("Infinite test metric")
    frame["succeeded"] = frame.metric_value.notna()
    if not frame.succeeded.eq(frame.selected_variant.ne("failed")).all():
        raise ValueError("Test availability and selected variant disagree")
    if frame.metric_source.eq("rf_default_imputed").any():
        raise ValueError("RF imputation is not used in the pairwise win-rate matrix")
    return frame.sort_values(["dataset", "model"]).reset_index(drop=True)


def pairwise_matrix(inputs, metadata, models):
    datasets = metadata.set_index("dataset").sort_index()
    if not datasets.index.is_unique or set(inputs.dataset) != set(datasets.index):
        raise ValueError("Pairwise inputs and dataset metadata disagree")
    if inputs.duplicated(["dataset", "model"]).any():
        raise ValueError("Duplicate model/dataset scores")
    if set(inputs.model) != set(models):
        raise ValueError("Pairwise model roster differs from the input roster")
    if (
        not inputs.task.eq(inputs.dataset.map(datasets.task)).all()
        or not datasets.task.isin(ALL_TASKS).all()
    ):
        raise ValueError("Pairwise input task types disagree with dataset metadata")
    if np.isinf(inputs.metric_value).any():
        raise ValueError("Infinite test metric")
    oriented = inputs.assign(score=_orient_scores(inputs))
    scores = oriented.pivot(index="dataset", columns="model", values="score").reindex(
        index=datasets.index, columns=models
    )
    present = inputs.assign(present=True).pivot(
        index="dataset", columns="model", values="present"
    )
    present = present.reindex(index=datasets.index, columns=models).notna()
    eligible = pd.DataFrame(True, index=datasets.index, columns=models)
    for model in models:
        if model in CLASSIFICATION_ONLY_MODELS:
            eligible[model] = datasets.task.isin(CLASSIFICATION_TASKS)
    if (present & ~eligible).any().any():
        raise ValueError("An unsupported evaluation was included")
    if (eligible & ~present).any().any():
        raise ValueError("A supported evaluation record is missing")
    matrix = pd.DataFrame(np.nan, index=models, columns=models)
    counts = pd.DataFrame(0, index=models, columns=models)
    rows = []
    for a, b in combinations(models, 2):
        keep = eligible[a] & eligible[b]
        n = int(keep.sum())
        if not n:
            raise ValueError(f"No jointly supported tasks for {a} and {b}")
        left, right = (scores.loc[keep, a], scores.loc[keep, b])
        success_a, success_b = (left.notna(), right.notna())
        both = success_a & success_b
        only_a, only_b = (success_a & ~success_b, success_b & ~success_a)
        score_wins_a, score_wins_b = (
            int((both & (left > right)).sum()),
            int((both & (left < right)).sum()),
        )
        observed_ties = int((both & (left == right)).sum())
        both_failed = int((~success_a & ~success_b).sum())
        wins_a, wins_b = (
            score_wins_a + int(only_a.sum()),
            score_wins_b + int(only_b.sum()),
        )
        ties = observed_ties + both_failed
        assert wins_a + wins_b + ties == n
        for (
            first,
            second,
            wins,
            losses,
            score_wins,
            completion_wins,
            success_first,
            success_second,
        ) in [
            (
                a,
                b,
                wins_a,
                wins_b,
                score_wins_a,
                int(only_a.sum()),
                success_a,
                success_b,
            ),
            (
                b,
                a,
                wins_b,
                wins_a,
                score_wins_b,
                int(only_b.sum()),
                success_b,
                success_a,
            ),
        ]:
            rate = 100 * (wins + 0.5 * ties) / n
            matrix.loc[first, second] = rate
            counts.loc[first, second] = n
            rows.append(
                dict(
                    model_a=first,
                    model_b=second,
                    n_common=n,
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    win_rate_pct=rate,
                    wins_on_score=score_wins,
                    wins_by_completion=completion_wins,
                    n_failed_a=int((~success_first).sum()),
                    n_failed_b=int((~success_second).sum()),
                    n_both_observed=int(both.sum()),
                    n_observed_ties=observed_ties,
                    n_both_failed=both_failed,
                )
            )
    matrix.index.name = counts.index.name = "model"
    return (matrix, counts, pd.DataFrame(rows), scores)


def render(matrix, manifest, output):
    if matrix.shape[0] != matrix.shape[1] or set(matrix.index) != set(matrix.columns):
        raise ValueError("Expected a square win-rate matrix with matching model labels")
    mean_win_rates = matrix.mean(axis=1)
    order = sorted(matrix.index, key=lambda m: (-mean_win_rates[m], m.casefold()))
    displayed = matrix.reindex(index=order, columns=order)
    path = output / f"win_rate_{len(matrix)}.pdf"
    plot_heatmap(
        order,
        displayed.to_numpy().tolist(),
        path,
        sort_by_win_rate=False,
        formats=("pdf",),
        reverse_columns=False,
        row_label="Row model",
        column_label="Column model",
        colorbar_label="Row model win rate (%)",
    )
    manifest["policy"].update(
        ordering=f"Descending mean pairwise win rate across the other {len(matrix)-1} models, equal weight per opponent, alphabetical ties",
        displayed_row_order=order,
        displayed_column_order=order,
        cell_direction="Row model's win rate against the column model",
    )
    manifest["mean_pairwise_win_rate_pct"] = {
        m: float(mean_win_rates[m]) for m in order
    }
    manifest.update(
        figure=str(path),
        figure_sha256=digest(path),
        renderer_sha256=digest(Path(__file__).resolve().parents[1] / "heatmap.py"),
        render_script_sha256=digest(
            Path(__file__).resolve().parents[1] / "figures/win_rates.py"
        ),
        rendered_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    (output / "analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    ensure_results()
    if args.render_only:
        matrix = pd.read_csv(
            args.output_dir / "win_rate_matrix.csv",
            index_col=0,
            float_precision="round_trip",
        )
        manifest = json.loads((args.output_dir / "analysis.json").read_text())
        render(matrix, manifest, args.output_dir)
        return
    plan = load_benchmark(BENCHMARK_PLAN)
    config = load_benchmark_models_config(FAMILIES)
    models = sorted(config.expected_models_regression)
    expected = (
        set(config.expected_models_binary_classification) - CLASSIFICATION_ONLY_MODELS
    )
    if set(models) != expected:
        raise ValueError("Win-rate inputs differ from the configured eligible models")
    raw, metadata = collect_model_inputs(plan["datasets"], models, RESULTS, PROCESSED)
    inputs = prepare_supported_inputs(raw, models)
    if len(inputs) != len(models) * len(plan["datasets"]) or set(inputs.dataset) != set(
        plan["datasets"]
    ):
        raise ValueError("Unexpected supported model/task grid")
    matrix, counts, pairs, scores = pairwise_matrix(inputs, metadata, models)
    if pairs.n_common.value_counts().to_dict() != {
        len(plan["datasets"]): len(models) * (len(models) - 1)
    }:
        raise ValueError("Unexpected pairwise task denominators")
    for row in raw.itertuples():
        if digest(Path(row.report_path)) != row.report_sha256:
            raise ValueError(
                f"Main report changed during the analysis: {row.report_path}"
            )
    for row in metadata.itertuples():
        if digest(Path(row.metadata_path)) != row.metadata_sha256:
            raise ValueError(
                f"Dataset metadata changed during the analysis: {row.metadata_path}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame, index in [
        ("inputs.csv", inputs, False),
        ("datasets.csv", metadata, False),
        ("win_rate_matrix.csv", matrix, True),
        ("pair_counts.csv", counts, True),
        ("pairwise_results.csv", pairs, False),
        ("oriented_scores.csv", scores, True),
    ]:
        frame.to_csv(args.output_dir / name, index=index)
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        results_root=str(RESULTS.resolve()),
        models=models,
        n_models=len(models),
        n_datasets=len(plan["datasets"]),
        n_supported_model_dataset_pairs=len(inputs),
        tasks={str(k): int(v) for k, v in metadata.task.value_counts().items()},
        selection_counts={
            str(k): int(v) for k, v in inputs.metric_source.value_counts().items()
        },
        pair_denominators={
            str(k): int(v) for k, v in pairs.n_common.value_counts().items()
        },
        policy=dict(
            eligibility="All configured datasets for every pair of task-eligible models",
            score_selection="Complete Tuned mean, then own complete Default mean, otherwise failure",
            cv="All outer folds required within each variant, averaged before score selection",
            failures="A successful result beats a failure, two failures receive half credit each, no imputation",
            unsupported="Excluded from pair eligibility",
            ties="Exact primary-metric ties and two failures receive half a win",
            weighting="Each jointly supported dataset has equal weight",
        ),
        source_hashes={
            str(p): digest(p)
            for p in [
                BENCHMARK_PLAN,
                FAMILIES,
                Path(__file__).resolve().parents[1] / "figures/win_rates.py",
                Path(__file__).resolve().parents[1] / "figures/elo_regimes.py",
                Path(__file__).resolve().parents[1] / "results.py",
            ]
        },
        outputs={p.name: digest(p) for p in args.output_dir.glob("*.csv")},
        source_reports_verified_unchanged=True,
    )
    render(matrix, manifest, args.output_dir)
    print(json.dumps(manifest["selection_counts"]), flush=True)
    print("Pair denominators:", manifest["pair_denominators"], flush=True)
    print("Wrote", args.output_dir / "win_rate_21.pdf", flush=True)


if __name__ == "__main__":
    main()
