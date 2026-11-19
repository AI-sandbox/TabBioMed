"""Pretraining analysis."""

from __future__ import annotations
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, read_result
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.common import (
    load_benchmark_models_config,
    model_hpo_policy,
)
from tabbiomed.analysis.comparisons import compare
from tabbiomed.analysis.elo.missing import apply_missing_policy
from tabbiomed.analysis.elo.splitwise import battle_statistics, fit_elo
from tabbiomed.analysis.pretraining_overlap import pretraining_overlaps

OUTPUT = _OUTPUT_ROOT / "pretraining"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_table(ranks, datasets):
    table = ranks.loc[sorted(datasets)].mean().rename("mean_rank").to_frame()
    table.index.name = "model"
    table["rank_position"] = table.mean_rank.rank(method="average")
    return table.sort_values(["mean_rank", "model"])


def elo_tables(frame):
    names, _, design, labels, statistics = battle_statistics(frame)
    values = fit_elo(design, labels, statistics.sum(axis=0))
    values += 1000 - values[names.index("RandomForest::default")]
    entries = pd.DataFrame({"competitor": names, "elo": values}).set_index("competitor")
    entries["rank_position"] = entries.elo.rank(ascending=False, method="average")
    chosen = entries.reset_index()
    chosen[["model", "variant"]] = chosen.competitor.str.split("::", expand=True)
    chosen = chosen[
        chosen.variant.eq(
            np.where(chosen.model.map(model_hpo_policy(frame)), "tuned", "default")
        )
    ]
    models = chosen.set_index("model")[["variant", "elo"]].copy()
    models["rank_position"] = models.elo.rank(ascending=False, method="average")
    return (
        models.sort_values(["elo", "model"], ascending=[False, True]),
        entries.sort_values("elo", ascending=False),
    )


def verify_main_baselines(selected, ranks, availability):
    source = _OUTPUT_ROOT / "model_summary"
    previous = pd.read_csv(source / "selected_results.csv").set_index(
        ["model", "dataset"]
    )
    current = selected.set_index(["model", "dataset"])
    columns = [
        "task",
        "primary_metric",
        "metric_value",
        "metric_source",
        "selected_variant",
        "default_metric",
        "tuned_metric",
        "n_outer_splits",
        "n_default_splits",
        "n_tuned_splits",
    ]
    pd.testing.assert_frame_equal(
        previous[columns].sort_index(),
        current[columns].sort_index(),
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )
    expected = pd.read_csv(source / "summary.csv").set_index("model")
    np.testing.assert_allclose(
        ranks.mean().reindex(expected.index), expected.rank_mean, rtol=0, atol=1e-12
    )
    for policy in ("rf_imputed", "complete_case"):
        frame, _ = apply_missing_policy(availability, policy)
        _, table = elo_tables(frame)
        path = _OUTPUT_ROOT / "elo" / policy / "elo_all_entries.csv"
        old = pd.read_csv(path)
        old = old.set_index("competitor")
        np.testing.assert_allclose(
            table.elo.reindex(old.index), old.elo, rtol=0, atol=1e-07
        )


def latex_pvalue(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ensure_results()
    overlaps = pretraining_overlaps()
    plan = load_benchmark(BENCHMARK_PLAN)
    models = load_benchmark_models_config().expected_models_regression
    datasets = set(plan["datasets"])
    excluded = set(overlaps["all"]) & datasets
    if not excluded < datasets:
        raise ValueError("Overlap exclusion must leave datasets for comparison")
    jobs = [
        (m, d, info) for d, info in sorted(plan["datasets"].items()) for m in models
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = pd.DataFrame(pool.map(lambda job: read_result(*job), jobs))
    raw = selected.pivot(
        index="dataset", columns="model", values="metric_value"
    ).reindex(columns=models)
    tasks = selected.groupby("dataset").task.first()
    normalized = normalized_scores_with_failures(raw, tasks)
    ranks = normalized.rank(axis=1, ascending=False, method="average")
    rows = []
    for row in selected.itertuples():
        for variant in ("default", "tuned"):
            if variant == "tuned" and not row.hpo_enabled:
                continue
            rows.append(
                dict(
                    model=row.model,
                    hpo_enabled=row.hpo_enabled,
                    dataset=row.dataset,
                    task=row.task,
                    variant=variant,
                    competitor=f"{row.model}::{variant}",
                    fold=0,
                    metric_value=getattr(row, f"{variant}_metric"),
                    imputed=False,
                    n_outer_splits=row.n_outer_splits,
                    report_path=row.report_path,
                )
            )
    availability = pd.DataFrame(rows)
    comparisons = {}
    with threadpool_limits(limits=1):
        verify_main_baselines(selected, ranks, availability)
        args.output.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.output / "selected_results.csv", index=False)
        availability.to_csv(args.output / "availability.csv", index=False)
        normalized.to_csv(args.output / "normalized_scores.csv")
        ranks.to_csv(args.output / "dataset_ranks.csv")
        table_full = rank_table(ranks, datasets)
        table_reduced = rank_table(ranks, datasets - excluded)
        stats, delta = compare(table_full, table_reduced, "mean_rank", False)
        stats.update(
            n_datasets_full=len(datasets), n_datasets_retained=len(datasets - excluded)
        )
        comparisons["all_datasets_mean_rank"] = stats
        delta.to_csv(args.output / "all_datasets_mean_rank_comparison.csv")
        for policy in ("rf_imputed", "complete_case"):
            full, missing = apply_missing_policy(availability, policy)
            reduced = full[~full.dataset.isin(excluded)].copy()
            target = args.output / policy
            target.mkdir(exist_ok=True)
            full.to_csv(target / "metric_inputs_full.csv", index=False)
            reduced.to_csv(target / "metric_inputs_excluded.csv", index=False)
            a, aa = elo_tables(full)
            b, bb = elo_tables(reduced)
            policies = [
                ("elo_models", a, b, "elo", True),
                ("elo_configurations", aa, bb, "elo", True),
            ]
            if policy == "complete_case":
                policies.insert(
                    0,
                    (
                        "mean_rank",
                        rank_table(ranks, set(full.dataset)),
                        rank_table(ranks, set(reduced.dataset)),
                        "mean_rank",
                        False,
                    ),
                )
            comparisons[policy] = dict(
                n_datasets_full=full.dataset.nunique(),
                n_datasets_retained=reduced.dataset.nunique(),
                missing_policy=missing,
                n_imputed_retained=int(reduced.imputed.sum()),
            )
            for label, before, after, score, higher in policies:
                stats, delta = compare(before, after, score, higher)
                delta.to_csv(target / f"{label}_comparison.csv")
                comparisons[policy][label] = stats
                print(policy, label, json.dumps(stats), flush=True)
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        results_root=str(RESULTS),
        n_models=len(models),
        n_configurations=availability.competitor.nunique(),
        n_datasets=len(datasets),
        n_overlap_datasets=len(excluded),
        n_retained_datasets=len(datasets - excluded),
        iltm_overlaps=overlaps["iLTM"],
        excluded_datasets=sorted(excluded),
        overlaps=overlaps,
        input_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__).resolve().parents[1] / "pretraining_overlap.py",
                BENCHMARK_PLAN,
                Path(__file__).resolve().parents[1] / "studies/pretraining.py",
            ]
        },
        source_reports_verified_against_main=True,
        full_rank_and_elo_baselines_match_main=True,
        mean_rank_policy="Main CD policy: complete Tuned else complete Default else failure=0 tied with worst success, all datasets retained, average ties.",
        elo_policy="Main Elo policy: complete CV means, separate Default/Tuned in a joint configuration pool, RF-default imputation or complete-case, half ties, default RF anchored to 1000.",
        comparisons=comparisons,
    )
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    overlap_rows = [
        {"dataset": d, **{m: d in v for m, v in overlaps.items() if m != "all"}}
        for d in sorted(excluded)
    ]
    pd.DataFrame(overlap_rows).to_csv(args.output / "overlap.csv", index=False)
    latex = []
    specifications = [
        (
            "All datasets",
            f"Mean rank ({len(models)} models)",
            comparisons["all_datasets_mean_rank"],
            comparisons["all_datasets_mean_rank"],
        )
    ]
    for policy, label in [
        ("rf_imputed", "RF-imputed"),
        ("complete_case", "Complete-case"),
    ]:
        scope = comparisons[policy]
        for key, title in [
            ("mean_rank", f"Mean rank ({len(models)} models)"),
            ("elo_models", f"Elo ({len(models)} models)"),
            (
                "elo_configurations",
                f"Elo ({availability.competitor.nunique()} configurations)",
            ),
        ]:
            if key in scope:
                specifications.append((label, title, scope, scope[key]))
    for label, title, scope, stats in specifications:
        latex.append(
            f"{label} & {title} & ${scope['n_datasets_full']}/{scope['n_datasets_retained']}$ & {stats['kendall_tau']:.3f} & {latex_pvalue(stats['kendall_pvalue'])} & ${stats['top5_overlap']}/5$ & {stats['max_abs_rank_shift']:g} "
            + "\\\\"
        )
    (args.output / "sensitivity_rows.tex").write_text("\n".join(latex) + "\n")
    print(json.dumps(comparisons["all_datasets_mean_rank"], indent=2), flush=True)


if __name__ == "__main__":
    main()
