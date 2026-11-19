"""Pareto figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import CONFIG, PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-main-pareto-mpl")
import numpy as np
import pandas as pd
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, valid_split_identity
from tabbiomed.analysis.common import (
    analysis_dataset_metadata,
    canon_domain,
    DEFAULT_MODEL_FAMILIES,
    load_benchmark_models_config,
)
from tabbiomed.analysis.pareto_style import Panel, Style, plot_pareto_pair
from tabbiomed.analysis.pareto_math import pareto_mask_min_x_max_y
from tabbiomed.analysis.elo.splitwise import battle_statistics, fit_elo

OUTPUT = _OUTPUT_ROOT / "pareto"
CLOCKS = {
    "train": (
        "train_per_1k",
        "x_median_train_per_1k",
        "Training cost per 1K training rows (s) (median)",
    ),
    "inference": (
        "inference_per_1k",
        "x_median_inference_per_1k",
        "Inference time per 1K test rows (s) (median)",
    ),
}
DOMAINS = {
    "Clinical & Healthcare": "ehr",
    "Drug Discovery": "drug",
    "Genomics": "genom",
    "Metabolomics": "mtbls",
    "Proteomics": "ptomics",
    "Single-cell": "sc",
    "Systems Biology": "sb",
    "Transcriptomics": "trans",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_numeric_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def expected_competitors(models, default_only):
    return sorted(
        (
            f"{model}::{variant}"
            for model in models
            for variant in (
                ["default"] if model in default_only else ["default", "tuned"]
            )
        )
    )


def validate_inputs(metrics, elo, datasets, models, default_only):
    competitors = expected_competitors(models, default_only)
    if (
        metrics.duplicated(["dataset", "competitor"]).any()
        or elo.competitor.duplicated().any()
    ):
        raise ValueError("Duplicate main leaderboard entries")
    if set(elo.competitor) != set(competitors) or not np.isfinite(elo.elo).all():
        raise ValueError("Elo competitors differ from the expected main model roster")
    if set(metrics.dataset) != set(datasets):
        raise ValueError("Main leaderboard datasets differ from the benchmark")
    for dataset, frame in metrics.groupby("dataset"):
        if set(frame.competitor) != set(competitors) or len(frame) != len(competitors):
            raise ValueError(f"Incomplete main leaderboard grid: {dataset}")
        if not np.isfinite(frame.metric_value).all():
            raise ValueError("Main RF-imputed scores must all be finite")
        rf = frame[frame.competitor == "RandomForest::default"].iloc[0]
        if bool(rf.imputed):
            raise ValueError("RandomForest Default reference cannot be imputed")
        if not np.allclose(
            frame.loc[frame.imputed, "metric_value"], rf.metric_value, rtol=0, atol=0
        ):
            raise ValueError(
                "A missing score does not inherit its dataset's RF Default score"
            )
    anchor = elo.set_index("competitor").loc["RandomForest::default", "elo"]
    if not np.isclose(anchor, 1000, rtol=0, atol=1e-08):
        raise ValueError("Main leaderboard RF Default anchor must be 1000")


def unit_timing_rates(unit, variant):
    block = unit[variant]

    def number(value):
        return (
            float(value)
            if value is not None and np.isfinite(value) and (value >= 0)
            else np.nan
        )

    train = number(block["training_cost_seconds"])
    inference = number(block["test_prediction_seconds"])
    n_train, n_test = (
        unit["outer_training_pool_samples"],
        unit["split_samples"]["test"],
    )
    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            "Timing normalization requires nonempty training and test partitions"
        )
    return dict(
        train_per_1k=train * 1000 / n_train,
        inference_per_1k=inference * 1000 / n_test,
        train_seconds=train,
        inference_seconds=inference,
        n_training_pool=n_train,
        n_test=n_test,
        final_fit_seconds=block.get("final_fit_seconds"),
        n_fitted=block.get("final_fit_samples"),
        final_validation_prediction_seconds=block.get(
            "final_validation_prediction_seconds"
        ),
        hpo_search_seconds=(
            unit.get("hpo", {}).get("official_search_wall_seconds")
            if variant == "tuned"
            else 0.0
        ),
        search_clock_status=(
            unit.get("hpo", {}).get("search_clock_status")
            if variant == "tuned"
            else "not_applicable"
        ),
    )


def collect_timings(metrics, selections, plan):
    matrix = metrics.set_index(["model", "dataset", "variant"]).sort_index()

    def read(selection):
        model, dataset = (selection["model"], selection["dataset"])
        path = RESULTS / model / dataset / "report.json"
        if sha(path) != selection["report_sha256"]:
            raise ValueError(f"Main report changed since the Elo analysis: {path}")
        report = json.loads(path.read_text())
        info = plan["datasets"][dataset]
        if (
            report.get("artifact_type") != "evaluation_summary"
            or report.get("schema_version") != 1
            or report.get("model") != model
            or (report.get("dataset") != dataset)
        ):
            raise ValueError(f"Invalid consolidated Main report: {path}")
        report_units = (
            {f["fold_id"]: f for f in report.get("cv_fold_metrics", [])}
            if info["n_folds"]
            else {0: report}
        )
        timing_path = path.with_name("timing.json")
        timing = json.loads(timing_path.read_text())
        if (
            timing.get("artifact_type") != "evaluation_timing_summary"
            or timing.get("schema_version") != 1
            or timing.get("scope") != "main"
            or (timing.get("model") != model)
            or (timing.get("dataset") != dataset)
            or (timing.get("report_sha256") != selection["report_sha256"])
            or (not valid_split_identity(timing, info))
        ):
            raise ValueError(
                f"Timing summary does not match its Main evaluation: {timing_path}"
            )
        timing_units = {
            u["fold_id"] if info["n_folds"] else 0: u for u in timing["units"]
        }
        expected = set(range(info["n_folds"] or 1))
        if set(timing_units) != expected or set(report_units) != expected:
            raise ValueError(
                f"Timing and report folds differ from the benchmark: {path}"
            )
        return (
            (model, dataset),
            dict(
                report_units=report_units,
                timing_units=timing_units,
                report_path=path,
                timing_path=timing_path,
                timing_sha256=sha(timing_path),
                report_sha256=selection["report_sha256"],
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        loaded = dict(pool.map(read, selections))
    collected = []
    for selection in selections:
        model, dataset = (selection["model"], selection["dataset"])
        original = loaded[model, dataset]
        results, fold_rows = ([], [])
        for variant in ["default", "tuned"]:
            if (model, dataset, variant) not in matrix.index:
                continue
            cell = matrix.loc[model, dataset, variant]
            row = dict(
                model=model,
                dataset=dataset,
                variant=variant,
                competitor=cell.competitor,
                metric_value=cell.metric_value,
                imputed=bool(cell.imputed),
                timing_imputed=bool(cell.imputed),
                train_per_1k=np.nan,
                inference_per_1k=np.nan,
                report_path=str(original["report_path"]),
                report_sha256=selection["report_sha256"],
            )
            source_model, source_variant = (
                ("RandomForest", "default") if row["imputed"] else (model, variant)
            )
            source = loaded[source_model, dataset]
            row.update(
                timing_source_model=source_model,
                timing_source_variant=source_variant,
                timing_path=str(source["timing_path"]),
                timing_sha256=source["timing_sha256"],
            )
            measured = []
            key = "default" if source_variant == "default" else "best"
            for fold, unit in sorted(source["timing_units"].items()):
                block = (source["report_units"][fold] or {}).get(key) or {}
                if (
                    not block.get("fixed_split_test_evaluated")
                    or block.get("failure")
                    or block.get("error")
                    or block.get("post_hpo_test_error")
                    or (unit[source_variant]["evaluation_status"] != "success")
                ):
                    raise ValueError(
                        f"Leaderboard timing source has no successful test fit: {source['report_path']}/{source_variant}/{fold}"
                    )
                values = unit_timing_rates(unit, source_variant)
                measured.append(values)
                fold_rows.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        variant=variant,
                        fold=fold,
                        timing_imputed=row["timing_imputed"],
                        timing_source_model=source_model,
                        timing_source_variant=source_variant,
                        **values,
                    )
                )
            row.update(
                n_outer_folds=len(measured),
                n_train_timed_folds=sum(
                    (np.isfinite(r["train_per_1k"]) for r in measured)
                ),
                n_inference_timed_folds=sum(
                    (np.isfinite(r["inference_per_1k"]) for r in measured)
                ),
            )
            for field, _, _ in CLOCKS.values():
                values = [r[field] for r in measured]
                if np.isfinite(values).all():
                    row[field] = float(np.mean(values))
            results.append(row)
        collected.append((results, fold_rows))
    return (
        pd.DataFrame([r for rows, _ in collected for r in rows]),
        pd.DataFrame([r for _, rows in collected for r in rows]),
    )


def summarize_costs(timings, elo, field, median_field):
    timings = timings.copy()
    if "timing_imputed" not in timings:
        if timings.imputed.any():
            raise ValueError("Fallback timing requires explicit source provenance")
        timings["timing_imputed"] = False
    observed = timings[np.isfinite(timings[field]) & (timings[field] >= 0)]
    summary = observed.groupby("competitor", as_index=False).agg(
        model=("model", "first"),
        variant=("variant", "first"),
        **{median_field: (field, "median")},
        n_datasets=("dataset", "nunique"),
        n_fallback_datasets=("timing_imputed", "sum"),
    )
    summary = summary.merge(
        elo[["competitor", "elo"]], on="competitor", validate="one_to_one"
    )
    if (summary[median_field] <= 0).any():
        raise ValueError("Median cost must be positive on a log axis")
    summary = summary.sort_values(
        [median_field, "elo"], ascending=[True, False]
    ).reset_index(drop=True)
    summary["is_pareto"] = pareto_mask_min_x_max_y(
        summary[median_field].to_numpy(), summary.elo.to_numpy()
    )
    return summary


def fit_scope_elo(metrics):
    if metrics.groupby("dataset").fold.nunique().ne(1).any():
        raise ValueError("CV must be averaged before fitting Pareto Elo")
    names, _, design, labels, statistics = battle_statistics(metrics)
    ratings = fit_elo(design, labels, statistics.sum(axis=0))
    ratings += 1000 - ratings[names.index("RandomForest::default")]
    return pd.DataFrame({"competitor": names, "elo": ratings})


def scope_datasets(plan, *, include_regimes=True):
    domains = {name: [] for name in DOMAINS}
    for dataset, info in sorted(plan["datasets"].items()):
        metadata = analysis_dataset_metadata(dataset, info, PROCESSED)
        name = canon_domain(metadata.get("label"))
        if name not in domains:
            raise ValueError(f"Unknown dataset domain: {name}")
        domains[name].append(dataset)
    scopes = [
        (f"domain:{slug}", name, datasets, "per_domain", f"pareto_scatter_{slug}")
        for name, slug in DOMAINS.items()
        for datasets in [domains[name]]
    ]
    if not include_regimes:
        return scopes
    path = _OUTPUT_ROOT / "elo_regimes" / "all_tasks_datasets.csv"
    regimes = read_numeric_csv(path)
    if regimes.dataset.duplicated().any() or set(regimes.dataset) != set(
        plan["datasets"]
    ):
        raise ValueError("Published regime assignments differ from the benchmark")
    for label, group in regimes.groupby("quadrant", sort=True):
        code = label.split()[0].lower()
        scopes.append(
            (
                f"regime:{code}",
                label,
                sorted(group.dataset),
                "per_regime",
                f"pareto_scatter_{code}",
            )
        )
    return scopes


def make_summaries(timings, elo):
    summaries = {}
    for clock, (field, median, _) in CLOCKS.items():
        summaries[clock] = summarize_costs(timings, elo, field, median)
        if set(summaries[clock].competitor) != set(elo.competitor):
            raise ValueError(f"A configuration has no successful {clock} timing")
    return summaries


def render(
    summaries,
    config,
    output,
    stem="pareto_scatter",
    *,
    colors_path=CONFIG / "pareto_colors.json",
):
    models = config.expected_models_regression
    colors = json.loads(colors_path.read_text())
    families = ["Baselines", "Tree-based", "Neural networks", "Foundation models"]
    groups = [
        (
            family,
            [m for m, f in config.model_family.items() if f == family and m in models],
        )
        for family in families
    ]
    panels = [
        Panel(summaries[clock], median, label)
        for clock, (_, median, label) in CLOCKS.items()
    ]
    style = Style(
        arrow_kind="block",
        arrow_color="#1565c0",
        arrow_head_w=3.5,
        edge_front=1.3,
        leader_cost=1500,
        legend_gap=0.012,
        legend_colspacing=0.7,
        legend_fit="axes",
        font_serif=("Times New Roman", "Times", "Liberation Serif"),
        center_frames=True,
        legend=21,
        legend_title=22,
        variant_legend=27,
        label=19.3,
        label_colors={"ElasticNet": "#af82cf", "TabICLv2": "#3db515"},
    )
    plot_pareto_pair(
        panels,
        output / f"{stem}.png",
        y_col="elo",
        y_label="Elo",
        metric="elo",
        mode="labels",
        style=style,
        colors=colors,
        legend_groups=groups,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--plan", type=Path, default=BENCHMARK_PLAN)
    parser.add_argument("--model-families", type=Path, default=DEFAULT_MODEL_FAMILIES)
    parser.add_argument("--colors", type=Path, default=CONFIG / "pareto_colors.json")
    parser.add_argument("--elo-dir", type=Path, default=_OUTPUT_ROOT / "elo")
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS.values()))
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["main", "domains", "regimes"],
        default=["main", "domains", "regimes"],
    )
    args = parser.parse_args()
    ensure_results()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_benchmark_models_config(args.model_families, plan_path=args.plan)
    models = config.expected_models_regression
    plan = load_benchmark(args.plan)
    if len(models) < 2 or not plan["datasets"]:
        raise ValueError("Expected at least two models and a nonempty dataset scope")
    default_only = {
        m for m, settings in plan["models"].items() if not settings["hpo_enabled"]
    }
    elo_path = args.elo_dir / "rf_imputed/elo_all_entries.csv"
    metrics_path = args.elo_dir / "rf_imputed/metric_inputs.csv"
    selection_path = args.elo_dir / "selected_reports.json"
    elo, metrics = (
        read_numeric_csv(elo_path),
        read_numeric_csv(metrics_path),
    )
    validate_inputs(metrics, elo, plan["datasets"], models, default_only)
    recomputed = fit_scope_elo(metrics).set_index("competitor").elo.sort_index()
    reference = elo.set_index("competitor").elo.sort_index()
    difference = float(np.max(np.abs(recomputed - reference)))
    if difference > 1e-08:
        raise ValueError(
            f"Subset solver differs from main Elo policy: maximum difference {difference}"
        )
    selections = json.loads(selection_path.read_text())
    if {(r["model"], r["dataset"]) for r in selections} != {
        (m, d) for m in models for d in plan["datasets"]
    }:
        raise ValueError("Report selections differ from the main scope")
    timings, folds = collect_timings(metrics, selections, plan)
    timings.to_csv(args.output / "timing_by_dataset.csv", index=False)
    folds.to_csv(args.output / "timing_by_split.csv", index=False)
    elo.to_csv(args.output / "main_elo_ratings.csv", index=False)
    scoped_manifests = []
    if "main" in args.scopes:
        summaries = make_summaries(timings, elo)
        for clock in CLOCKS:
            summaries[clock].to_csv(args.output / f"{clock}_summary.csv", index=False)
        render(summaries, config, args.output, colors_path=args.colors)
    for key, label, datasets, subdirectory, stem in scope_datasets(
        plan, include_regimes="regimes" in args.scopes
    ):
        if (
            key.startswith("domain:")
            and args.domains is not None
            and key.split(":")[1] not in args.domains
        ):
            continue
        if (
            key.startswith("domain:")
            and "domains" not in args.scopes
            or (key.startswith("regime:") and "regimes" not in args.scopes)
        ):
            continue
        scope = metrics[metrics.dataset.isin(datasets)].copy()
        ratings = fit_scope_elo(scope)
        validate_inputs(scope, ratings, datasets, models, default_only)
        summaries = make_summaries(timings[timings.dataset.isin(datasets)], ratings)
        destination = args.output / subdirectory
        data = destination / "data"
        data.mkdir(parents=True, exist_ok=True)
        ratings.to_csv(data / f"{stem}_elo.csv", index=False)
        for clock in CLOCKS:
            summaries[clock].to_csv(data / f"{stem}_{clock}_summary.csv", index=False)
        print(
            key,
            len(datasets),
            f"datasets, {len(models)} models, {len(ratings)} configurations,",
            int(scope.imputed.sum()),
            "RF fills",
            flush=True,
        )
        render(summaries, config, destination, stem, colors_path=args.colors)
        record = dict(
            scope=key,
            label=label,
            datasets=datasets,
            n_datasets=len(datasets),
            n_models=len(models),
            n_competitors=len(elo),
            imputed_cells=int(scope.imputed.sum()),
            elo_policy="Domain/regime subset of main RF-imputed grid, all compared configurations retained, CV means, half-win ties, Default RF anchored to 1000",
            figure_sha256=sha(destination / f"{stem}.pdf"),
        )
        (data / f"{stem}_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        scoped_manifests.append(record)
    manifest = dict(
        results_root=str(RESULTS),
        n_datasets=len(plan["datasets"]),
        n_models=len(models),
        n_competitors=len(elo),
        score_policy="Reuse main RF-imputed Elo, Default RF raw primary score for each missing dataset/configuration, complete outer-fold means, half-win ties",
        timing_policy="Default: one final fit plus its validation prediction. Tuned: Default selection charge plus official HPO wall time plus final fit and validation prediction. Missing evaluation rows inherit RF Default score and timing. Existing successful rows with incomplete timing remain missing. Training costs normalized by outer training-pool rows, test prediction by test rows. Complete fold means precede dataset medians.",
        imputed_score_cells=int(metrics.imputed.sum()),
        leaderboard_source=str(elo_path),
        leaderboard_sha256=sha(elo_path),
        metric_inputs_sha256=sha(metrics_path),
        selected_reports_sha256=sha(selection_path),
        benchmark_plan_sha256=sha(args.plan),
        timing_by_dataset_sha256=sha(args.output / "timing_by_dataset.csv"),
        timing_by_split_sha256=sha(args.output / "timing_by_split.csv"),
        reconstructed_search_clocks=int(
            folds.search_clock_status.eq("reconstructed").sum()
        ),
        script_sha256=sha(__file__),
        renderer_sha256=sha(Path(__file__).resolve().parents[1] / "pareto_style.py"),
        colors_sha256=sha(args.colors),
        all_report_checksums_verified=True,
        elo_identical_to_main=True,
        subset_solver_main_fit_max_abs_difference=difference,
        scopes=args.scopes,
        subset_figures=scoped_manifests,
    )
    if "main" in args.scopes:
        manifest["figure_sha256"] = sha(args.output / "pareto_scatter.pdf")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
