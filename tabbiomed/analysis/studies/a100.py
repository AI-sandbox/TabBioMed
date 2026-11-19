"""A100 analysis."""

from __future__ import annotations
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-a100-sensitivity-mpl")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from threadpoolctl import threadpool_limits
from tabbiomed.analysis.results import (
    BENCHMARK_PLAN,
    RESULTS,
    _test_metric,
    read_result,
)
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.common import (
    load_benchmark_models_config,
)
from tabbiomed.analysis.studies.pretraining import elo_tables
from tabbiomed.analysis.elo.missing import apply_missing_policy

OUTPUT = _OUTPUT_ROOT / "a100"
CLASSIFICATION = {"binary", "classification"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retry_metrics(main, retry, info):
    if (
        retry.get("artifact_type") != "evaluation_summary"
        or retry.get("schema_version") != 1
        or any(
            (
                main.get(k) != retry.get(k)
                for k in ("model", "dataset", "seed", "primary_metric")
            )
        )
        or (main.get("search_space") != retry.get("search_space"))
        or (retry.get("status") not in ("complete", "incomplete"))
    ):
        raise ValueError("A100 report does not match its Main evaluation")
    hardware = retry.get("hardware", {})
    if (
        "A100" not in hardware.get("name", "")
        or hardware.get("total_memory", 0) < 75 * 2**30
    ):
        raise ValueError("A100 retry requires recorded 80 GB hardware")
    n = int(info["n_folds"])
    if n:
        folds = retry.get("cv_fold_metrics", [])
        indexed = {f["fold_id"]: f for f in folds}
        if len(indexed) != len(folds) or set(indexed) - set(range(n)):
            raise ValueError("Invalid A100 outer folds")
        units = [indexed.get(i, {}) for i in range(n)]
    else:
        if retry.get("cv_fold_metrics"):
            raise ValueError("Unexpected CV retry for a fixed-split dataset")
        units = [retry]
    variants = retry.get("evaluated_variants", ["default", "best"])
    out = {}
    for key, column in [("default", "default_metric"), ("best", "tuned_metric")]:
        values = [
            (
                _test_metric(unit.get(key), main["primary_metric"])
                if key in variants
                else None
            )
            for unit in units
        ]
        out[column] = (
            float(np.mean(values)) if all((v is not None for v in values)) else None
        )
    return out


def fill_missing(main, retries):
    out = main.set_index(["model", "dataset"]).copy()
    for variant in ("default", "tuned"):
        column = variant + "_metric"
        out[variant + "_source"] = np.where(out[column].notna(), "main", "missing")
    changes = []
    for row in retries:
        key = (row["model"], row["dataset"])
        if key not in out.index:
            raise ValueError("Retry pair is outside the benchmark")
        for variant in ("default", "tuned"):
            column = variant + "_metric"
            value = row[column]
            if (
                pd.isna(out.at[key, column])
                and value is not None
                and np.isfinite(value)
            ):
                out.at[key, column] = value
                out.at[key, variant + "_source"] = "a100_retry"
                changes.append(
                    dict(
                        model=key[0],
                        dataset=key[1],
                        variant=variant,
                        metric_value=value,
                    )
                )
    return (out.reset_index(), pd.DataFrame(changes))


def mean_rank_table(frame, models):
    selected = frame.tuned_metric.fillna(frame.default_metric)
    values = frame.assign(value=selected).pivot(
        index="dataset", columns="model", values="value"
    )
    values = values.reindex(columns=models)
    tasks = frame.groupby("dataset").task.first()
    normalized = normalized_scores_with_failures(values, tasks)
    ranks = normalized.rank(axis=1, ascending=False, method="average")
    summary = ranks.mean().rename("mean_rank").to_frame()
    summary.index.name = "model"
    summary["position"] = summary.mean_rank.rank(method="average")
    summary["n_success"] = values.notna().sum()
    return (summary.sort_values(["mean_rank", "model"]), ranks)


def elo_table(frame):
    rows = []
    for row in frame.itertuples():
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
                    metric_value=getattr(row, variant + "_metric"),
                )
            )
    inputs, missing = apply_missing_policy(pd.DataFrame(rows), "rf_imputed")
    models, configurations = elo_tables(inputs)
    return (
        models.rename(columns={"rank_position": "position"}),
        configurations,
        inputs,
        missing,
    )


def compare(before, after, score):
    result = before[[score, "position"]].join(
        after[[score, "position"]], lsuffix="_main", rsuffix="_a100"
    )
    result["position_change"] = result.position_a100 - result.position_main
    result = result.sort_values(["position_main", score + "_main"])
    top_main = before.sort_values("position", kind="stable").head(5).index.tolist()
    top_a100 = after.sort_values("position", kind="stable").head(5).index.tolist()
    stats = dict(
        kendall_tau=float(
            kendalltau(result.position_main, result.position_a100).statistic
        ),
        top5_main=top_main,
        top5_a100=top_a100,
        top5_overlap=len(set(top_main) & set(top_a100)),
        max_absolute_position_change=float(result.position_change.abs().max()),
        n_positions_changed=int(result.position_change.ne(0).sum()),
    )
    return (result, stats)


def plot_comparisons(elo, ranks, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue, grey = ("#376FA3", "#858B94")
    n = len(ranks)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.3 * n + 1.3))
    for ax, table, title in zip(axes, [elo, ranks], ["RF-imputed Elo", "Mean rank"]):
        table = table.sort_values("position_main", kind="stable")
        y = np.arange(len(table))
        ax.hlines(
            y, table.position_main, table.position_a100, color="#B5BBC2", linewidth=1.3
        )
        ax.scatter(
            table.position_main,
            y,
            s=48,
            facecolor="white",
            edgecolor=grey,
            linewidth=1.1,
            zorder=2,
        )
        ax.scatter(table.position_a100, y, s=23, color=blue, zorder=3)
        ax.set_yticks(y, table.index)
        ax.set_ylim(n - 0.45, -0.65)
        ax.set_xlim(0.5, n + 0.6)
        ax.set_xticks([1, 5, 10, 15, n] if n < 23 else [1, 5, 10, 15, 20, n])
        ax.set_xlabel("Leaderboard position (1 = best)", labelpad=8)
        ax.set_title(title, pad=12)
        ax.grid(axis="x", color="#DDE1E6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.1, top=0.91, wspace=0.66)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        ax.title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        for ax in axes
    ]
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markeredgecolor=grey,
                markerfacecolor="white",
                markersize=7,
                label="Main",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markeredgecolor=blue,
                markerfacecolor=blue,
                markersize=6,
                label="With A100 retries",
            ),
        ],
        loc="center",
        bbox_to_anchor=(
            np.mean([(b.x0 + b.x1) / 2 for b in boxes]),
            np.mean([(b.y0 + b.y1) / 2 for b in boxes]),
        ),
        frameon=False,
        fontsize=11.5,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ensure_results()
    plan = load_benchmark(BENCHMARK_PLAN)
    config = load_benchmark_models_config()
    jobs = [
        (m, d, info)
        for d, info in sorted(plan["datasets"].items())
        for m in (
            config.expected_models_regression
            if info["task"] not in CLASSIFICATION
            else config.expected_models_binary_classification
        )
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        main_rows = list(pool.map(lambda j: read_result(*j), jobs))
    main_frame = pd.DataFrame(main_rows)
    retries, hashes = ([], {r["report_path"]: r["report_sha256"] for r in main_rows})
    for row in main_rows:
        path = Path(row["report_path"]).parent / "a100_retry/report.json"
        if not path.is_file():
            continue
        baseline = json.loads(Path(row["report_path"]).read_text())
        retry = json.loads(path.read_text())
        scores = retry_metrics(baseline, retry, plan["datasets"][row["dataset"]])
        hashes[str(path)] = digest(path)
        retries.append(
            dict(
                model=row["model"],
                dataset=row["dataset"],
                **scores,
                status=retry["status"],
                hardware=retry["hardware"]["name"],
                report_path=str(path),
                report_sha256=hashes[str(path)],
            )
        )
    augmented, changes = fill_missing(main_frame, retries)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    main_frame.to_csv(output / "main_results.csv", index=False)
    main_selection_fields = [
        "metric_value",
        "metric_source",
        "selected_variant",
        "n_default_splits",
        "n_tuned_splits",
    ]
    augmented.drop(columns=main_selection_fields).to_csv(
        output / "augmented_results.csv", index=False
    )
    pd.DataFrame(retries).to_csv(output / "retries.csv", index=False)
    changes.to_csv(output / "recovered_evaluations.csv", index=False)
    manifest = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        results_root=str(RESULTS),
        n_retries=len(retries),
        n_recovered_evaluations=len(changes),
        scopes={},
        mean_rank_policy="Complete Tuned else complete own Default, failures tie with worst success, complete outer-fold means.",
        elo_policy="Joint Default/Tuned pool, Models without HPO use Default, complete outer-fold means, RF Default imputation, half-win ties, RF Default anchored to 1000.",
        retry_policy="Fill only missing Main scores from the corresponding successful A100 variant, preserve all observed Main values.",
    )
    with threadpool_limits(limits=1):
        for name, models, datasets in [
            ("all_tasks", config.expected_models_regression, set(plan["datasets"])),
            (
                "classification",
                config.expected_models_binary_classification,
                {d for d, i in plan["datasets"].items() if i["task"] in CLASSIFICATION},
            ),
        ]:
            before = main_frame[
                main_frame.model.isin(models) & main_frame.dataset.isin(datasets)
            ].copy()
            after = augmented[
                augmented.model.isin(models) & augmented.dataset.isin(datasets)
            ].copy()
            assert len(before) == len(after) == len(datasets) * len(models)
            a, ar = mean_rank_table(before, models)
            b, br = mean_rank_table(after, models)
            rank_comparison, rank_stats = compare(a, b, "mean_rank")
            ea, ca, ia, ma = elo_table(before)
            eb, cb, ib, mb = elo_table(after)
            elo_comparison, elo_stats = compare(ea, eb, "elo")
            target = output / name
            target.mkdir(exist_ok=True)
            for label, table in [
                ("mean_rank_main", a),
                ("mean_rank_a100", b),
                ("dataset_ranks_main", ar),
                ("dataset_ranks_a100", br),
                ("mean_rank_comparison", rank_comparison),
                ("elo_main", ea),
                ("elo_a100", eb),
                ("elo_comparison", elo_comparison),
                ("elo_configurations_main", ca),
                ("elo_configurations_a100", cb),
            ]:
                table.to_csv(target / (label + ".csv"))
            ia.to_csv(target / "elo_inputs_main.csv", index=False)
            ib.to_csv(target / "elo_inputs_a100.csv", index=False)
            scope = dict(
                n_datasets=len(datasets),
                n_models=len(models),
                n_configurations=len(ca),
                mean_rank=rank_stats,
                elo=elo_stats,
                missing_main=ma,
                missing_a100=mb,
                model_dataset_coverage_main=int(a.n_success.sum()),
                model_dataset_coverage_a100=int(b.n_success.sum()),
                retries=sum(
                    (r["model"] in models and r["dataset"] in datasets for r in retries)
                ),
            )
            if name == "all_tasks":
                reference = _OUTPUT_ROOT / "model_summary"
                snapshot = pd.read_csv(reference / "selected_results.csv").set_index(
                    ["model", "dataset"]
                )
                current = before.set_index(["model", "dataset"])
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
                    snapshot[columns].sort_index(),
                    current[columns].sort_index(),
                    check_exact=False,
                    rtol=0,
                    atol=1e-12,
                )
                expected_ranks = (
                    pd.read_csv(reference / "summary.csv").set_index("model").rank_mean
                )
                np.testing.assert_allclose(
                    a.mean_rank.reindex(expected_ranks.index),
                    expected_ranks,
                    atol=1e-12,
                    rtol=0,
                )
                reference_elo = _OUTPUT_ROOT / "elo" / "rf_imputed/elo_all_entries.csv"
                expected_elo = pd.read_csv(reference_elo)
                expected_elo = expected_elo.set_index("competitor").elo
                np.testing.assert_allclose(
                    ca.elo.reindex(expected_elo.index), expected_elo, atol=1e-07, rtol=0
                )
                scope["main_rank_and_elo_match_paper"] = True
            plot_comparisons(elo_comparison, rank_comparison, output / (name + ".pdf"))
            manifest["scopes"][name] = scope
            print(name, json.dumps(scope), flush=True)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Source changed during analysis: {path}")
    manifest.update(
        all_source_hashes_verified=True,
        benchmark_plan_sha256=digest(BENCHMARK_PLAN),
        script_sha256=digest(Path(__file__).resolve().parents[1] / "studies/a100.py"),
        figures={name: digest(output / (name + ".pdf")) for name in manifest["scopes"]},
    )
    (output / "sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
