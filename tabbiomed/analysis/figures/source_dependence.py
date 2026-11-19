"""Source dependence figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import CONFIG, PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-domain-elo-mpl")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from threadpoolctl import threadpool_limits
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS
from tabbiomed.analysis.common import load_benchmark_models_config
from tabbiomed.analysis.figures.elo_regimes import (
    collect_model_inputs,
    digest,
    models_for_case,
    prepare_case,
)
from tabbiomed.analysis.elo.splitwise import (
    _orient_scores,
    battle_statistics,
    fit_elo,
)

SHARED = CONFIG / "shared_observation_groups.json"
FAMILIES = CONFIG / "source_families.json"
OUTPUT = _OUTPUT_ROOT / "source_dependence"
SERIES = [
    ("unweighted", "Unweighted", "black", "D"),
    ("same_x", "Same-X weighted", "#eb6834", "s"),
    ("family", "Source-family weighted", "#2a78d6", "^"),
]


def partition(groups, datasets):
    available = set(datasets)
    result = {d: f"singleton::{d}" for d in available}
    assigned = set()
    for group, members in groups.items():
        if not members:
            raise ValueError(f"Empty group: {group}")
        for dataset in members:
            if dataset not in available:
                raise ValueError(f"Group contains out-of-scope dataset: {dataset}")
            if dataset in assigned:
                raise ValueError(f"Duplicate group membership: {dataset}")
            assigned.add(dataset)
            result[dataset] = group
    return result


def load_groups(datasets, shared_path=SHARED, families_path=FAMILIES):
    same_x = json.loads(shared_path.read_text())["groups"]
    definitions = json.loads(families_path.read_text())["families"]
    families = {}
    for name, definition in definitions.items():
        members = list(definition.get("datasets", []))
        for group in definition.get("shared_sample_groups", []):
            if group not in same_x:
                raise ValueError(f"Unknown shared-sample group: {group}")
            members.extend(same_x[group])
        families[name] = sorted(members)
    partition(same_x, datasets)
    partition(families, datasets)
    exclusions = dict(same_x)
    for name, members in families.items():
        if name in exclusions and set(exclusions[name]) != set(members):
            raise ValueError(f"Conflicting exclusion group: {name}")
        exclusions[name] = members
    return (same_x, families, exclusions)


def weighted_statistics(statistics, datasets, groups=None):
    labels = datasets if groups is None else [groups[d] for d in datasets]
    keys = sorted(set(labels))
    codes = {label: i for i, label in enumerate(keys)}
    ids = np.array([codes[label] for label in labels])
    sizes = np.bincount(ids, minlength=len(keys))
    weights = 1.0 / sizes[ids]
    grouped = np.zeros((len(keys), statistics.shape[1]))
    np.add.at(grouped, ids, statistics * weights[:, None])
    return (grouped, weights)


def weighted_elo(frame, groups=None, bootstrap=200, seed=0):
    names, datasets, design, labels, statistics = battle_statistics(frame)
    grouped, _ = weighted_statistics(statistics, datasets, groups)
    point = fit_elo(design, labels, grouped.sum(axis=0))
    offset = 1000 - point[names.index("RandomForest")]
    table = pd.DataFrame(dict(model=names, value=point + offset))
    if bootstrap:
        rng = np.random.default_rng(seed)
        samples = np.array(
            [
                fit_elo(
                    design,
                    labels,
                    grouped[rng.integers(0, len(grouped), len(grouped))].sum(axis=0),
                )
                for _ in range(bootstrap)
            ]
        )
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        table["lo"], table["hi"] = (lo + offset, hi + offset)
    else:
        table["lo"], table["hi"] = (np.nan, np.nan)
    table["rank_position"] = table.value.rank(ascending=False, method="average")
    return table.sort_values(["value", "model"], ascending=[False, True]).reset_index(
        drop=True
    )


def compare(reference, variant):
    left = reference.set_index("model").sort_index()
    right = variant.set_index("model").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Ranking comparisons need identical model rosters")
    shifts = (left.rank_position - right.rank_position).abs()
    a, b = (list(reference.model.head(5)), list(variant.model.head(5)))
    return dict(
        kendall_tau_b=float(kendalltau(left.value, right.value).statistic),
        mean_abs_shift=float(shifts.mean()),
        max_abs_shift=float(shifts.max()),
        leader=str(variant.model.iloc[0]),
        top5=",".join(b),
        top5_count=len(set(a) & set(b)),
        top5_fraction=len(set(a) & set(b)) / 5,
    )


def analyze(inputs, shared, families, exclusions, bootstrap, seed):
    datasets = sorted(inputs.dataset.unique())
    baseline = weighted_elo(inputs, bootstrap=bootstrap, seed=seed)
    tables = {"unweighted": baseline}
    comparisons, dropped = ([], [])
    boards = [baseline.assign(variant="unweighted", ranking="elo")]
    for name, groups in [("same_x", shared), ("family", families)]:
        task_groups = partition(groups, datasets)
        table = weighted_elo(inputs, task_groups, bootstrap, seed)
        tables[name] = table
        comparisons.append(
            dict(
                variant=name,
                n_groups=len(set(task_groups.values())),
                **compare(baseline, table),
            )
        )
        boards.append(table.assign(variant=name, ranking="elo"))
    for name, members in exclusions.items():
        remaining = inputs[~inputs.dataset.isin(members)]
        table = weighted_elo(remaining, bootstrap=0, seed=seed)
        dropped.append(
            dict(
                group=name,
                n_removed=len(datasets) - remaining.dataset.nunique(),
                **compare(baseline, table),
            )
        )
        boards.append(table.assign(variant=f"lofo::{name}", ranking="elo"))
    return (
        tables,
        pd.DataFrame(comparisons),
        pd.DataFrame(dropped),
        pd.concat(boards, ignore_index=True),
    )


def plot(boards, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 15,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )
    tables = {
        v: boards[boards.ranking.eq("elo") & boards.variant.eq(v)].set_index("model")
        for v, *_ in SERIES
    }
    order = (
        tables["unweighted"]
        .sort_values(["value", "model"], ascending=[False, True])
        .index
    )
    fig, ax = plt.subplots(figsize=(12.5, 4.7))
    handles = []
    for i, (variant, label, color, marker) in enumerate(SERIES):
        table = tables[variant].reindex(order)
        x = np.arange(len(order)) + (i - 1) * 0.19
        ax.vlines(
            x,
            table.lo,
            table.hi,
            color="0.35" if variant == "unweighted" else color,
            lw=1.1,
            zorder=2,
        )
        ax.scatter(
            x, table.value, marker=marker, color=color, s=34, linewidth=0.5, zorder=3
        )
        handles.append(
            Line2D([0], [0], color=color, marker=marker, lw=1.1, label=label)
        )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        order, rotation=45, ha="right", rotation_mode="anchor", fontweight="normal"
    )
    ax.set_xlim(-0.55, len(order) - 0.45)
    ax.set_ylabel("Elo", labelpad=6)
    ax.axhline(1000, color="0.45", lw=0.9, ls="--", zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".84", lw=0.65)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        fontsize=19,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    panel = ax.get_position()
    center = (panel.x0 + panel.x1) * fig.get_figwidth() / 2
    half = max(center - tight.x0, tight.x1 - center)
    box = Bbox.from_extents(center - half, tight.y0, center + half, tight.y1).padded(
        0.09
    )
    fig.savefig(output, bbox_inches=box)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    ensure_results()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.render_only:
        boards = pd.read_csv(
            args.output_dir / "leaderboards_long.csv", float_precision="round_trip"
        )
        manifest = json.loads((args.output_dir / "analysis.json").read_text())
    else:
        plan = load_benchmark(BENCHMARK_PLAN)
        config = load_benchmark_models_config()
        models = models_for_case(
            config.expected_models_binary_classification, "all_tasks"
        )
        if set(models) != set(config.expected_models_regression):
            raise ValueError("Inputs differ from the configured eligible models")
        raw, metadata = collect_model_inputs(
            plan["datasets"], models, RESULTS, PROCESSED
        )
        inputs = prepare_case(raw, models, "all_tasks")
        shared, families, exclusions = load_groups(metadata.dataset)
        if (
            len(set(partition(shared, metadata.dataset).values())),
            len(set(partition(families, metadata.dataset).values())),
            len(exclusions),
        ) != (71, 42, 19):
            raise ValueError("Unexpected source grouping")
        reference = _OUTPUT_ROOT / "elo_regimes" / "all_tasks_inputs.csv"
        existing = (
            pd.read_csv(reference, float_precision="round_trip")
            .set_index(["dataset", "model"])
            .sort_index()
        )
        current = inputs.set_index(["dataset", "model"]).sort_index()
        pd.testing.assert_series_equal(current.metric_value, existing.metric_value)
        print(
            f"{len(models)} models, {len(plan['datasets'])} tasks, {int(inputs.imputed.sum())} RF fills",
            flush=True,
        )
        with threadpool_limits(limits=1):
            tables, comparisons, dropped, boards = analyze(
                inputs, shared, families, exclusions, args.bootstrap, args.seed
            )
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
        inputs.to_csv(args.output_dir / "inputs.csv", index=False)
        metadata.to_csv(args.output_dir / "datasets.csv", index=False)
        for name, table in tables.items():
            table.to_csv(args.output_dir / f"elo_{name}.csv", index=False)
        comparisons.to_csv(args.output_dir / "weighted_summary.csv", index=False)
        dropped.to_csv(args.output_dir / "lofo_summary.csv", index=False)
        boards.to_csv(args.output_dir / "leaderboards_long.csv", index=False)
        (args.output_dir / "groups.json").write_text(
            json.dumps(
                dict(same_x=shared, family=families, exclusions=exclusions), indent=2
            )
            + "\n"
        )
        oriented = inputs.assign(score=_orient_scores(inputs))
        oriented.pivot(index="dataset", columns="model", values="score").to_csv(
            args.output_dir / "score_matrix_rf_imputed.csv"
        )
        manifest = dict(
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            results_root=str(RESULTS.resolve()),
            models=models,
            n_models=len(models),
            n_datasets=len(plan["datasets"]),
            bootstrap=args.bootstrap,
            seed=args.seed,
            selection_counts={
                str(k): int(v) for k, v in inputs.metric_source.value_counts().items()
            },
            policy=dict(
                score_selection="Complete Tuned mean, then own complete Default mean, then RF Default",
                default_only="LinearModel and NeoLsSVM",
                cv="All outer folds required within each variant",
                ties="Half win each",
                weighting="Each group's total task weight is one",
                uncertainty="Dataset bootstrap when unweighted, cluster bootstrap when weighted",
                calibration="RandomForest model = 1000, same point-fit offset for interval endpoints",
            ),
            weighted=comparisons.to_dict(orient="records"),
            exclusions=dict(
                n_groups=len(dropped),
                mean_kendall_tau_b=float(dropped.kendall_tau_b.mean()),
                min_kendall_tau_b=float(dropped.kendall_tau_b.min()),
                min_top5_count=int(dropped.top5_count.min()),
                max_abs_shift=float(dropped.max_abs_shift.max()),
            ),
            source_hashes={
                str(p): digest(p)
                for p in [
                    BENCHMARK_PLAN,
                    SHARED,
                    FAMILIES,
                    Path(__file__).resolve().parents[1]
                    / "figures/source_dependence.py",
                    Path(__file__).resolve().parents[1] / "figures/elo_regimes.py",
                    Path(__file__).resolve().parents[1] / "results.py",
                    Path(__file__).resolve().parents[1] / "elo/splitwise.py",
                    Path(__file__).resolve().parents[1] / "elo/fitting.py",
                ]
            },
            source_reports_verified_unchanged=True,
            matches_current_regime_inputs=True,
            outputs={
                p.name: digest(p)
                for p in args.output_dir.iterdir()
                if p.suffix in {".csv"} or p.name == "groups.json"
            },
        )
        print(comparisons.to_string(index=False), flush=True)
        print(json.dumps(manifest["exclusions"]), flush=True)
    figure = args.output_dir / "source_dependence.pdf"
    plot(boards, figure)
    manifest.update(
        figure=str(figure),
        figure_sha256=digest(figure),
        render_script_sha256=digest(
            Path(__file__).resolve().parents[1] / "figures/source_dependence.py"
        ),
    )
    (args.output_dir / "analysis.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote {figure}", flush=True)


if __name__ == "__main__":
    main()
