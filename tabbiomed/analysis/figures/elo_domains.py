"""Elo domains figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import MODEL_FAMILIES, CONFIG, PROCESSED, OUTPUT as _OUTPUT_ROOT
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
from threadpoolctl import threadpool_limits
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS
from tabbiomed.analysis.common import (
    load_benchmark_models_config,
)
from tabbiomed.analysis.figures.elo_regimes import (
    collect_model_inputs,
    digest,
    models_for_case,
    prepare_case,
)
from tabbiomed.analysis.elo.splitwise import battle_statistics, fit_elo

OUTPUT = _OUTPUT_ROOT / "elo_domains"
GROUPS = CONFIG / "shared_observation_groups.json"
CLINICAL_SAMPLE_CUTOFF = 2000
MIN_BOOTSTRAP_GROUPS = 5
DEFAULT_REFERENCE = "RandomForest"
DOMAINS = {
    "clinical & healthcare": ("ehr", "Clinical and Healthcare"),
    "drug discovery": ("drug", "Drug Discovery and Molecular Properties"),
    "genomics": ("genom", "Genomics"),
    "metabolomics": ("mtbls", "Metabolomics"),
    "proteomics": ("ptomics", "Proteomics"),
    "single-cell": ("sc", "Single-Cell"),
    "systems biology": ("sb", "Systems Biology"),
    "transcriptomics": ("trans", "Transcriptomics"),
}
STYLE = {
    "font": "Times New Roman",
    "model_fontsize": 16,
    "width_inches": 12.5,
    "panel_height_inches": 4.2,
}


def prepare_domain_inputs(inputs, models):
    frame = inputs.copy()
    tuned = pd.to_numeric(frame.tuned_metric, errors="coerce")
    default = pd.to_numeric(frame.default_metric, errors="coerce")
    default_only = ~frame.hpo_enabled
    use_tuned = ~default_only & np.isfinite(tuned)
    use_default = ~use_tuned & np.isfinite(default)
    frame["raw_metric"] = np.where(
        use_tuned, tuned, np.where(use_default, default, np.nan)
    )
    frame["selected_variant"] = np.select(
        [use_tuned, use_default], ["tuned", "default"], default="failed"
    )
    frame["metric_source"] = "failed"
    frame.loc[use_tuned, "metric_source"] = "tuned_test"
    frame.loc[use_default & default_only, "metric_source"] = "default_only"
    frame.loc[use_default & ~default_only, "metric_source"] = "default_fallback"
    frame = prepare_case(frame, models, "all_tasks")
    frame["score_source_model"] = np.where(
        frame.imputed, DEFAULT_REFERENCE, frame.model
    )
    frame["score_source_variant"] = np.where(
        frame.imputed, "default", frame.selected_variant
    )
    return frame


def observation_partition(datasets, groups):
    available = set(datasets)
    partition = {d: f"singleton::{d}" for d in available}
    assigned = set()
    for name, members in groups.items():
        for dataset in members:
            if dataset in assigned:
                raise ValueError(f"Duplicate shared-observation membership: {dataset}")
            if dataset in available:
                partition[dataset] = name
            assigned.add(dataset)
    return partition


def assign_panels(metadata, groups):
    if metadata.empty or metadata.dataset.duplicated().any():
        raise ValueError("Expected unique, nonempty dataset metadata")
    frame = metadata.copy()
    frame["domain_key"] = frame.domain.str.casefold().str.strip()
    if not set(frame.domain_key).issubset(DOMAINS):
        raise ValueError("Unknown biomedical domain")
    frame["domain_slug"] = frame.domain_key.map(lambda k: DOMAINS[k][0])
    frame["domain"] = frame.domain_key.map(lambda k: DOMAINS[k][1])
    partition = observation_partition(frame.dataset, groups)
    frame["observation_group"] = frame.dataset.map(partition)
    frame["panel"] = frame.domain_slug
    clinical = frame.domain_slug.eq("ehr")
    frame.loc[clinical & frame.num_samples.le(CLINICAL_SAMPLE_CUTOFF), "panel"] = (
        "ehr_up_to_2000"
    )
    frame.loc[clinical & frame.num_samples.gt(CLINICAL_SAMPLE_CUTOFF), "panel"] = (
        "ehr_over_2000"
    )
    if frame.groupby("observation_group").domain_slug.nunique().gt(1).any():
        raise ValueError("A shared-observation group crosses biomedical domains")
    if frame.groupby("observation_group").panel.nunique().gt(1).any():
        raise ValueError("A shared-observation group crosses panels")
    return frame.drop(columns="domain_key").sort_values(
        ["domain_slug", "panel", "dataset"]
    )


def group_bootstrap_counts(statistics, group_labels, bootstrap, seed):
    if bootstrap < 1 or len(group_labels) != len(statistics):
        raise ValueError("Invalid bootstrap count or group labels")
    names = sorted(set(group_labels))
    if len(names) < 2:
        raise ValueError("Cluster bootstrap needs at least two observation groups")
    positions = {name: i for i, name in enumerate(names)}
    grouped = np.zeros((len(names), statistics.shape[1]))
    for row, name in zip(statistics, group_labels):
        grouped[positions[name]] += row
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        indices = rng.integers(0, len(names), len(names))
        yield grouped[indices].sum(axis=0)


def cluster_rate(frame, group_map, bootstrap=200, seed=0):
    names, datasets, design, labels, statistics = battle_statistics(frame)
    group_labels = [group_map[d] for d in datasets]
    point = fit_elo(design, labels, statistics.sum(axis=0))
    if bootstrap:
        samples = np.array(
            [
                fit_elo(design, labels, counts)
                for counts in group_bootstrap_counts(
                    statistics, group_labels, bootstrap, seed
                )
            ]
        )
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    else:
        lo = hi = np.full(len(names), np.nan)
    offset = 1000 - point[names.index("RandomForest")]
    table = pd.DataFrame(
        dict(
            model=names,
            elo_calibrated=point + offset,
            elo_cal_lo_95=lo + offset,
            elo_cal_hi_95=hi + offset,
        )
    )
    table["elo_rank"] = table.elo_calibrated.rank(ascending=False, method="average")
    return table


def panel_description(metadata):
    return dict(
        panel=str(metadata.panel.iloc[0]),
        domain=str(metadata.domain.iloc[0]),
        domain_slug=str(metadata.domain_slug.iloc[0]),
        n_datasets=len(metadata),
        n_observation_groups=int(metadata.observation_group.nunique()),
        samples_min=int(metadata.num_samples.min()),
        samples_max=int(metadata.num_samples.max()),
        features_min=int(metadata.num_features.min()),
        features_max=int(metadata.num_features.max()),
        task_counts={str(k): int(v) for k, v in metadata.task.value_counts().items()},
        datasets=sorted(metadata.dataset),
        observation_groups={
            str(g): sorted(f.dataset) for g, f in metadata.groupby("observation_group")
        },
    )


def summarize(inputs, metadata, models, bootstrap, seed):
    summaries, panels = ([], [])
    for i, (panel, meta) in enumerate(metadata.groupby("panel", sort=True)):
        frame = inputs[inputs.dataset.isin(meta.dataset)]
        if (
            frame.groupby("dataset")
            .model.apply(set)
            .map(lambda v: v != set(models))
            .any()
        ):
            raise ValueError(f"Incomplete model roster for {panel}")
        description = panel_description(meta)
        panel_bootstrap = (
            bootstrap
            if description["n_observation_groups"] >= MIN_BOOTSTRAP_GROUPS
            else 0
        )
        table = cluster_rate(
            frame,
            meta.set_index("dataset").observation_group.to_dict(),
            bootstrap=panel_bootstrap,
            seed=seed + i,
        )
        description["bootstrap_replicates"] = panel_bootstrap
        panels.append(description)
        table["panel"] = panel
        table["domain_slug"] = description["domain_slug"]
        table["n_datasets"] = len(meta)
        table["n_observation_groups"] = description["n_observation_groups"]
        table["n_imputed"] = table.model.map(
            frame.groupby("model").imputed.sum()
        ).astype(int)
        table["n_default_fallback"] = table.model.map(
            frame.assign(fallback=frame.metric_source.eq("default_fallback"))
            .groupby("model")
            .fallback.sum()
        ).astype(int)
        summaries.append(table)
        print(
            f"{panel}: {len(meta)} datasets, {description['n_observation_groups']} groups",
            flush=True,
        )
    return (pd.concat(summaries, ignore_index=True), panels)


def range_text(minimum, maximum):
    return f"{minimum:,}" if minimum == maximum else f"{minimum:,}–{maximum:,}"


def plot_domain(summary, panels, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [STYLE["font"], "Times", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": STYLE["model_fontsize"],
            "ytick.labelsize": 15,
        }
    )
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(STYLE["width_inches"], STYLE["panel_height_inches"] * len(panels)),
        squeeze=False,
        sharey=True,
    )
    for ax, panel in zip(axes.ravel(), panels):
        sub = summary[summary.panel.eq(panel["panel"])].sort_values(
            "elo_calibrated", ascending=False
        )
        x = np.arange(len(sub))
        intervals = np.isfinite(sub.elo_cal_lo_95) & np.isfinite(sub.elo_cal_hi_95)
        ax.vlines(
            x[intervals],
            sub.loc[intervals, "elo_cal_lo_95"],
            sub.loc[intervals, "elo_cal_hi_95"],
            color="0.35",
            lw=1.1,
            zorder=2,
        )
        ax.scatter(x, sub.elo_calibrated, marker="D", s=34, color="black", zorder=3)
        ax.axhline(1000, color="0.45", lw=0.9, ls="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            sub.model,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontweight="normal",
        )
        ax.set_xlim(-0.55, len(sub) - 0.45)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.84", linewidth=0.65)
        ax.set_ylabel("Elo")
        ax.set_title(
            f"{panel['n_datasets']} datasets",
            loc="left",
            fontsize=17,
            fontweight="bold",
            pad=34,
        )
        ax.text(
            0,
            1.035,
            f"Samples {range_text(panel['samples_min'], panel['samples_max'])}    Features {range_text(panel['features_min'], panel['features_max'])}",
            transform=ax.transAxes,
            fontsize=15,
            va="bottom",
        )
    fig.tight_layout(pad=0.5, h_pad=1.8)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def render(summary, manifest, output):
    figures = {}
    for slug, _ in DOMAINS.values():
        panels = [p for p in manifest["panels"] if p["domain_slug"] == slug]
        panels.sort(key=lambda p: p["samples_min"])
        if not panels:
            continue
        path = output / f"domain_elo_{slug}.pdf"
        plot_domain(summary, panels, path)
        figures[slug] = dict(path=str(path), sha256=digest(path))
    manifest.update(
        figures=figures,
        figure_style=STYLE,
        render_script_sha256=digest(
            Path(__file__).resolve().parents[1] / "figures/elo_domains.py"
        ),
    )
    (output / "analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=BENCHMARK_PLAN)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED)
    parser.add_argument("--model-families", type=Path, default=MODEL_FAMILIES)
    parser.add_argument("--groups", type=Path, default=GROUPS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    ensure_results(args.results_dir)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.render_only:
        summary = pd.read_csv(
            args.output_dir / "summary.csv", float_precision="round_trip"
        )
        manifest = json.loads((args.output_dir / "analysis.json").read_text())
        render(summary, manifest, args.output_dir)
        return
    plan = load_benchmark(args.plan, args.processed_path)
    config = load_benchmark_models_config(args.model_families, plan_path=args.plan)
    models = models_for_case(
        list(config.expected_models_binary_classification), "all_tasks"
    )
    if set(models) != set(config.expected_models_regression):
        raise ValueError("Domain inputs differ from the configured eligible models")
    raw, metadata = collect_model_inputs(
        plan["datasets"], models, args.results_dir, args.processed_path
    )
    metadata["domain"] = [
        json.loads(Path(p).read_text())["label"] for p in metadata.metadata_path
    ]
    metadata = assign_panels(metadata, json.loads(args.groups.read_text())["groups"])
    if set(metadata.dataset) != set(plan["datasets"]):
        raise ValueError("Domain inputs differ from the configured dataset scope")
    inputs = prepare_domain_inputs(raw, models).merge(
        metadata[["dataset", "domain", "domain_slug", "panel", "observation_group"]],
        on="dataset",
        validate="many_to_one",
    )
    with threadpool_limits(limits=1):
        summary, panels = summarize(inputs, metadata, models, args.bootstrap, args.seed)
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
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        results_root=str(args.results_dir.resolve()),
        models=models,
        n_models=len(models),
        n_datasets=len(metadata),
        panels=panels,
        policy=dict(
            score_selection="Complete Tuned test mean, then the model's complete Default test mean",
            default_only="Models without HPO use Default",
            missing="Same-task complete RF Default mean only if neither own variant is complete",
            ties="Half win each",
            point_weights="Equal weight per dataset",
            anchor="RandomForest model competitor = 1000",
            bootstrap=args.bootstrap,
            seed=args.seed,
            uncertainty="Percentile 95% cluster-bootstrap intervals, resampling shared-observation groups",
            minimum_groups_for_intervals=MIN_BOOTSTRAP_GROUPS,
            interval_calibration="Point-fit offset, as in the main Elo pipeline",
            grouping="Pooled domains except Clinical, split at 2000 samples using metadata only",
        ),
        source_hashes={
            str(p): digest(p)
            for p in [
                args.plan,
                args.groups,
                args.model_families,
                Path(__file__).resolve().parents[1] / "figures/elo_domains.py",
                Path(__file__).resolve().parents[1] / "results.py",
                Path(__file__).resolve().parents[1] / "figures/elo_regimes.py",
                Path(__file__).resolve().parents[1] / "elo/splitwise.py",
                Path(__file__).resolve().parents[1] / "elo/fitting.py",
            ]
        },
        outputs={
            name: digest(args.output_dir / name)
            for name in ["inputs.csv", "datasets.csv", "summary.csv"]
        },
        selection_counts={
            str(k): int(v) for k, v in inputs.metric_source.value_counts().items()
        },
        source_reports_verified_unchanged=True,
        metadata_verified_unchanged=True,
    )
    render(summary, manifest, args.output_dir)
    print(f"Wrote eight domain figures to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
