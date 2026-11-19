"""Elo regimes figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import MODEL_FAMILIES, PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.datasets import dataset_directory, load_metadata
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, read_result
from tabbiomed.analysis.regimes import QUADRANT_LABELS, assign_quadrant
from tabbiomed.analysis.common import load_benchmark_models_config
from tabbiomed.analysis.elo.splitwise import rate

DEFAULT_ANCHOR_MODEL = "RandomForest"
CALIBRATION_ELO = 1000
CLASSIFICATION_TASKS = frozenset({"binary", "classification"})
ALL_TASKS = CLASSIFICATION_TASKS | {"regression", "multiregression"}
CLASSIFICATION_ONLY_MODELS = (
    load_benchmark_models_config().regression_optional_missing_models
)
CASES = ("classification", "all_tasks")
DEFAULT_OUTPUT = _OUTPUT_ROOT / "elo_regimes"
FIGURE_STYLE = {
    "font": "Times New Roman",
    "size_inches": [12.5, 13.6],
    "x_tick_size": 19,
    "y_tick_size": 19,
    "y_label_size": 22,
    "panel_label_size": 20,
    "range_size": 18,
    "ranges": "inline, regular weight",
    "horizontal_margins": "equal around panel frames",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models_for_case(models: list[str], case: str) -> list[str]:
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")
    return sorted(
        (
            m
            for m in models
            if case == "classification" or m not in CLASSIFICATION_ONLY_MODELS
        )
    )


def collect_model_inputs(
    datasets: dict[str, dict[str, Any]],
    models: list[str],
    results_dir: Path,
    processed_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = []
    for dataset, info in sorted(datasets.items()):
        path = dataset_directory(dataset, processed_path) / "metadata.json"
        meta = load_metadata(dataset, processed_path)
        if (
            meta.get("task") != info["task"]
            or info["task"] not in ALL_TASKS
            or (not meta.get("READY"))
        ):
            raise ValueError(f"Invalid task metadata: {path}")
        dimensions = {key: float(meta[key]) for key in ("num_samples", "num_features")}
        if any(
            (
                not np.isfinite(v) or v <= 0 or (not v.is_integer())
                for v in dimensions.values()
            )
        ):
            raise ValueError(f"Invalid dataset dimensions: {path}")
        metadata.append(
            dict(
                dataset=dataset,
                task=info["task"],
                **{k: int(v) for k, v in dimensions.items()},
                metadata_path=str(path),
                metadata_sha256=digest(path),
            )
        )
    jobs = [
        (m, d, info)
        for d, info in sorted(datasets.items())
        for m in models
        if info["task"] in CLASSIFICATION_TASKS or m not in CLASSIFICATION_ONLY_MODELS
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: read_result(*job, root=results_dir), jobs))
    rows = [{**result, "raw_metric": result["metric_value"]} for result in results]
    return (pd.DataFrame(rows), pd.DataFrame(metadata))


def prepare_case(inputs: pd.DataFrame, models: list[str], case: str) -> pd.DataFrame:
    tasks = CLASSIFICATION_TASKS if case == "classification" else ALL_TASKS
    frame = inputs[inputs.model.isin(models) & inputs.task.isin(tasks)].copy()
    if frame.empty or frame.duplicated(["dataset", "model"]).any():
        raise ValueError(f"Empty or duplicate model/dataset inputs for {case}")
    if any((set(g.model) != set(models) for _, g in frame.groupby("dataset"))):
        raise ValueError(f"Incomplete model roster for {case}")
    rf = frame[frame.model.eq(DEFAULT_ANCHOR_MODEL)].set_index("dataset").default_metric
    if not np.isfinite(pd.to_numeric(rf)).all():
        raise ValueError(
            "Every dataset needs a complete RandomForest default mean for imputation"
        )
    frame["rf_default_metric"] = frame.dataset.map(rf)
    frame["metric_value"] = pd.to_numeric(frame.raw_metric)
    frame["imputed"] = ~np.isfinite(frame.metric_value)
    frame.loc[frame.imputed, "metric_value"] = frame.loc[
        frame.imputed, "rf_default_metric"
    ]
    frame.loc[frame.imputed, "metric_source"] = "rf_default_imputed"
    frame["score_source_model"] = np.where(
        frame.imputed, DEFAULT_ANCHOR_MODEL, frame.model
    )
    frame["score_source_variant"] = np.where(
        frame.imputed, "default", frame.selected_variant
    )
    frame["competitor"] = frame.model
    frame["fold"] = 0
    return frame.sort_values(["dataset", "model"]).reset_index(drop=True)


def assign_quadrants(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if metadata.empty or metadata.dataset.duplicated().any():
        raise ValueError("Regimes need nonempty, unique dataset metadata")
    med_s = float(np.median(np.log10(metadata.num_samples)))
    med_f = float(np.median(np.log10(metadata.num_features)))
    out = metadata.copy()
    out["quadrant"] = [
        assign_quadrant(s, f, med_s, med_f)
        for s, f in zip(out.num_samples, out.num_features)
    ]
    return (out, {"med_log10_samples": med_s, "med_log10_features": med_f})


def summarize_quadrants_elo(
    inputs: pd.DataFrame, models: list[str], *, bootstrap: int, seed: int
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    stats = {}
    for quadrant in QUADRANT_LABELS:
        frame = inputs[inputs.quadrant.eq(quadrant)]
        if frame.empty:
            raise ValueError(f"No datasets in {quadrant}")
        print(
            f"{quadrant}: {frame.dataset.nunique()} datasets, {len(models)} models",
            flush=True,
        )
        table, _, meta = rate(
            frame, bootstrap=bootstrap, seed=seed, reference=DEFAULT_ANCHOR_MODEL
        )
        if set(table.competitor) != set(models):
            raise ValueError(f"Elo dropped requested models in {quadrant}")
        table = table.rename(
            columns={
                "competitor": "model",
                "elo": "elo_calibrated",
                "elo_lo_95": "elo_cal_lo_95",
                "elo_hi_95": "elo_cal_hi_95",
            }
        )
        table["quadrant"] = quadrant
        table["n_datasets_in_quadrant"] = frame.dataset.nunique()
        table["n_imputed"] = table.model.map(
            frame.groupby("model").imputed.sum()
        ).astype(int)
        table["elo_rank"] = table.elo_calibrated.rank(ascending=False, method="average")
        rows.append(table)
        stats[quadrant] = {
            **meta,
            "calibration": "RandomForest model competitor = 1000",
            "split_weights": "1 per dataset after complete outer-fold averaging",
            "n_imputed_cells": int(frame.imputed.sum()),
            "n_default_fallback_cells": int(
                frame.metric_source.eq("default_fallback").sum()
            ),
        }
    return (pd.concat(rows, ignore_index=True), stats)


def _quadrant_ranges(ds_meta: list[dict[str, Any]]) -> str:
    samples = [int(row["num_samples"]) for row in ds_meta]
    features = [int(row["num_features"]) for row in ds_meta]
    return f"{len(ds_meta)} tasks, samples {min(samples):,}-{max(samples):,}, features {min(features):,}-{max(features):,}"


def plot_cleveland_quadrants_elo(
    *,
    summary: pd.DataFrame,
    quad_to_ds_meta: dict[str, list[dict[str, Any]]],
    out_pdf: Path,
    panel_labels: tuple[str, ...] = QUADRANT_LABELS,
    size_inches: tuple[float, float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from matplotlib.ticker import MultipleLocator
    from matplotlib.transforms import Bbox

    labels = panel_labels
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FIGURE_STYLE["font"], "Times"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 13,
            "axes.labelsize": FIGURE_STYLE["y_label_size"],
            "axes.titlesize": 15,
            "xtick.labelsize": FIGURE_STYLE["x_tick_size"],
            "ytick.labelsize": FIGURE_STYLE["y_tick_size"],
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n = len(labels)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=size_inches or FIGURE_STYLE["size_inches"],
        sharey=True,
        constrained_layout=False,
    )
    axes_list = list(np.atleast_1d(axes).ravel())
    y_vals: list[float] = []
    for ax, label in zip(axes_list, labels):
        sub = summary[summary["quadrant"] == label].copy()
        sub = sub.sort_values("elo_calibrated", ascending=False, na_position="last")
        sub = sub[np.isfinite(sub["elo_calibrated"].to_numpy(dtype=float))]
        ordered = [str(m) for m in sub["model"].tolist()]
        elos = sub["elo_calibrated"].to_numpy(dtype=float)
        lo = sub["elo_cal_lo_95"].to_numpy(dtype=float)
        hi = sub["elo_cal_hi_95"].to_numpy(dtype=float)
        for i, (elo, low, high) in enumerate(zip(elos, lo, hi)):
            if not np.isfinite(elo):
                continue
            y_vals.append(float(elo))
            if np.isfinite(low) and np.isfinite(high):
                ax.vlines(
                    i, low, high, color="0.35", linewidth=1.1, alpha=0.95, zorder=2
                )
                y_vals.extend([float(low), float(high)])
            ax.scatter(
                [i],
                [elo],
                s=34,
                color="black",
                marker="D",
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        header = HPacker(
            children=[
                TextArea(
                    label,
                    textprops={
                        "size": FIGURE_STYLE["panel_label_size"],
                        "weight": "bold",
                    },
                ),
                TextArea(
                    _quadrant_ranges(quad_to_ds_meta[label]),
                    textprops={"size": FIGURE_STYLE["range_size"], "weight": "normal"},
                ),
            ],
            align="baseline",
            pad=0,
            sep=9,
        )
        ax.add_artist(
            AnchoredOffsetbox(
                loc="lower left",
                child=header,
                frameon=False,
                pad=0,
                bbox_to_anchor=(0, 1.025),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
        )
        ax.set_xticks(list(range(len(ordered))))
        ax.set_xticklabels(
            ordered,
            fontsize=FIGURE_STYLE["x_tick_size"],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlim(-0.55, len(ordered) - 0.45)
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.axhline(
            CALIBRATION_ELO, color="0.45", linewidth=0.8, linestyle="--", zorder=1
        )
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)
        ax.grid(False, axis="x")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("0.15")
        ax.tick_params(axis="both", width=0.8, length=3, pad=2)
    if y_vals:
        y_min = min(y_vals)
        y_max = max(y_vals)
        pad = max(25.0, 0.035 * (y_max - y_min + 1.0))
        for ax in axes_list:
            ax.set_ylim(y_min - pad, y_max + pad)
    fig.supylabel("Elo", fontsize=FIGURE_STYLE["y_label_size"], x=0.008)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.012, 0, 1, 1), pad=0.35, h_pad=0.8)
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    panel = axes_list[0].get_position()
    center = (panel.x0 + panel.x1) * fig.get_figwidth() / 2
    half_width = max(center - tight_bbox.x0, tight_bbox.x1 - center)
    export_bbox = Bbox.from_extents(
        center - half_width, tight_bbox.y0, center + half_width, tight_bbox.y1
    ).padded(0.09)
    fig.savefig(out_pdf, bbox_inches=export_bbox)
    plt.close(fig)


def write_case(
    inputs: pd.DataFrame,
    metadata: pd.DataFrame,
    models: list[str],
    case: str,
    output_dir: Path,
    bootstrap: int,
    seed: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    frame = prepare_case(inputs, models, case)
    meta, split = assign_quadrants(
        metadata[metadata.dataset.isin(frame.dataset)].copy()
    )
    frame = frame.merge(
        meta[["dataset", "quadrant"]], on="dataset", validate="many_to_one"
    )
    summary, stats = summarize_quadrants_elo(
        frame, models, bootstrap=bootstrap, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{case}.pdf"
    plot_cleveland_quadrants_elo(
        summary=summary,
        quad_to_ds_meta={
            q: meta[meta.quadrant.eq(q)].to_dict("records") for q in QUADRANT_LABELS
        },
        out_pdf=pdf,
    )
    frame.to_csv(output_dir / f"{case}_inputs.csv", index=False)
    meta.to_csv(output_dir / f"{case}_datasets.csv", index=False)
    summary.to_csv(output_dir / f"{case}_summary.csv", index=False)
    manifest = {
        **provenance,
        "case": case,
        "models": models,
        "n_models": len(models),
        "n_datasets": len(meta),
        "tasks": meta.task.value_counts().to_dict(),
        "quadrant_split": {**split, "pool": case, "equality": "high side"},
        "quadrant_counts": meta.quadrant.value_counts().to_dict(),
        "selection_counts": {
            str(k): int(v) for k, v in frame.metric_source.value_counts().items()
        },
        "elo_by_quadrant": stats,
        "figure": str(pdf),
        "figure_sha256": digest(pdf),
        "figure_style": FIGURE_STYLE,
        "policy": {
            "competitors": "one per model",
            "score_selection": "Complete Tuned test mean, then the model's complete Default mean. LinearModel and NeoLsSVM use Default.",
            "cv": "Require every outer fold for that variant, then average before Elo and imputation",
            "imputation": "Only pairs missing both own variants use the same dataset's complete RandomForest Default mean",
            "anchor": "RandomForest model competitor = 1000",
            "bootstrap": bootstrap,
            "seed": seed,
            "ties": "half win each",
            "panels": "Q1-Q4, four rows and one column",
            "figure_formats": ["pdf"],
        },
    }
    (output_dir / f"{case}_meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {pdf}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=BENCHMARK_PLAN)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED)
    parser.add_argument("--model-families", type=Path, default=MODEL_FAMILIES)
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ensure_results(args.results_dir)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    plan = load_benchmark(args.plan, args.processed_path)
    models = list(
        load_benchmark_models_config(
            args.model_families, plan_path=args.plan
        ).expected_models_binary_classification
    )
    plan_models = set(plan["models"])
    if set(models) != plan_models:
        raise ValueError("Model families and comparison plan define different models")
    datasets = {
        d: info
        for d, info in plan["datasets"].items()
        if "all_tasks" in args.cases or info["task"] in CLASSIFICATION_TASKS
    }
    needed_models = sorted(
        set().union(*(set(models_for_case(models, c)) for c in args.cases))
    )
    inputs, metadata = collect_model_inputs(
        datasets, needed_models, args.results_dir, args.processed_path
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(args.results_dir.resolve()),
        "plan": str(args.plan),
        "plan_sha256": digest(args.plan),
        "model_families_sha256": digest(args.model_families),
        "script_sha256": digest(
            Path(__file__).resolve().parents[1] / "figures/elo_regimes.py"
        ),
    }
    with threadpool_limits(limits=1):
        for case in dict.fromkeys(args.cases):
            write_case(
                inputs,
                metadata,
                models_for_case(models, case),
                case,
                args.output_dir,
                args.bootstrap,
                args.seed,
                provenance,
            )


if __name__ == "__main__":
    main()
