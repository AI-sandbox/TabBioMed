"""Runtime figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-runtime-scaling-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabbiomed.analysis.common import (
    DEFAULT_INCLUDE_DATASETS,
    DEFAULT_MODEL_FAMILIES,
    TASKS_DEFAULT,
    BenchmarkModelsConfig,
    load_benchmark_models_config,
    load_include_datasets,
)
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, valid_split_identity
from tabbiomed.analysis.figures.elo_regimes import collect_model_inputs

OUTPUT = _OUTPUT_ROOT / "runtime"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "legend.title_fontsize": 15,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.8,
        "lines.markersize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)
MODEL_CLASS_ORDER = ["Baselines", "Tree-based", "Neural networks", "Foundation models"]
TIMING_CONFIG: dict[str, dict[str, str]] = {
    "train": {
        "file_prefix": "train_time",
        "y_label": "Training time (s)",
    },
    "inference": {
        "file_prefix": "inference_time",
        "y_label": "Inference time (s)",
    },
}
CLASS_AGG_PLOT_HEX: dict[str, str] = {
    "Baselines": "#4c4c4c",
    "Tree-based": "#0072B2",
    "Neural networks": "#E69F00",
    "Foundation models": "#CC79A7",
}
CLASS_CYCLE_HEX: list[str] = [
    "#1b9e77",
    "#7570b3",
    "#d95f02",
    "#66a61e",
    "#e7298a",
    "#a6761d",
    "#1f78b4",
    "#b15928",
    "#8dd3c7",
    "#bebada",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]


def _hex_to_rgba(h: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b = mcolors.to_rgb(h)
    return (float(r), float(g), float(b), float(alpha))


def class_cycle_color_map(
    sorted_models: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    n = len(CLASS_CYCLE_HEX)
    return {
        m: _hex_to_rgba(CLASS_CYCLE_HEX[i % n]) for i, m in enumerate(sorted_models)
    }


def color_for_model(
    model_name: str, cmap: dict[str, tuple[float, float, float, float]]
) -> tuple[float, float, float, float]:
    if model_name in cmap:
        return cmap[model_name]
    h = hash(model_name) % 10007 / 10007
    r, g, b = mcolors.hsv_to_rgb((h, 0.65, 0.8))
    return (float(r), float(g), float(b), 1.0)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return np.nan
    return float(value) if np.isfinite(value) and value >= 0 else np.nan


def read_selected_timing(selection: dict[str, Any], info: dict[str, Any]):
    path = Path(selection["report_path"])
    if digest(path) != selection["report_sha256"]:
        raise ValueError(f"Main report changed: {path}")
    timing_path = path.with_name("timing.json")
    payload = timing_path.read_bytes()
    timing = json.loads(payload)
    expected_header = dict(
        artifact_type="evaluation_timing_summary",
        schema_version=1,
        scope="main",
        model=selection["model"],
        dataset=selection["dataset"],
        report_sha256=selection["report_sha256"],
    )
    if any(
        (timing.get(key) != value for key, value in expected_header.items())
    ) or not valid_split_identity(timing, info):
        raise ValueError(
            f"Timing summary does not match its Main evaluation: {timing_path}"
        )
    units = timing.get("units", [])
    indexed = {unit["fold_id"]: unit for unit in units}
    expected_folds = set(range(info["n_folds"])) if info["n_folds"] else {None}
    if len(indexed) != len(units) or set(indexed) != expected_folds:
        raise ValueError(f"Unexpected timing folds: {timing_path}")
    row = dict(
        selection,
        timing_path=str(timing_path),
        timing_sha256=hashlib.sha256(payload).hexdigest(),
        train_seconds=np.nan,
        inference_seconds=np.nan,
        n_train_timed_folds=0,
        n_inference_timed_folds=0,
    )
    if selection["selected_variant"] == "failed":
        return (row, [])
    folds = []
    for fold, unit in indexed.items():
        block = unit[selection["selected_variant"]]
        if block.get("evaluation_status") != "success":
            raise ValueError(
                f"Selected timing variant has no successful evaluation: {timing_path}/{fold}"
            )
        fit = measured_seconds(block.get("final_fit_seconds"))
        validation = measured_seconds(block.get("final_validation_prediction_seconds"))
        train = measured_seconds(block.get("final_fit_and_validation_seconds"))
        if np.isfinite(train) and np.isfinite(fit) and np.isfinite(validation):
            if not np.isclose(train, fit + validation, rtol=1e-10, atol=1e-08):
                raise ValueError(
                    f"Final-fit timing components disagree: {timing_path}/{fold}"
                )
        folds.append(
            dict(
                model=selection["model"],
                dataset=selection["dataset"],
                fold=fold,
                selected_variant=selection["selected_variant"],
                train_seconds=train,
                final_fit_seconds=fit,
                final_validation_prediction_seconds=validation,
                inference_seconds=measured_seconds(
                    block.get("test_prediction_seconds")
                ),
                n_fitted=block.get("final_fit_samples"),
                n_test=unit["split_samples"]["test"],
            )
        )
    for field, count in [
        ("train_seconds", "n_train_timed_folds"),
        ("inference_seconds", "n_inference_timed_folds"),
    ]:
        values = np.array([fold[field] for fold in folds])
        row[count] = int(np.isfinite(values).sum())
        if np.isfinite(values).all():
            row[field] = float(values.mean())
    return (row, folds)


def collect_rows(
    *,
    results_dir: Path,
    processed_path: Path,
    include: frozenset[str],
    bm: BenchmarkModelsConfig,
    require_ready: bool,
    tasks: set[str] | None,
) -> pd.DataFrame:
    results_dir = Path(results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Consolidated Main results directory does not exist: {results_dir}"
        )
    plan = load_benchmark(BENCHMARK_PLAN, processed_path)
    models = sorted(bm.expected_models_regression)
    if set(include) != set(plan["datasets"]):
        raise ValueError("Runtime inputs differ from the configured dataset scope")
    if not require_ready:
        raise ValueError("Datasets must satisfy READY checks")
    inputs, metadata = collect_model_inputs(
        plan["datasets"], models, results_dir, processed_path
    )
    if tasks is not None:
        inputs = inputs[inputs.task.isin(tasks)].copy()
        metadata = metadata[metadata.task.isin(tasks)].copy()
    if len(inputs) != len(models) * len(metadata):
        raise ValueError("Incomplete model/dataset grid")
    with ThreadPoolExecutor(max_workers=8) as pool:
        collected = list(
            pool.map(
                lambda row: read_selected_timing(row, plan["datasets"][row["dataset"]]),
                inputs.to_dict("records"),
            )
        )
    frame = pd.DataFrame([row for row, _ in collected]).merge(
        metadata, on=["dataset", "task"], validate="many_to_one"
    )
    frame["model_class"] = frame.model.map(bm.model_family)
    if not frame.model_class.isin(MODEL_CLASS_ORDER).all():
        raise ValueError("Unexpected model family")
    frame.attrs["fold_timings"] = [fold for _, folds in collected for fold in folds]
    return frame


def bin_edges_from_datasets_with_results(
    num_samples_one_per_dataset: np.ndarray, n_bins_cap: int
) -> tuple[np.ndarray, int]:
    s = np.asarray(num_samples_one_per_dataset, dtype=float)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return (np.array([1.0, 10.0], dtype=float), 1)
    n_bins_cap = max(1, int(n_bins_cap))
    n_bins_eff = min(n_bins_cap, s.size)
    if s.size == 1:
        x = float(s[0])
        pad = max(x * 1e-06, 1.0)
        lo = max(x - pad, x * 0.5 if x > 0 else 1e-09)
        hi = x + pad
        return (np.array([lo, hi], dtype=float), 1)
    qs = np.linspace(0.0, 1.0, n_bins_eff + 1)
    log_s = np.log10(s)
    edges = (10.0 ** np.quantile(log_s, qs)).astype(float)
    edges[0], edges[-1] = (float(s.min()), float(s.max()))
    edges = np.unique(edges)
    if edges.size < 2:
        lo, hi = (float(np.min(s)), float(np.max(s)))
        if hi <= lo:
            hi = lo * 1.001
        return (np.array([lo, hi], dtype=float), 1)
    return (edges, edges.size - 1)


def bin_index_for_samples(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    x = np.asarray(values, dtype=float)
    idx = np.full(x.shape, -1, dtype=int)
    for i in range(n_bins):
        lo, hi = (edges[i], edges[i + 1])
        if i < n_bins - 1:
            m = (x >= lo) & (x < hi)
        else:
            m = (x >= lo) & (x <= hi)
        idx[m] = i
    return idx


def geometric_bin_centers(edges: np.ndarray) -> np.ndarray:
    return np.sqrt(edges[:-1].astype(float) * edges[1:].astype(float))


def aggregate_median_band(
    seconds: np.ndarray,
    bin_idx: np.ndarray,
    n_bins: int,
    low_pct: float,
    high_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.full(n_bins, np.nan)
    lo = np.full(n_bins, np.nan)
    hi = np.full(n_bins, np.nan)
    for b in range(n_bins):
        t = seconds[bin_idx == b]
        t = t[np.isfinite(t) & (t > 0)]
        if t.size == 0:
            continue
        med[b] = float(np.median(t))
        lo[b] = float(np.percentile(t, low_pct))
        hi[b] = float(np.percentile(t, high_pct))
    return (med, lo, hi)


def class_filename_slug(model_class: str) -> str:
    s = model_class.replace(" ", "_")
    s = re.sub("[^a-zA-Z0-9_.-]+", "_", s)
    return s.strip("_") or "class"


def _save_fig(fig: plt.Figure, out_stem: Path, formats: set[str]) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dataframe_for_timing(df_base: pd.DataFrame, timing: str) -> pd.DataFrame:
    if timing not in ("train", "inference"):
        raise ValueError(timing)
    df = df_base[df_base["model_class"].isin(MODEL_CLASS_ORDER)].copy()
    df = df[np.isfinite(df["num_samples"]) & (df["num_samples"] > 0)]
    src = "train_seconds" if timing == "train" else "inference_seconds"
    df["seconds"] = df[src].astype(float)
    df = df[np.isfinite(df["seconds"]) & (df["seconds"] > 0)]
    return df


def family_bin_summary(
    df: pd.DataFrame, edges: np.ndarray, timing: str, low_pct: float, high_pct: float
) -> pd.DataFrame:
    rows = []
    centers = geometric_bin_centers(edges)
    for family in MODEL_CLASS_ORDER:
        subset = df[df.model_class.eq(family)].copy()
        subset["bin"] = bin_index_for_samples(subset.num_samples.to_numpy(), edges)
        for index, center in enumerate(centers):
            group = subset[subset.bin.eq(index)]
            values = group.seconds.to_numpy()
            quantiles = (
                np.percentile(values, [low_pct, 50, high_pct])
                if len(values)
                else [np.nan] * 3
            )
            rows.append(
                dict(
                    timing=timing,
                    model_class=family,
                    bin=index,
                    sample_lo=edges[index],
                    sample_hi=edges[index + 1],
                    sample_center=center,
                    n_datasets=group.dataset.nunique(),
                    n_models=group.model.nunique(),
                    n_observations=len(values),
                    lower_seconds=quantiles[0],
                    median_seconds=quantiles[1],
                    upper_seconds=quantiles[2],
                )
            )
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", alpha=0.18, linewidth=0.7)
    ax.grid(False, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", width=1.0, length=5)
    ax.tick_params(axis="both", which="minor", width=0.6, length=2.5)


def draw_models_curves_on_ax(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    edges: np.ndarray,
    models: Iterable[str],
    model_colors: dict[str, tuple[float, float, float, float]],
    *,
    low_pct: float,
    high_pct: float,
) -> None:
    centers = geometric_bin_centers(edges)
    n_bins = len(edges) - 1
    for m in models:
        sub = df_plot[df_plot["model"].eq(m)]
        ns = sub["num_samples"].astype(float).to_numpy()
        sec = sub["seconds"].astype(float).to_numpy()
        ok = np.isfinite(ns) & (ns > 0) & np.isfinite(sec) & (sec > 0)
        ns, sec = (ns[ok], sec[ok])
        if ns.size == 0:
            continue
        bidx = bin_index_for_samples(ns, edges)
        valid = bidx >= 0
        bidx, sec = (bidx[valid], sec[valid])
        med, lo, hi = aggregate_median_band(sec, bidx, n_bins, low_pct, high_pct)
        has = np.isfinite(med)
        if not np.any(has):
            continue
        color = color_for_model(m, model_colors)
        idx_nonempty = np.flatnonzero(has)
        xc = centers[idx_nonempty]
        for k in range(len(idx_nonempty) - 1):
            i0 = int(idx_nonempty[k])
            i1 = int(idx_nonempty[k + 1])
            x1, x2 = (float(centers[i0]), float(centers[i1]))
            ax.fill(
                [x1, x2, x2, x1],
                [lo[i0], lo[i1], hi[i1], hi[i0]],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.plot(
            xc,
            med[idx_nonempty],
            color=color,
            linewidth=2.6,
            label=m,
            marker="o",
            markersize=5.5,
            markeredgewidth=0.8,
        )


def draw_class_aggregated_curves_on_ax(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    edges: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
) -> None:
    centers = geometric_bin_centers(edges)
    n_bins = len(edges) - 1
    for mc in MODEL_CLASS_ORDER:
        sub = df_plot[df_plot["model_class"] == mc]
        if sub.empty:
            continue
        ns = sub["num_samples"].astype(float).to_numpy()
        sec = sub["seconds"].astype(float).to_numpy()
        ok = np.isfinite(ns) & (ns > 0) & np.isfinite(sec) & (sec > 0)
        ns, sec = (ns[ok], sec[ok])
        if ns.size == 0:
            continue
        bidx = bin_index_for_samples(ns, edges)
        valid = bidx >= 0
        bidx, sec = (bidx[valid], sec[valid])
        med, lo, hi = aggregate_median_band(sec, bidx, n_bins, low_pct, high_pct)
        has = np.isfinite(med)
        if not np.any(has):
            continue
        color = _hex_to_rgba(CLASS_AGG_PLOT_HEX[mc])
        idx_nonempty = np.flatnonzero(has)
        xc = centers[idx_nonempty]
        for k in range(len(idx_nonempty) - 1):
            i0 = int(idx_nonempty[k])
            i1 = int(idx_nonempty[k + 1])
            x1, x2 = (float(centers[i0]), float(centers[i1]))
            ax.fill(
                [x1, x2, x2, x1],
                [lo[i0], lo[i1], hi[i1], hi[i0]],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.plot(
            xc,
            med[idx_nonempty],
            color=color,
            linewidth=3.0,
            label=mc,
            marker="o",
            markersize=6,
            markeredgewidth=0.9,
        )


def plot_class_panel(
    df_class: pd.DataFrame,
    edges: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
    log_y: bool,
    y_axis_label: str,
) -> plt.Figure:
    models = sorted(df_class["model"].unique())
    model_colors = class_cycle_color_map(models)
    n_models = len(models)
    fig_w = min(16, 10 + 0.35 * n_models)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    draw_models_curves_on_ax(
        ax, df_class, edges, models, model_colors, low_pct=low_pct, high_pct=high_pct
    )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(y_axis_label)
    style_axes(ax)
    ax.legend(loc="best", title="Model", frameon=False, borderpad=0.6, handlelength=2.0)
    fig.tight_layout()
    return fig


def plot_all_models_panel(
    df: pd.DataFrame,
    edges: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
    log_y: bool,
    y_axis_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    draw_class_aggregated_curves_on_ax(
        ax, df, edges, low_pct=low_pct, high_pct=high_pct
    )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(y_axis_label)
    style_axes(ax)
    ax.legend(
        loc="upper left", frameon=False, ncol=2, columnspacing=1.0, handlelength=1.5
    )
    fig.tight_layout()
    return fig


def parse_save_formats_arg(s: str) -> set[str]:
    allowed = frozenset({"png", "pdf"})
    parts = {p.strip().lower() for p in str(s).split(",") if p.strip()}
    bad = parts - allowed
    if bad:
        raise SystemExit(f"Unknown save format(s) {bad}; allowed: {sorted(allowed)}")
    if not parts:
        raise SystemExit("At least one of png,pdf is required for --save-formats.")
    return parts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Training and inference time versus sample count from consolidated results."
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--processed-path", type=Path, default=PROCESSED)
    p.add_argument("--include", type=Path, default=DEFAULT_INCLUDE_DATASETS)
    p.add_argument("--model-families", type=Path, default=DEFAULT_MODEL_FAMILIES)
    p.add_argument("--output-dir", "-o", type=Path, default=OUTPUT)
    p.add_argument(
        "--require-ready", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--n-bins", type=int, default=12)
    p.add_argument("--low-pct", type=float, default=25.0)
    p.add_argument("--high-pct", type=float, default=75.0)
    p.add_argument("--tasks", type=str, default=",".join(TASKS_DEFAULT))
    p.add_argument("--linear-y", action="store_true")
    p.add_argument("--timing", choices=("train", "inference", "both"), default="both")
    p.add_argument("--save-formats", type=str, default="png,pdf")
    p.add_argument(
        "--per-class",
        action="store_true",
        help="Also render the individual-model family panels.",
    )
    args = p.parse_args()
    ensure_results(args.results_dir)
    tasks = {t.strip().lower() for t in str(args.tasks).split(",") if t.strip()}
    bad = tasks - set(TASKS_DEFAULT)
    if bad:
        raise SystemExit(f"Unsupported task(s): {sorted(bad)}")
    save_formats = parse_save_formats_arg(args.save_formats)
    log_y = not bool(args.linear_y)
    n_bins_cap = max(1, min(40, int(args.n_bins)))
    low_pct = float(args.low_pct)
    high_pct = float(args.high_pct)
    if not 0 <= low_pct < high_pct <= 100:
        raise SystemExit("Require 0 <= --low-pct < --high-pct <= 100")
    include = load_include_datasets(args.include)
    bm = load_benchmark_models_config(args.model_families)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = collect_rows(
        results_dir=Path(args.results_dir),
        processed_path=args.processed_path,
        include=include,
        bm=bm,
        require_ready=bool(args.require_ready),
        tasks=tasks,
    )
    if df_all.empty:
        raise SystemExit(
            "No rows collected. Check results_dir/processed_path and filtering."
        )
    fold_timings = df_all.attrs.pop("fold_timings")
    pd.DataFrame(fold_timings).to_csv(out_dir / "timing_by_split.csv", index=False)
    df_all.sort_values(["model_class", "model", "dataset"]).to_csv(
        out_dir / "time_vs_samples_raw.csv", index=False
    )
    edges_all, _ = bin_edges_from_datasets_with_results(
        df_all.drop_duplicates("dataset").num_samples.to_numpy(), n_bins_cap
    )
    summaries = []
    timing_arg = str(args.timing)
    timings: tuple[str, ...] = (
        ("train", "inference") if timing_arg == "both" else (timing_arg,)
    )
    for timing in timings:
        tc = TIMING_CONFIG[timing]
        file_prefix = tc["file_prefix"]
        y_axis_label = tc["y_label"]
        df = dataframe_for_timing(df_all, timing)
        if df.empty:
            continue
        df.sort_values(["model_class", "model", "dataset"])[
            ["model", "model_class", "dataset", "task", "num_samples", "seconds"]
        ].to_csv(out_dir / f"{file_prefix}_vs_samples_raw.csv", index=False)
        for mc in MODEL_CLASS_ORDER if args.per_class else []:
            sub = df[df["model_class"] == mc].copy()
            if sub.empty:
                continue
            one_per_ds = sub.drop_duplicates(subset=["dataset"], keep="first")
            ns_ds = one_per_ds["num_samples"].astype(float).to_numpy()
            edges, _ = bin_edges_from_datasets_with_results(ns_ds, n_bins_cap)
            fig = plot_class_panel(
                sub,
                edges,
                low_pct=low_pct,
                high_pct=high_pct,
                log_y=log_y,
                y_axis_label=y_axis_label,
            )
            _save_fig(
                fig,
                out_dir / f"{file_prefix}_vs_samples_{class_filename_slug(mc)}",
                save_formats,
            )
        summaries.append(family_bin_summary(df, edges_all, timing, low_pct, high_pct))
        fig_all = plot_all_models_panel(
            df,
            edges_all,
            low_pct=low_pct,
            high_pct=high_pct,
            log_y=log_y,
            y_axis_label=y_axis_label,
        )
        _save_fig(
            fig_all, out_dir / f"{file_prefix}_vs_samples_all_families", save_formats
        )
    pd.concat(summaries, ignore_index=True).to_csv(
        out_dir / "family_bin_summary.csv", index=False
    )
    for row in df_all.itertuples():
        for path, expected in [
            (row.report_path, row.report_sha256),
            (row.timing_path, row.timing_sha256),
            (row.metadata_path, row.metadata_sha256),
        ]:
            if digest(Path(path)) != expected:
                raise ValueError(f"Input changed during analysis: {path}")
    manifest = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        results_root=str(RESULTS.resolve()),
        models=sorted(df_all.model.unique()),
        n_datasets=df_all.dataset.nunique(),
        n_model_dataset_pairs=len(df_all),
        selection_counts=df_all.metric_source.value_counts().to_dict(),
        timed_datasets={
            timing: dataframe_for_timing(df_all, timing).dataset.nunique()
            for timing in timings
        },
        bin_edges=edges_all.tolist(),
        policy=dict(
            selection="Complete Tuned result, then own complete Default result",
            training="One final fit plus its validation prediction, excluding HPO and supplementary refits",
            inference="Recorded held-out test prediction only",
            missing="Unavailable evaluation or clock stays missing, no timing imputation",
            cv="Require every outer timing fold, then average seconds within each dataset",
            aggregation="Pooled model/dataset observations within each family and sample-count bin",
            center="Median",
            band_percentiles=[low_pct, high_pct],
        ),
        source_hashes={
            str(path): digest(path)
            for path in [
                BENCHMARK_PLAN,
                args.model_families,
                Path(__file__).resolve().parents[1] / "figures/runtime.py",
                Path(__file__).resolve().parents[1] / "results.py",
                Path(__file__).resolve().parents[1] / "figures/elo_regimes.py",
            ]
        },
        outputs={
            p.name: digest(p)
            for p in out_dir.iterdir()
            if p.is_file() and p.name != "analysis.json"
        },
    )
    (out_dir / "analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["selection_counts"]), flush=True)
    print("Saved runtime scaling plots to", out_dir, flush=True)


if __name__ == "__main__":
    main()
