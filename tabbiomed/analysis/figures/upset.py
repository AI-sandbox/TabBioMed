"""Upset figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.datasets import load_metadata
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, build_results
from tabbiomed.analysis.common import (
    DEFAULT_MODEL_FAMILIES,
    TASKS_DEFAULT,
    canon_domain,
    load_benchmark_models_config,
    rank_metric_ascending,
)

_REGRESSION_TASKS = frozenset({"regression", "multiregression"})
_CLASSIFICATION_TASKS = frozenset({"binary", "classification"})
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
CLASS_ORDER = ("Baselines", "Tree-based", "Neural networks", "Foundation models")
MODEL_CLASS_COLORS: dict[str, str] = {
    "Baselines": "#4c4c4c",
    "Tree-based": "#0072B2",
    "Neural networks": "#E69F00",
    "Foundation models": "#CC79A7",
}
DOMAIN_PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]
INTERSECTION_BAR_FACE = "#bfe8fc"
INTERSECTION_BAR_EDGE = "#7ebad6"
MATRIX_CONNECTOR_COLOR = "#4a4a4a"
INACTIVE_DOT_COLOR = "#d9d9d9"


def build_topk_by_raw_metric(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "task",
                "domain",
                "model_class",
                "model",
                "metric_value",
                "topk_rank",
            ]
        )
    out_rows: list[dict[str, Any]] = []
    for dataset, g in df.groupby("dataset", sort=False):
        task = str(g["task"].iloc[0])
        asc = rank_metric_ascending(task)
        ranked = g.sort_values(
            by=["metric_value", "model"], ascending=[asc, True], kind="mergesort"
        ).reset_index(drop=True)
        ranked["topk_rank"] = ranked.index + 1
        topk = ranked.head(max(1, int(top_k))).copy()
        for _, r in topk.iterrows():
            out_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "domain": str(r["domain"]),
                    "model_class": str(r["model_class"]),
                    "model": str(r["model"]),
                    "metric_value": float(r["metric_value"]),
                    "topk_rank": int(r["topk_rank"]),
                }
            )
    out = pd.DataFrame(out_rows)
    return out.sort_values(
        ["task", "dataset", "topk_rank", "model_class", "model"]
    ).reset_index(drop=True)


def _write_detail(df_detail: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_detail.to_csv(out_dir / f"{stem}.csv", index=False)


def _darken_edge(face_hex: str, factor: float = 0.42) -> str:
    r, g, b = mcolors.to_rgb(face_hex)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _lighten_row_shading(hex_color: str, mix_white: float = 0.88) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    return mcolors.to_hex(
        (
            r * (1 - mix_white) + mix_white,
            g * (1 - mix_white) + mix_white,
            b * (1 - mix_white) + mix_white,
        )
    )


def _style_upset_bars_and_totals(upset: Any) -> None:
    styler = getattr(upset, "style_categories", None)
    if callable(styler):
        for cat in CLASS_ORDER:
            hc = MODEL_CLASS_COLORS.get(cat, "#888888")
            styler(cat, bar_facecolor=hc, shading_facecolor=_lighten_row_shading(hc))
    for i in range(len(upset.subset_styles)):
        upset.subset_styles[i]["facecolor"] = INTERSECTION_BAR_FACE
        upset.subset_styles[i]["edgecolor"] = INTERSECTION_BAR_EDGE


def _pattern_tuple_from_index(idx: Any, n_cats: int) -> tuple[bool, ...]:
    if hasattr(idx, "__len__") and (not isinstance(idx, (str, bytes))):
        t = tuple(idx)
        if len(t) >= n_cats:
            return tuple((bool(x) for x in t[:n_cats]))
    return tuple((bool(idx[i]) for i in range(n_cats)))


def _build_pattern_domain_counts(
    ind: pd.DataFrame,
) -> dict[tuple[bool, ...], dict[str, int]]:
    from collections import Counter

    out: dict[tuple[bool, ...], Counter[str]] = {}
    for ds in ind.index.astype(str):
        pat = tuple((bool(ind.loc[ds, c]) for c in CLASS_ORDER))
        dom = str(ind.loc[ds, "_domain"]) if "_domain" in ind.columns else ""
        out.setdefault(pat, Counter())[dom] += 1
    return out


def _patch_intersection_bars_domain_stacked(
    upset_obj: Any,
    pattern_domain_counts: dict[tuple[bool, ...], dict[str, int]],
    domain_order: list[str],
    domain_colors: dict[str, str],
    inter_ax: Any | None,
    fig: plt.Figure | None,
) -> None:
    from collections import Counter

    intersections = getattr(upset_obj, "intersections", None)
    if intersections is None or len(intersections) == 0:
        return
    n_int = len(intersections)
    n_cats = len(CLASS_ORDER)
    best_rects: list[Any] = []
    if inter_ax is not None:
        cand = [p for p in inter_ax.patches if isinstance(p, Rectangle)]
        if len(cand) == n_int:
            best_rects = cand
    if not best_rects and fig is not None:
        for ax in fig.axes:
            rects = [p for p in ax.patches if isinstance(p, Rectangle)]
            if len(rects) == n_int:
                inter_ax = ax
                best_rects = rects
                break
    if inter_ax is None or not best_rects:
        return
    hs = [float(r.get_height()) for r in best_rects]
    ws = [float(r.get_width()) for r in best_rects]
    stack_vertical = max(hs) >= max(ws)
    rects = sorted(best_rects, key=lambda p: (p.get_x(), -p.get_y()))
    for t in list(inter_ax.texts):
        t.remove()
    ymax = max(float(inter_ax.get_ylim()[1]), 1.0)
    y_headroom = -0.008 * ymax
    xmax = max(float(inter_ax.get_xlim()[1]), 1.0)
    x_pad = 0.015 * xmax
    for rect, (idx, total) in zip(rects, intersections.items()):
        pat = _pattern_tuple_from_index(idx, n_cats)
        ctr: Counter[str] | None = pattern_domain_counts.get(pat)
        if ctr is None:
            for k, v in pattern_domain_counts.items():
                if tuple((bool(x) for x in k)) == pat:
                    ctr = v
                    break
            if ctr is None:
                ctr = Counter()
        x0 = float(rect.get_x())
        y0 = float(rect.get_y())
        w = float(rect.get_width())
        h = float(rect.get_height())
        rect.remove()
        tot = int(total) if hasattr(total, "__int__") else int(float(total))
        if tot <= 0:
            continue
        denom = sum((int(ctr.get(d, 0)) for d in domain_order))
        if stack_vertical:
            total_h = h
            if total_h <= 0 or w <= 0:
                continue
            if denom <= 0:
                inter_ax.bar(
                    x0,
                    total_h,
                    bottom=y0,
                    width=w,
                    align="edge",
                    color="#dddddd",
                    edgecolor="white",
                    linewidth=0.4,
                )
            else:
                bottom = y0
                for d in domain_order:
                    cnt = int(ctr.get(d, 0))
                    if cnt <= 0:
                        continue
                    seg_h = total_h * (cnt / float(denom))
                    inter_ax.bar(
                        x0,
                        seg_h,
                        bottom=bottom,
                        width=w,
                        align="edge",
                        facecolor=domain_colors.get(d, "#bbbbbb"),
                        edgecolor="white",
                        linewidth=0.45,
                    )
                    bottom += seg_h
            inter_ax.text(
                x0 + w / 2.0,
                y0 + total_h + y_headroom,
                str(tot),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="0.15",
            )
        else:
            total_w = w
            if total_w <= 0 or h <= 0:
                continue
            if denom <= 0:
                inter_ax.barh(
                    y0,
                    total_w,
                    left=x0,
                    height=h,
                    align="edge",
                    color="#dddddd",
                    edgecolor="white",
                    linewidth=0.4,
                )
            else:
                left = x0
                for d in domain_order:
                    cnt = int(ctr.get(d, 0))
                    if cnt <= 0:
                        continue
                    seg_w = total_w * (cnt / float(denom))
                    inter_ax.barh(
                        y0,
                        seg_w,
                        left=left,
                        height=h,
                        align="edge",
                        facecolor=domain_colors.get(d, "#bbbbbb"),
                        edgecolor="white",
                        linewidth=0.45,
                    )
                    left += seg_w
            inter_ax.text(
                x0 + total_w + x_pad,
                y0 + h / 2.0,
                str(tot),
                ha="left",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="0.15",
            )


def make_upset_plot(
    df_detail: pd.DataFrame, out_stem: Path, title: str | None, *, formats: set[str]
) -> bool:
    if df_detail.empty:
        return False
    from upsetplot import UpSet, from_indicators

    class UpSetTabBioMed(UpSet):
        def plot_matrix(self, ax):
            ax = self._reorient(ax)
            data = self.intersections
            n_cats = data.index.nlevels
            inclusion = data.index.to_frame().values
            cat_names = list(data.index.names)

            def dot_style(i: int, j: int) -> dict[str, Any]:
                if not inclusion[i, j]:
                    return {
                        "facecolor": self._other_dots_color,
                        "edgecolor": self._other_dots_color,
                        "linewidth": 0,
                    }
                name = (
                    str(cat_names[j])
                    if j < len(cat_names) and cat_names[j] is not None
                    else CLASS_ORDER[j]
                )
                fc = MODEL_CLASS_COLORS.get(name, "#888888")
                ec = _darken_edge(fc, factor=0.5)
                return {"facecolor": fc, "edgecolor": ec, "linewidth": 1.0}

            styles = [
                [dot_style(i, j) for j in range(n_cats)] for i in range(len(data))
            ]
            styles_flat = sum(styles, [])
            style_columns = {
                "facecolor": "facecolors",
                "edgecolor": "edgecolors",
                "linewidth": "linewidths",
                "linestyle": "linestyles",
                "hatch": "hatch",
            }
            styles_df = (
                pd.DataFrame(styles_flat)
                .reindex(columns=style_columns.keys())
                .astype(
                    {
                        "facecolor": "O",
                        "edgecolor": "O",
                        "linewidth": float,
                        "linestyle": "O",
                        "hatch": "O",
                    }
                )
            )
            styles_df["linewidth"] = styles_df["linewidth"].fillna(1)
            styles_df["facecolor"] = styles_df["facecolor"].fillna(self._facecolor)
            styles_df["edgecolor"] = styles_df["edgecolor"].fillna(
                styles_df["facecolor"]
            )
            styles_df["linestyle"] = styles_df["linestyle"].fillna("solid")
            del styles_df["hatch"]
            x = np.repeat(np.arange(len(data)), n_cats)
            y = np.tile(np.arange(n_cats), len(data))
            if self._element_size is not None:
                pt = (self._element_size * 0.42) ** 2
            else:
                pt = 260
            ax.scatter(
                *self._swapaxes(x, y),
                s=pt,
                zorder=10,
                **styles_df.rename(columns=style_columns),
            )
            if self._with_lines:
                idx = np.flatnonzero(inclusion)
                line_data = (
                    pd.Series(y[idx], index=x[idx])
                    .groupby(level=0)
                    .aggregate(["min", "max"])
                )
                ax.vlines(
                    line_data.index.values,
                    line_data["min"],
                    line_data["max"],
                    lw=2.5,
                    colors=MATRIX_CONNECTOR_COLOR,
                    zorder=5,
                )
            tick_axis = ax.yaxis
            tick_axis.set_ticks(np.arange(n_cats))
            tick_axis.set_ticklabels(
                data.index.names, rotation=0 if self._horizontal else -90
            )
            ax.xaxis.set_visible(False)
            ax.tick_params(axis="both", which="both", length=0)
            if not self._horizontal:
                ax.yaxis.set_ticks_position("top")
            ax.set_frame_on(False)
            ax.set_xlim(-0.5, x[-1] + 0.5, auto=False)
            ax.grid(False)

    ds_classes = (
        df_detail.groupby("dataset", as_index=False)["model_class"]
        .agg(lambda s: set(s.astype(str)))
        .rename(columns={"model_class": "class_set"})
    )
    ds_domain = df_detail.groupby("dataset", as_index=False)["domain"].first()
    if ds_classes.empty:
        return False
    ind = pd.DataFrame(
        False, index=ds_classes["dataset"].astype(str), columns=list(CLASS_ORDER)
    )
    for _, r in ds_classes.iterrows():
        ds = str(r["dataset"])
        cls_set = r["class_set"]
        for c in CLASS_ORDER:
            ind.loc[ds, c] = c in cls_set
    dom_map = {str(r["dataset"]): str(r["domain"]) for _, r in ds_domain.iterrows()}
    ind["_domain"] = ind.index.map(lambda d: dom_map.get(str(d), ""))
    pattern_domain_counts = _build_pattern_domain_counts(ind)
    ind_plot = ind.drop(columns=["_domain"])
    domains_sorted = sorted(
        set(df_detail["domain"].astype(str).unique().tolist())
        | {k for ctr in pattern_domain_counts.values() for k in ctr.keys()}
    )
    domain_colors = {
        d: DOMAIN_PALETTE[i % len(DOMAIN_PALETTE)] for i, d in enumerate(domains_sorted)
    }
    upset_data = from_indicators(list(CLASS_ORDER), ind_plot)
    fig = plt.figure(figsize=(12.5, 7.5))
    upset = UpSetTabBioMed(
        upset_data,
        subset_size="count",
        show_counts=True,
        show_percentages=False,
        sort_by="cardinality",
        sort_categories_by=None,
    )
    upset._other_dots_color = INACTIVE_DOT_COLOR
    _style_upset_bars_and_totals(upset)
    plot_out = upset.plot(fig=fig)
    totals_ax = plot_out.get("totals") if isinstance(plot_out, dict) else None
    if totals_ax is not None:
        totals_ax.grid(False)
    inter_ax = plot_out.get("intersections") if isinstance(plot_out, dict) else None
    if inter_ax is not None:
        inter_ax.grid(axis="y", color="0.82", linewidth=0.6)
    _patch_intersection_bars_domain_stacked(
        upset, pattern_domain_counts, domains_sorted, domain_colors, inter_ax, fig
    )
    legend_handles = [
        Patch(facecolor=domain_colors[d], edgecolor="0.35", label=str(d))
        for d in domains_sorted
    ]
    fig.legend(
        handles=legend_handles,
        title="Domain",
        loc="upper left",
        bbox_to_anchor=(0.12, 0.92),
        bbox_transform=fig.transFigure,
        fontsize=11,
        title_fontsize=12,
        frameon=False,
        borderaxespad=0.5,
    )
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=12, width=1.0)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        for txt in ax.texts:
            if txt.get_fontsize() < 12:
                txt.set_fontsize(12)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if "png" in formats:
        fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    if "pdf" in formats:
        fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def filter_datasets_with_all_models(
    df: pd.DataFrame, required_models: tuple[str, ...] | set[str]
) -> pd.DataFrame:
    required = set(required_models)
    if df.empty or not required:
        return df.iloc[0:0].copy()
    scoped = df[df["model"].astype(str).isin(required)].copy()
    keep: list[str] = []
    for dataset, group in scoped.groupby("dataset", sort=False):
        present = set(group["model"].astype(str))
        if required.issubset(present):
            keep.append(str(dataset))
    if not keep:
        return scoped.iloc[0:0].copy()
    return scoped[scoped["dataset"].astype(str).isin(keep)].copy()


def filter_task_aware_complete(
    df: pd.DataFrame,
    classification_models: tuple[str, ...],
    all_task_models: tuple[str, ...],
    optional_missing: set[str],
) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    classification_required = set(classification_models)
    all_task_required = set(all_task_models)
    keep: list[str] = []
    for dataset, group in df.groupby("dataset", sort=False):
        task = str(group["task"].iloc[0]).strip().lower()
        present = set(group["model"].astype(str))
        if task in _CLASSIFICATION_TASKS:
            if classification_required.issubset(present):
                keep.append(str(dataset))
        elif task in _REGRESSION_TASKS:
            if all_task_required.issubset(present):
                keep.append(str(dataset))
    if not keep:
        return df.iloc[0:0].copy()
    out = df[df["dataset"].astype(str).isin(keep)].copy()
    optional = set(optional_missing)
    if optional:
        drop = out["task"].astype(str).isin(_REGRESSION_TASKS) & out["model"].astype(
            str
        ).isin(optional)
        out = out.loc[~drop].copy()
    return out


def select_upset_scores(inputs: pd.DataFrame) -> pd.DataFrame:
    out = inputs.copy()
    default_only = out.metric_source.eq("default_only")
    use_tuned = ~default_only & out.tuned_metric.notna()
    use_default = ~use_tuned & out.default_metric.notna()
    out["metric_value"] = np.where(use_tuned, out.tuned_metric, out.default_metric)
    out["selected_variant"] = np.select(
        [use_tuned, use_default], ["tuned", "default"], default="missing"
    )
    out["metric_source"] = np.select(
        [use_tuned, use_default & default_only, use_default],
        ["tuned_test", "default_only", "default_fallback"],
        default="missing",
    )
    return out


def impute_upset_scores(inputs: pd.DataFrame) -> pd.DataFrame:
    out = inputs.copy()
    if out.duplicated(["model", "dataset"]).any():
        raise ValueError("Duplicate model/dataset inputs")
    illegal = out.task.isin(_REGRESSION_TASKS) & out.model.isin(
        load_benchmark_models_config().regression_optional_missing_models
    )
    if illegal.any():
        raise ValueError("Task-ineligible pairs must not enter UpSet imputation")
    reference = (
        out.loc[out.model.eq("RandomForest")].set_index("dataset").default_metric
    )
    out["imputed"] = out.metric_value.isna()
    fill = out.dataset.map(reference)
    if (out.imputed & fill.isna()).any():
        raise ValueError("Missing RF Default for an eligible missing score")
    out.loc[out.imputed, "metric_value"] = fill[out.imputed]
    out.loc[out.imputed, "metric_source"] = "rf_default_imputed"
    return out


def collect_inputs(processed_path, bm, tasks, require_ready=True, results_dir=RESULTS):
    plan = load_benchmark(BENCHMARK_PLAN, processed_path)
    domains = {}
    datasets = {}
    for name, info in plan["datasets"].items():
        if info["task"] not in tasks:
            continue
        metadata = load_metadata(name, processed_path)
        if metadata.get("task") != info["task"]:
            raise ValueError(f"Task metadata differs from benchmark plan: {name}")
        if require_ready and (not metadata.get("READY")):
            raise ValueError(f"Dataset is not READY: {name}")
        domain = canon_domain(metadata.get("label"))
        if domain == "Unlabeled":
            raise ValueError(f"Dataset has no biomedical domain: {name}")
        domains[name] = domain
        datasets[name] = info
    inputs = build_results(datasets, bm, root=Path(results_dir).expanduser().resolve())
    inputs["model_class"] = inputs.model.map(bm.model_family)
    inputs["domain"] = inputs.dataset.map(domains)
    if not inputs.model_class.isin(CLASS_ORDER).all():
        raise ValueError("Unrecognized model family")
    return select_upset_scores(inputs)


def _write_upset_figure(
    df_rows: pd.DataFrame,
    out_dir: Path,
    stem: str,
    *,
    top_k: int,
    title: str | None,
    formats: set[str],
) -> None:
    if df_rows.empty:
        print(f"[skip] {stem}: no rows")
        return
    detail = build_topk_by_raw_metric(df_rows, top_k=top_k)
    _write_detail(detail, out_dir, stem)
    wrote = make_upset_plot(detail, out_dir / stem, title=title, formats=formats)
    if not wrote:
        print(f"[skip] {stem}: plot not written")


def parse_variants_arg(s: str) -> set[str]:
    allowed = {"global", "per_task"}
    parts = {p.strip().lower() for p in s.split(",") if p.strip()}
    bad = parts - allowed
    if bad:
        raise SystemExit(f"Unknown variant(s) {bad}; allowed: {sorted(allowed)}")
    return parts


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS,
        help="Consolidated Main results root",
    )
    p.add_argument("--processed-path", type=Path, default=PROCESSED)
    p.add_argument("--model-families", type=Path, default=DEFAULT_MODEL_FAMILIES)
    p.add_argument("--output-dir", "-o", type=Path, default=_OUTPUT_ROOT / "upset")
    p.add_argument(
        "--require-ready", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--tasks", default=",".join(TASKS_DEFAULT))
    p.add_argument("--variants", default="global,per_task")
    p.add_argument("--save-formats", default="png,pdf")
    args = p.parse_args()
    results_root = ensure_results(args.results_dir)
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    if not tasks or tasks - set(TASKS_DEFAULT):
        p.error("Unsupported or empty task scope")
    if args.top_k < 1:
        p.error("--top-k must be positive")
    variants = parse_variants_arg(args.variants)
    formats = parse_save_formats_arg(args.save_formats)
    bm = load_benchmark_models_config(args.model_families)
    classification_models = bm.expected_models_binary_classification
    all_task_models = bm.expected_models_regression
    inputs = collect_inputs(
        args.processed_path, bm, tasks, args.require_ready, results_dir=results_root
    )
    observed = inputs.dropna(subset=["metric_value"]).copy()
    complete_all_tasks = filter_datasets_with_all_models(observed, all_task_models)
    complete_task_aware = filter_task_aware_complete(
        observed,
        classification_models,
        all_task_models,
        set(bm.regression_optional_missing_models),
    )
    imputed = impute_upset_scores(inputs)
    specs = [
        ("upset_rf_imputed", imputed),
        (f"upset_complete_{len(all_task_models)}", complete_all_tasks),
        (
            f"upset_complete_{len(classification_models)}_task_aware",
            complete_task_aware,
        ),
        (f"upset_observed_{len(classification_models)}", observed),
    ]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    inputs.to_csv(output / "input_metrics.csv", index=False)
    for stem, frame in specs:
        frame.to_csv(
            output / ("rows_" + stem.removeprefix("upset_") + ".csv"), index=False
        )
        print(
            f"{stem}: {frame.dataset.nunique()} datasets, {frame.model.nunique()} models",
            flush=True,
        )
        if "global" in variants:
            _write_upset_figure(
                frame, output, stem, top_k=args.top_k, title=None, formats=formats
            )
        if "per_task" in variants:
            for task in TASKS_DEFAULT:
                sub = frame.loc[frame.task.eq(task)]
                if not sub.empty:
                    _write_upset_figure(
                        sub,
                        output / "per_task" / task,
                        stem,
                        top_k=args.top_k,
                        title=None,
                        formats=formats,
                    )
    metadata = {
        "results_root": str(RESULTS),
        "selection": "Complete Tuned else complete Default per model/dataset, Default-only models use Default",
        "n_default_fallback": int(inputs.metric_source.eq("default_fallback").sum()),
        "cv": "Mean over every expected outer fold, incomplete variants have no score",
        "rf_imputation": "RF Default on the same dataset, task-eligible pairs only",
        "top_k": args.top_k,
        "ties": "Primary metric, then model name, take exactly top-k",
        "n_datasets": int(inputs.dataset.nunique()),
        "n_models": len(classification_models),
        "n_imputed": int(imputed.imputed.sum()),
        "dataset_counts": {stem: int(frame.dataset.nunique()) for stem, frame in specs},
        "script_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "figures/upset.py").read_bytes()
        ).hexdigest(),
        "benchmark_plan_sha256": hashlib.sha256(
            BENCHMARK_PLAN.read_bytes()
        ).hexdigest(),
        "model_families_sha256": hashlib.sha256(
            args.model_families.read_bytes()
        ).hexdigest(),
    }
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Wrote outputs under:", output, flush=True)


if __name__ == "__main__":
    main()
