from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range, wilcoxon
from statsmodels.stats.multitest import multipletests
from tabbiomed.analysis.common import (
    HIGHER_IS_BETTER,
    PRIMARY_METRIC,
    BenchmarkModelsConfig,
)

CD_TASK_ORDER = ["regression", "binary", "classification", "multiregression"]
ALPHA = 0.05
POOL_BINARY_CLASSIFICATION: tuple[str, ...] = ("binary", "classification")
POOL_REGRESSION: tuple[str, ...] = ("regression", "multiregression")
GLOBAL_CD_SCOPE_TAG = "global_all_tasks"


def _summary_dict_for_json(out: dict[str, Any]) -> dict[str, Any]:
    d = {
        k: v
        for k, v in out.items()
        if k not in ("ranks", "normalized_scores", "raw_scores")
    }
    wdr = d.pop("wilcoxon_diagram_avg_ranks", None)
    if wdr is not None:
        d["wilcoxon_diagram_mean_ranks_norm"] = {
            str(k): float(v) for k, v in wdr.items()
        }
    _strip_matrix_keys = frozenset({"p_value_matrix", "pmat", "wmat"})
    for key in ("nemenyi", "wilcoxon_holm"):
        sub = d.get(key)
        if isinstance(sub, dict):
            d[key] = {k: v for k, v in sub.items() if k not in _strip_matrix_keys}
    return d


def expected_models_for_cd_ranks(
    task: str, bm: BenchmarkModelsConfig
) -> tuple[str, ...]:
    if task in ("regression", "multiregression"):
        return bm.expected_models_regression
    return bm.expected_models_binary_classification


def _annotate_cd_grid_metadata(
    out: dict[str, Any], bm: BenchmarkModelsConfig, models: tuple[str, ...]
) -> None:
    out["expected_models"] = list(models)
    scope_tag = str(out.get("scope_tag") or "")
    if scope_tag in (
        "regression",
        "multiregression",
        "pooled_reg",
        GLOBAL_CD_SCOPE_TAG,
    ):
        out["use_reduced_regression_grid"] = True
        out["regression_optional_missing_models_excluded"] = sorted(
            bm.regression_optional_missing_models
        )
    if scope_tag == GLOBAL_CD_SCOPE_TAG:
        out["global_all_tasks"] = True
        out["rank_direction_note"] = (
            "Per-dataset average-tie ranks use normalized scores, with failures scored zero."
        )


def _scores_smallest_is_best(
    pivot: pd.DataFrame, task_per_dataset: pd.Series
) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=pivot.index, columns=pivot.columns, dtype=float)
    for idx in pivot.index:
        if idx not in task_per_dataset.index:
            continue
        task = str(task_per_dataset.loc[idx])
        primary = PRIMARY_METRIC.get(task, "roc_auc")
        row = pd.to_numeric(pivot.loc[idx], errors="coerce")
        if HIGHER_IS_BETTER.get(primary, False):
            out.loc[idx] = -row
        else:
            out.loc[idx] = row
    return out


def _mean_ranks_smallest_is_best(scores: pd.DataFrame) -> pd.DataFrame:
    arr = scores.to_numpy(dtype=float)
    r = rankdata(arr, axis=1)
    return pd.DataFrame(r, index=scores.index, columns=scores.columns)


def normalized_scores_with_failures(
    raw_scores: pd.DataFrame, task_per_dataset: pd.Series
) -> pd.DataFrame:
    raw_scores = raw_scores.replace([np.inf, -np.inf], np.nan)
    oriented = _scores_smallest_is_best(raw_scores, task_per_dataset)
    worst = oriented.max(axis=1)
    span = worst - oriented.min(axis=1)
    scaled = oriented.rsub(worst, axis=0).div(span.where(span > 0), axis=0)
    zero_span = (span == 0) & oriented.notna().any(axis=1)
    if zero_span.any():
        scaled.loc[zero_span] = oriented.loc[zero_span].notna().astype(float)
    return scaled.fillna(0.0)


def run_wilcoxon_holm_from_matrix(
    values: pd.DataFrame, models: tuple[str, ...], alpha: float
) -> dict[str, Any]:
    piv = values.reindex(columns=list(models))
    if not np.isfinite(piv.to_numpy(dtype=float)).all():
        return {"error": "Wilcoxon input matrix has non-finite scores"}
    n_datasets, n_models = piv.shape
    if n_datasets < 1:
        return {"error": "no datasets"}
    pairs = [(i, j) for i in range(n_models) for j in range(i + 1, n_models)]
    pvalues = []
    for i, j in pairs:
        differences = piv.iloc[:, i].to_numpy() - piv.iloc[:, j].to_numpy()
        pvalues.append(
            float(
                wilcoxon(
                    differences,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
            if np.any(differences != 0)
            else 1.0
        )
    adjusted = multipletests(pvalues, method="holm")[1] if pairs else []
    wmat = pd.DataFrame(np.eye(n_models), index=list(models), columns=list(models))
    for (i, j), pvalue in zip(pairs, adjusted):
        wmat.iloc[i, j] = wmat.iloc[j, i] = float(pvalue)
    significant = [[models[i], models[j]] for i, j in pairs if wmat.iloc[i, j] < alpha]
    return {
        "test": "wilcoxon_signed_rank_holm",
        "score_basis": "per_dataset_minmax_successes_failure_zero",
        "zero_difference_policy": "omit, with p=1 for an entirely tied pair",
        "alpha": alpha,
        "n_datasets": n_datasets,
        "n_models": n_models,
        "n_significant_pairs": len(significant),
        "n_pairwise_tests": len(pairs),
        "significant_pairs": significant,
        "p_value_matrix": wmat.to_dict(),
        "wmat": wmat,
    }


def run_nemenyi_friedman_block_scores(
    block_scores: pd.DataFrame, models: tuple[str, ...], alpha: float
) -> dict[str, Any]:
    import scikit_posthocs as sp

    x = block_scores.reindex(columns=list(models)).to_numpy(dtype=float)
    n_datasets, n_models = x.shape
    p_raw = sp.posthoc_nemenyi_friedman(x)
    pmat = pd.DataFrame(p_raw.to_numpy(), index=models, columns=models)
    q_alpha = float(
        studentized_range.ppf(1.0 - alpha, k=n_models, df=np.inf) / np.sqrt(2.0)
    )
    cd = float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets)))
    sig_pairs: list[list[str]] = []
    mlist = list(models)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            if float(pmat.iloc[i, j]) < alpha:
                sig_pairs.append([mlist[i], mlist[j]])
    rank_blocks = rankdata(x, axis=1)
    avg_ranks = pd.Series(rank_blocks.mean(axis=0), index=mlist).sort_values()
    p_nested: dict[str, dict[str, float]] = {}
    for col in pmat.columns:
        p_nested[str(col)] = {str(idx): float(pmat.loc[idx, col]) for idx in pmat.index}
    return {
        "cd": cd,
        "alpha": alpha,
        "n_datasets": n_datasets,
        "n_models": n_models,
        "n_significant_pairs": len(sig_pairs),
        "n_pairwise_tests": n_models * (n_models - 1) // 2,
        "significant_pairs": sig_pairs,
        "average_ranks": {str(k): float(v) for k, v in avg_ranks.items()},
        "p_value_matrix": p_nested,
        "pmat": pmat,
    }


def _upper_triangle_non_sig_pairs(
    n: int, non_sig: Callable[[int, int], bool]
) -> frozenset[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if non_sig(i, j):
                pairs.add((i, j))
    return frozenset(pairs)


def _cd_groups_from_upper_triangle_pairs(
    n: int, pairs: frozenset[tuple[int, int]]
) -> list[tuple[int, int, str]]:
    groups: list[tuple[int, int, str]] = []
    for i in range(n):
        tied_js = sorted((j for j in range(i + 1, n) if (i, j) in pairs))
        if not tied_js:
            continue
        prev = tied_js[0]
        seg_ends: list[int] = []
        for j in tied_js[1:]:
            if j == prev + 1:
                prev = j
            else:
                seg_ends.append(prev)
                prev = j
        seg_ends.append(prev)
        for seg_idx, end_j in enumerate(seg_ends):
            ls = "-" if seg_idx == 0 else ":"
            groups.append((i, end_j, ls))
    return groups


def _draw_rank_cd_paper_style(
    *,
    avg_ranks: pd.Series,
    groups: list[tuple[int, int, str]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    avg_ranks = avg_ranks.sort_values()
    models = avg_ranks.index.tolist()
    ranks = np.asarray(avg_ranks.values, dtype=float)
    n = len(models)
    if n == 0:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(12.0, 0.46 * n + 4.5)
    fig_h = max(4.6, 0.21 * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    top_y = n * 0.32 + 1.9
    ax.hlines(top_y, 1, n, color="black", lw=1.8)
    for r in range(1, n + 1):
        ax.vlines(r, top_y, top_y + 0.18, color="black", lw=1.5)
        ax.text(r, top_y + 0.28, str(r), ha="center", va="bottom", fontsize=18)
    ax.text(
        (n + 1) / 2,
        top_y + 0.72,
        "Average Rank",
        ha="center",
        va="bottom",
        fontsize=22,
        fontweight="bold",
    )
    split = n // 2
    left_models = models[:split]
    right_models = models[split:][::-1]
    rank_lookup = dict(zip(models, ranks))
    bar_y_start = top_y - 0.22
    bar_step = 0.14
    lowest_bar_y = bar_y_start - bar_step * max(0, len(groups) - 1)
    row_step = 0.28
    label_gap_below_bars = 0.25
    top_label_y = lowest_bar_y - label_gap_below_bars
    left_ys = top_label_y - np.arange(len(left_models)) * row_step
    right_ys = top_label_y - np.arange(len(right_models)) * row_step
    left_label_x = -0.45
    right_label_x = n + 0.45
    line_pad = 0.02
    ax.set_xlim(-1.25, n + 1.25)
    y_bottom_labels = float("inf")
    if len(left_ys) > 0:
        y_bottom_labels = min(y_bottom_labels, float(np.min(left_ys)))
    if len(right_ys) > 0:
        y_bottom_labels = min(y_bottom_labels, float(np.min(right_ys)))
    y_min = y_bottom_labels - 0.35 if np.isfinite(y_bottom_labels) else -0.45
    ax.set_ylim(float(y_min), float(top_y + 0.85))
    for model, y in zip(left_models, left_ys):
        x = float(rank_lookup[model])
        ax.plot([x, x], [top_y, y], color="black", lw=1.3)
        ax.plot([left_label_x + line_pad, x], [y, y], color="black", lw=1.3)
        ax.text(
            left_label_x,
            y,
            model,
            ha="right",
            va="center",
            fontsize=20,
            fontweight="normal",
        )
    for model, y in zip(right_models, right_ys):
        x = float(rank_lookup[model])
        ax.plot([x, x], [top_y, y], color="black", lw=1.3)
        ax.plot([x, right_label_x - line_pad], [y, y], color="black", lw=1.3)
        ax.text(
            right_label_x,
            y,
            model,
            ha="left",
            va="center",
            fontsize=20,
            fontweight="normal",
        )
    bar_y = bar_y_start
    for i, j, _ls in groups:
        m1 = models[i]
        m2 = models[j]
        x1 = float(rank_lookup[m1])
        x2 = float(rank_lookup[m2])
        if x2 < x1:
            x1, x2 = (x2, x1)
        ax.hlines(bar_y, x1, x2, color="black", lw=3.0, zorder=20, capstyle="butt")
        ax.vlines(
            [x1, x2], bar_y - 0.055, bar_y + 0.055, color="black", lw=2.0, zorder=21
        )
        bar_y -= bar_step
    ax.axis("off")
    plt.tight_layout(pad=0.25)
    suffix = str(out_path.suffix or "").lower()
    if suffix == ".pdf":
        fig.savefig(out_path, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cd_diagram_from_pmat(
    avg_ranks: pd.Series,
    *,
    cd: float,
    out_path: Path,
) -> None:
    avg_ranks = avg_ranks.sort_values()
    n = len(avg_ranks)
    if n == 0:
        return
    ranks = np.asarray(avg_ranks.values, dtype=float)

    def _nemenyi_non_sig(ii: int, jj: int) -> bool:
        if ii == jj:
            return False
        i, j = (ii, jj) if ii < jj else (jj, ii)
        return abs(float(ranks[i]) - float(ranks[j])) <= float(cd)

    pairs = _upper_triangle_non_sig_pairs(n, _nemenyi_non_sig)
    groups = _cd_groups_from_upper_triangle_pairs(n, pairs)
    _draw_rank_cd_paper_style(
        avg_ranks=avg_ranks,
        groups=groups,
        out_path=out_path,
    )


def _maximal_contiguous_non_sig_groups(
    models: list[str], wmat: pd.DataFrame, *, alpha: float
) -> list[tuple[int, int]]:
    n = len(models)
    w = wmat.reindex(index=models, columns=models)
    valid_groups: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ok = True
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    p = w.loc[models[a], models[b]]
                    if pd.isna(p) or float(p) < float(alpha):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid_groups.append((i, j))
    maximal: list[tuple[int, int]] = []
    for i, j in valid_groups:
        is_contained = False
        for a, b in valid_groups:
            if (a, b) != (i, j) and a <= i and (j <= b):
                is_contained = True
                break
        if not is_contained:
            maximal.append((i, j))
    maximal = sorted(set(maximal), key=lambda x: (-(x[1] - x[0]), x[0]))
    return maximal


def plot_wilcoxon_cd_style_diagram(
    avg_ranks: pd.Series,
    wmat: pd.DataFrame,
    *,
    alpha: float,
    out_path: Path,
) -> None:
    avg_ranks = avg_ranks.sort_values()
    n = len(avg_ranks)
    if n < 2:
        return
    models = avg_ranks.index.tolist()
    w = wmat.reindex(index=models, columns=models)
    groups = _maximal_contiguous_non_sig_groups(models, w, alpha=float(alpha))
    groups_with_ls = [(i, j, "-") for i, j in groups]
    _draw_rank_cd_paper_style(
        avg_ranks=avg_ranks,
        groups=groups_with_ls,
        out_path=out_path,
    )


def friedman_analysis(
    df_block: pd.DataFrame,
    models: tuple[str, ...],
    *,
    scope: str,
    scope_tag: str,
    pooled_task_names: tuple[str, ...] | None,
    task_per_dataset: pd.Series,
    primary_metric: str | None = None,
) -> dict[str, Any]:
    if df_block.duplicated(["dataset", "model"]).any():
        raise ValueError("Duplicate CD model/dataset results")
    if task_per_dataset.isna().any() or not task_per_dataset.isin(PRIMARY_METRIC).all():
        raise ValueError("Every CD dataset needs a recognized task")
    raw = df_block.pivot(index="dataset", columns="model", values="metric_value")
    raw = raw.reindex(index=task_per_dataset.index, columns=list(models)).astype(float)
    raw = raw.replace([np.inf, -np.inf], np.nan)
    normalized = normalized_scores_with_failures(raw, task_per_dataset)
    scores = -normalized
    ranks = _mean_ranks_smallest_is_best(scores)
    n_blocks, k = raw.shape
    failed = raw.isna()
    datasets_with_failures = raw.index[failed.any(axis=1)].tolist()
    source_counts = (
        df_block.loc[df_block["model"].isin(models), "metric_source"]
        .value_counts()
        .to_dict()
        if "metric_source" in df_block
        else {}
    )
    result = {
        "scope": scope,
        "scope_tag": scope_tag,
        "blocking": "all_datasets_tuned_default_failure_zero",
        "selection_rule": "complete Tuned test result, otherwise complete Default test result, otherwise failure",
        "cv_aggregation": "mean over every expected outer fold, before variant fallback",
        "failure_policy": "normalized score 0, tied with the worst successful model",
        "constant_row_policy": "all scores 0, including failures",
        "pooled_tasks": list(pooled_task_names) if pooled_task_names else None,
        "task": scope_tag if scope == "task" else None,
        "global_cd_all_tasks": scope == "global",
        "primary_metric": primary_metric or "mixed task primary metrics",
        "n_models": k,
        "df": max(0, k - 1),
        "alpha": ALPHA,
        "friedman_nemenyi_basis": "per_dataset_minmax_successes_failure_zero",
        "wilcoxon_basis": "norm_metric",
        "tests_independent": True,
        "n_datasets": n_blocks,
        "datasets": raw.index.tolist(),
        "n_datasets_complete": int((~failed.any(axis=1)).sum()),
        "n_datasets_with_failures": len(datasets_with_failures),
        "datasets_with_failures": datasets_with_failures,
        "n_datasets_excluded": 0,
        "n_failed_model_datasets": int(failed.to_numpy().sum()),
        "metric_source_counts": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "average_ranks": {
            str(key): float(value) for key, value in ranks.mean(axis=0).items()
        },
        "statistic": np.nan,
        "pvalue": np.nan,
        "significant_0.05": False,
        "nemenyi": None,
        "wilcoxon_holm": None,
        "wilcoxon_plot_note": "Per-dataset min-max scores, with failures assigned 0 and tied with the worst success.",
        "wilcoxon_diagram_avg_ranks": ranks.mean(axis=0),
        "ranks": ranks,
        "normalized_scores": normalized,
        "raw_scores": raw,
    }
    if n_blocks < 2 or k < 3:
        result["note"] = "Friedman needs at least two datasets and three models"
        return result
    if np.all(np.ptp(scores.to_numpy(), axis=1) == 0):
        statistic, pvalue = (0.0, 1.0)
    else:
        statistic, pvalue = friedmanchisquare(
            *(scores[model].to_numpy() for model in models)
        )
    result.update(
        statistic=float(statistic),
        pvalue=float(pvalue),
        nemenyi=run_nemenyi_friedman_block_scores(scores, models, ALPHA),
        wilcoxon_holm=run_wilcoxon_holm_from_matrix(normalized, models, ALPHA),
    )
    result["significant_0.05"] = bool(pvalue < ALPHA)
    return result


def friedman_for_task(df_task: pd.DataFrame, models: tuple[str, ...]) -> dict[str, Any]:
    task = str(df_task["task"].iloc[0]) if not df_task.empty else ""
    return friedman_analysis(
        df_task,
        models,
        scope="task",
        scope_tag=task,
        pooled_task_names=None,
        task_per_dataset=df_task.groupby("dataset")["task"].first(),
        primary_metric=PRIMARY_METRIC.get(task),
    )


def friedman_global_all_tasks(
    df: pd.DataFrame, bm: BenchmarkModelsConfig
) -> dict[str, Any]:
    models = bm.expected_models_regression
    sub = df[df["model"].isin(models)].copy()
    return friedman_analysis(
        sub,
        models,
        scope="global",
        scope_tag=GLOBAL_CD_SCOPE_TAG,
        pooled_task_names=None,
        task_per_dataset=df.groupby("dataset")["task"].first(),
    )


def friedman_pooled_tasks(
    df: pd.DataFrame, bm: BenchmarkModelsConfig, tasks: tuple[str, ...], scope_tag: str
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("Cannot pool an empty task list")
    models = expected_models_for_cd_ranks(tasks[0], bm)
    if any((expected_models_for_cd_ranks(task, bm) != models for task in tasks[1:])):
        raise ValueError("Pooled tasks must share the same supported models")
    sub = df[df["task"].isin(tasks)].copy()
    return friedman_analysis(
        sub,
        models,
        scope="pooled",
        scope_tag=scope_tag,
        pooled_task_names=tasks,
        task_per_dataset=sub.groupby("dataset")["task"].first(),
        primary_metric=PRIMARY_METRIC.get(tasks[0]),
    )


def _print_and_plot_cd_block(
    out: dict[str, Any],
    ranks: pd.DataFrame,
    *,
    plot_title_task: str,
    file_stem: str,
    plots_dir: Path | None,
    k_models: int,
) -> None:
    n_ds = out["n_datasets"]
    print(f"=== {plot_title_task}: {n_ds} datasets, {k_models} models ===")
    print(f"  Datasets excluded: {out['n_datasets_excluded']}")
    print(
        f"  Failed model/dataset results assigned zero: {out['n_failed_model_datasets']}"
    )
    print(f"  Score sources: {out['metric_source_counts']}")
    print(f"  Friedman statistic: {out['statistic']}, p-value: {out['pvalue']}")
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for key in ("raw_scores", "normalized_scores", "ranks"):
            out[key].to_csv(plots_dir / f"{file_stem}_{key}.csv", index_label="dataset")
        pd.Series(out["average_ranks"], name="average_rank").sort_values().to_csv(
            plots_dir / f"{file_stem}_average_ranks.csv", index_label="model"
        )
    nem, wh = (out.get("nemenyi"), out.get("wilcoxon_holm"))
    if isinstance(nem, dict) and (not nem.get("error")):
        matrix = nem.pop("pmat", None)
        if plots_dir is not None:
            matrix.to_csv(
                plots_dir / f"{file_stem}_nemenyi_pvalues.csv", index_label="model"
            )
            for ext in (".png", ".pdf"):
                plot_cd_diagram_from_pmat(
                    pd.Series(nem["average_ranks"]),
                    cd=nem["cd"],
                    out_path=plots_dir / f"cd_diagram_{file_stem}{ext}",
                )
    if isinstance(wh, dict) and (not wh.get("error")):
        matrix = wh.pop("wmat", None)
        if plots_dir is not None:
            matrix.to_csv(
                plots_dir / f"{file_stem}_wilcoxon_holm_pvalues.csv",
                index_label="model",
            )
            for ext in (".png", ".pdf"):
                plot_wilcoxon_cd_style_diagram(
                    out["wilcoxon_diagram_avg_ranks"],
                    matrix,
                    alpha=ALPHA,
                    out_path=plots_dir
                    / f"wilcoxon_cd_diagram_{file_stem}_wilcoxon_norm_metric{ext}",
                )
    for model, value in ranks.mean(axis=0).sort_values().items():
        print(f"    {model:20s} {value:8.4f}")
    print()


def run_cd_pipeline(
    df: pd.DataFrame,
    bm: BenchmarkModelsConfig,
    *,
    plots_dir: Path | None,
    output_json: Path | None,
) -> list[dict]:
    print(
        "Consolidated Main results only. Tuned, otherwise Default, otherwise failure."
    )
    print("CV variants require all outer folds, averaged before fallback.")
    print(
        "All datasets retained, all tests use per-dataset min-max scores with failures at zero."
    )
    analyses = []
    for task in CD_TASK_ORDER:
        sub = df[df["task"] == task].copy()
        if not sub.empty:
            models = expected_models_for_cd_ranks(task, bm)
            analyses.append((friedman_for_task(sub, models), models))
    for tag, tasks in (
        ("pooled_bc_cls", POOL_BINARY_CLASSIFICATION),
        ("pooled_reg", POOL_REGRESSION),
    ):
        models = expected_models_for_cd_ranks(tasks[0], bm)
        analyses.append((friedman_pooled_tasks(df, bm, tasks, tag), models))
    analyses.append(
        (friedman_global_all_tasks(df, bm), bm.expected_models_regression)
    )
    summaries = []
    for out, models in analyses:
        _annotate_cd_grid_metadata(out, bm, models)
        _print_and_plot_cd_block(
            out,
            out["ranks"],
            plot_title_task=out["scope_tag"],
            file_stem=out["scope_tag"],
            plots_dir=plots_dir,
            k_models=len(models),
        )
        summaries.append(_summary_dict_for_json(out))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"Wrote summary JSON to {output_json}")
    return summaries
