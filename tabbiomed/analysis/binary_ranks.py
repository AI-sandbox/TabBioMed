import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from .binary_metrics import ALT_METRICS, METRICS


def add_ranks(frame, metric):
    spec = next((s for s in METRICS if s["key"] == metric))
    result = frame.copy()
    values = pd.to_numeric(result[metric], errors="coerce")
    result[f"{metric}_rank"] = values.groupby(result.dataset).rank(
        ascending=not spec["higher_is_better"], method="average"
    )
    return result


def mean_rank_table(frame):
    ranked = frame.copy()
    for spec in METRICS:
        ranked = add_ranks(ranked, spec["key"])
    rows = []
    for model, group in ranked.groupby("model", sort=True):
        row = {"model": model}
        for spec in METRICS:
            key = spec["key"]
            scores = pd.to_numeric(group[key], errors="coerce")
            ranks = group[f"{key}_rank"]
            row.update(
                {
                    f"n_{key}": int(scores.notna().sum()),
                    f"mean_{key}": scores.mean(),
                    f"mean_rank_{key}": ranks.mean(),
                    f"median_rank_{key}": ranks.median(),
                }
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["mean_rank_roc_auc", "model"])
        .reset_index(drop=True)
    )


def agreement_tables(frame, threshold=2.0, top_k=3):
    rows = []
    frame = frame.sort_values(["model", "dataset"], kind="stable")
    for spec in ALT_METRICS:
        key = spec["key"]
        both = frame.dropna(subset=["roc_auc", key])
        both = add_ranks(add_ranks(both, "roc_auc"), key)
        for dataset, group in both.groupby("dataset", sort=True):
            roc, alt = (group.roc_auc_rank, group[f"{key}_rank"])
            n = len(group)
            roc_order = group.sort_values(
                ["roc_auc_rank", "roc_auc"], ascending=[True, False]
            )
            alt_order = group.sort_values(
                [f"{key}_rank", key], ascending=[True, not spec["higher_is_better"]]
            )
            roc_set = set(group.nsmallest(top_k, "roc_auc_rank").model)
            alt_set = set(group.nsmallest(top_k, f"{key}_rank").model)
            imbalance = group.imbalance_ratio.iloc[0]
            rows.append(
                dict(
                    metric=key,
                    label=spec["label"],
                    dataset=dataset,
                    imbalance_ratio=imbalance,
                    subset=("imbalanced" if imbalance > threshold else "balanced")
                    if pd.notna(imbalance)
                    else None,
                    n_models=n,
                    spearman_rho=float(spearmanr(roc, alt).statistic)
                    if n >= 3
                    else np.nan,
                    kendall_tau=float(kendalltau(roc, alt).statistic)
                    if n >= 3
                    else np.nan,
                    top1_flipped=roc_order.model.iloc[0] != alt_order.model.iloc[0],
                    roc_top1=roc_order.model.iloc[0],
                    alt_top1=alt_order.model.iloc[0],
                    top_k_jaccard=len(roc_set & alt_set) / len(roc_set | alt_set),
                )
            )
    detail = pd.DataFrame(rows)
    summary = []
    for spec in ALT_METRICS:
        part = detail[detail.metric.eq(spec["key"])]
        row = dict(
            metric=spec["key"],
            label=spec["label"],
            higher_is_better=spec["higher_is_better"],
            imbalance_threshold=threshold,
            n_datasets=len(part),
            median_spearman=part.spearman_rho.median(),
            median_kendall=part.kendall_tau.median(),
            top1_flips=int(part.top1_flipped.sum()),
            top1_flip_rate=part.top1_flipped.mean(),
            top_k=top_k,
            median_top_k_jaccard=part.top_k_jaccard.median(),
        )
        for subset in ("balanced", "imbalanced"):
            sub = part[part.subset.eq(subset)]
            row.update(
                {
                    f"n_{subset}": len(sub),
                    f"{subset}_top1_flips": int(sub.top1_flipped.sum()),
                    f"{subset}_top1_flip_rate": sub.top1_flipped.mean(),
                    f"{subset}_median_spearman": sub.spearman_rho.median(),
                }
            )
        summary.append(row)
    return (pd.DataFrame(summary), detail)
