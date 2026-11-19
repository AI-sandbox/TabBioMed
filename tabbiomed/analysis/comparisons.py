from __future__ import annotations
import pandas as pd
from scipy.stats import kendalltau


def compare(
    full: pd.DataFrame, reduced: pd.DataFrame, score: str, higher: bool
) -> tuple[dict, pd.DataFrame]:
    if set(full.index) != set(reduced.index):
        raise ValueError("Ranking comparisons must use identical competitor sets")
    delta = full.add_prefix("full_").join(reduced.add_prefix("excluded_"))
    delta["rank_shift"] = delta.excluded_rank_position - delta.full_rank_position
    delta["abs_rank_shift"] = delta.rank_shift.abs()
    stat = kendalltau(delta.full_rank_position, delta.excluded_rank_position)
    order_full = full.sort_values(
        [score, full.index.name], ascending=[not higher, True]
    )
    order_reduced = reduced.sort_values(
        [score, reduced.index.name], ascending=[not higher, True]
    )
    top_full, top_reduced = (
        set(order_full.head(5).index),
        set(order_reduced.head(5).index),
    )
    summary = {
        "n_competitors": len(full),
        "kendall_tau": float(stat.statistic),
        "kendall_pvalue": float(stat.pvalue),
        "top5_overlap": len(top_full & top_reduced),
        "top5_full": list(order_full.head(5).index),
        "top5_excluded": list(order_reduced.head(5).index),
        "max_abs_rank_shift": float(delta.abs_rank_shift.max()),
        "largest_movers": list(
            delta.index[delta.abs_rank_shift.eq(delta.abs_rank_shift.max())]
        ),
        "winner_full": str(order_full.index[0]),
        "winner_excluded": str(order_reduced.index[0]),
    }
    for label, table in [("full", order_full), ("excluded", order_reduced)]:
        summary[f"tie_at_top5_boundary_{label}"] = bool(
            len(table) > 5 and table.iloc[4][score] == table.iloc[5][score]
        )
    return (summary, delta.sort_values("full_rank_position"))
