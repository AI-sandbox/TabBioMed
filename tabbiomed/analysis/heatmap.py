from __future__ import annotations
from pathlib import Path


def plot_heatmap(
    roster: list[str],
    matrix: list[list[float | None]],
    out_png: Path,
    *,
    sort_by_win_rate: bool = True,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
    reverse_columns: bool = True,
    row_label: str = "Model A: Winning Model",
    column_label: str = "Model B: Compared Model",
    colorbar_label: str = "Win Rate (%)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n = len(roster)
    arr = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in matrix],
        dtype=float,
    )
    if sort_by_win_rate:
        avg_win = np.nanmean(arr, axis=1)
        order = np.argsort(avg_win)[::-1]
    else:
        order = np.arange(n)
    roster_sorted = [roster[i] for i in order]
    arr_sorted = arr[np.ix_(order, order)]
    col_order = order[::-1] if reverse_columns else order
    roster_cols = [roster[i] for i in col_order]
    arr_sorted = arr[np.ix_(order, col_order)]
    masked = np.ma.masked_invalid(arr_sorted)
    cmap = LinearSegmentedColormap.from_list(
        "indigo_teal", ["#352a86", "#f7f7f7", "#0f9d8a"], N=256
    )
    cmap.set_bad(color="white")
    fig_w = max(13.5, 0.7 * n)
    fig_h = max(12.0, 0.7 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        masked, vmin=0, vmax=100, cmap=cmap, interpolation="nearest", aspect="equal"
    )
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(roster_cols, rotation=55, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(roster_sorted)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(column_label, labelpad=18)
    ax.set_ylabel(row_label, labelpad=18)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0, labelsize=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = arr_sorted[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if v <= 25 or v >= 78 else "#333333"
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label(colorbar_label, fontsize=18, labelpad=12)
    cbar.ax.tick_params(labelsize=15, length=0)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    for extension in formats:
        fig.savefig(out_png.with_suffix(f".{extension}"), dpi=400, bbox_inches="tight")
    plt.close(fig)
