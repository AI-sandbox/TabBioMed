from pathlib import Path
import numpy as np


def plot_bars(
    table,
    frame,
    models,
    output,
    order=None,
    limits=None,
    transparent_bars=False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            "font.size": 16,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 13,
            "axes.linewidth": 1.0,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    indexed = table.set_index("competitor")
    partially_imputed = (
        set(frame.loc[frame.imputed.eq(True), "competitor"]) & set(indexed.index)
        if "imputed" in frame
        else set()
    )
    if order is None:
        strength = (
            table.assign(model=table.competitor.str.split("::").str[0])
            .groupby("model")
            .elo.max()
        )
        order = sorted(models, key=lambda m: (-strength.get(m, -1000000000.0), m))
    fig, ax = plt.subplots(figsize=(11, max(6.0, 0.42 * len(order))))
    colors = {"default": "#F58518", "tuned": "#4C78A8"}
    for i, model in enumerate(order):
        paired = all((f"{model}::{v}" in indexed.index for v in colors))
        for variant in ("tuned", "default"):
            key = f"{model}::{variant}"
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            facecolor = to_rgba(colors[variant], 0.42)
            if variant == "default":
                background = (
                    0.42 * np.array(to_rgb(colors["tuned"])) + 0.58
                    if paired
                    else np.ones(3)
                )
                facecolor = 0.9 * np.array(to_rgb(colors[variant])) + 0.1 * background
            ax.barh(
                i,
                row.elo,
                height=0.58 if variant == "tuned" else 0.34 if paired else 0.55,
                color=colors[variant] if transparent_bars else facecolor,
                alpha=(0.42 if variant == "tuned" else 0.9)
                if transparent_bars
                else None,
                hatch="///" if key in partially_imputed else None,
                edgecolor="0.25" if key in partially_imputed else "none",
                linewidth=0,
                zorder=1 if variant == "tuned" else 2,
            )
        for variant in ("tuned", "default"):
            key = f"{model}::{variant}"
            if key not in indexed.index:
                continue
            row = indexed.loc[key]
            shade = "0.45" if variant == "tuned" else "0.15"
            ax.plot(
                [row.elo_lo_95, row.elo_hi_95],
                [i] * 2,
                color=shade,
                lw=0.7,
                alpha=0.85 if variant == "tuned" else 0.95,
                marker="|",
                markersize=3,
                markeredgewidth=0.7,
                zorder=3,
            )
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.3, -0.7)
    ax.set_xlabel("Elo")
    ax.axvline(1000, color="0.25", ls="--", lw=1.4, zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.65", alpha=0.45, lw=0.8)
    ax.tick_params(axis="x", width=1.2, length=5)
    if limits is None:
        limits = (
            min(0, np.floor((table.elo_lo_95.min() - 60) / 100) * 100),
            np.ceil((table.elo_hi_95.max() + 60) / 100) * 100,
        )
    ax.set_xlim(*limits)
    legend_handles = [
        Line2D([], [], color="0.25", ls="--", lw=1.4, label="Random Forest = 1000"),
        Patch(facecolor=colors["tuned"], alpha=0.42, label="Tuned"),
        Patch(facecolor=colors["default"], alpha=0.9, label="Default"),
    ]
    if partially_imputed:
        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor="0.25",
                hatch="///",
                linewidth=0.5,
                label="Partially imputed",
            )
        )
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="0.8",
    )
    fig.tight_layout(pad=0.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return (order, limits)
