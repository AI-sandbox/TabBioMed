from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

DISTINCT_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#c51b7d",
    "#b15928",
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
]
MARKERS = {"default": "o", "tuned": "^"}
NORMALIZED_YTICKS = (0.0, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass
class Style:
    figsize: tuple[float, float] = (18.0, 11.6)
    tick: float = 20.0
    axis_label: float = 22.5
    legend: float = 18.5
    legend_title: float = 19.5
    label: float = 17.0
    badge: float = 17.0
    key: float = 17.5
    key_title: float = 18.5
    arrow_text: float = 16.0
    s_front: float = 265.0
    s_other: float = 175.0
    alpha_front: float = 1.0
    alpha_other: float = 1.0
    edge_front: float = 1.0
    edge_other: float = 0.6
    front_lw: float = 2.1
    front_dash: tuple = (0, (5, 4))
    front_color: str = "0.15"
    label_min_contrast: float = 3.0
    label_colors: dict = field(default_factory=dict)
    front_emphasis: str = "none"
    arrow_color: str = "#1565c0"
    arrow_kind: str = "block"
    arrow_head_w: float = 3.8
    legend_frame: bool = False
    legend_gap: float = 0.035
    legend_colspacing: float | None = None
    legend_fit: str = "shrink"
    legend_center_last: bool = False
    legend_right_col: tuple = ()
    variant_below: bool = False
    variant_gap: float = 0.012
    leader_cost: float = 0.0
    line_cost: float = 3000.0
    label_pins: dict = field(default_factory=dict)
    pad_left: tuple | None = None
    font_serif: tuple | None = None
    variant_legend: float | None = None
    center_frames: bool = False
    marker_z: dict = field(default_factory=dict)
    margins: dict = field(
        default_factory=lambda: dict(
            left=0.068, right=0.99, top=0.765, bottom=0.085, wspace=0.07
        )
    )


def model_colors(models: list[str]) -> dict[str, str]:
    return {
        m: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
        for i, m in enumerate(sorted(models))
    }


def variant_tag(variant: str) -> str:
    return "T" if str(variant) == "tuned" else "D"


def default_only_models(panels):
    variants = {}
    for panel in panels:
        for row in panel.summary.itertuples():
            variants.setdefault(str(row.model), set()).add(str(row.variant))
    return {model for model, values in variants.items() if values == {"default"}}


def front_label(model: str, variant: str, default_only) -> str:
    if variant == "default" and model in default_only:
        return model
    return f"{model}-{variant_tag(variant)}"


def _rel_luminance(rgb: np.ndarray) -> float:
    def ch(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (ch(float(c)) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_text_color(color: str, min_contrast: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    for _ in range(60):
        if 1.05 / (_rel_luminance(rgb) + 0.05) >= min_contrast:
            break
        rgb = rgb * 0.94
    return tuple((float(c) for c in rgb))


def apply_compressed_normalized_y(ax, ylim: tuple[float, float]) -> None:
    y_cut, compress = (0.6, 0.22)

    def fwd(y):
        y = np.asarray(y, dtype=float)
        return np.where(y <= y_cut, y * compress, y_cut * compress + (y - y_cut))

    def inv(y):
        y = np.asarray(y, dtype=float)
        return np.where(
            y <= y_cut * compress, y / compress, y_cut + (y - y_cut * compress)
        )

    ax.set_yscale("function", functions=(fwd, inv))
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(list(NORMALIZED_YTICKS)))
    ax.yaxis.set_minor_locator(FixedLocator([]))


def shared_y_config(
    metric: str, ys: np.ndarray, *, headroom: float
) -> tuple[tuple[float, float], list[float]]:
    ys = np.asarray(ys, dtype=float)
    ys = ys[np.isfinite(ys)]
    if metric == "norm":
        top = max(1.03, float(ys.max()) + headroom)
        return ((-0.03, top), list(NORMALIZED_YTICKS))
    lo, hi = (float(ys.min()), float(ys.max()))
    span = hi - lo if hi > lo else 1.0
    bottom = lo - 0.06 * span
    top = hi + max(0.05 * span, headroom * span)
    ticks = MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]).tick_values(bottom, hi)
    ticks = [float(t) for t in ticks if bottom <= t <= hi + 0.02 * span]
    return ((bottom, top), ticks)


def log_x_limits(
    xs: np.ndarray, *, pad_left: float, pad_right: float
) -> tuple[float, float]:
    lx = np.log10(np.asarray(xs, dtype=float))
    return (10 ** (float(lx.min()) - pad_left), 10 ** (float(lx.max()) + pad_right))


def style_axes(ax, *, x_label: str, y_label: str | None, style: Style) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(x_label, fontsize=style.axis_label)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=style.axis_label)
    ax.tick_params(
        axis="both", which="major", labelsize=style.tick, length=6, width=1.1
    )
    ax.tick_params(axis="x", which="minor", length=3, width=0.8)
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.6,
        color="0.88",
        zorder=0,
    )
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)


def _grown(s: float, pad_pt: float) -> float:
    return (math.sqrt(s) + 2.0 * pad_pt) ** 2


def front_pad_pt(style: Style) -> float:
    return {"none": 0.0, "thick": 0.6, "halo": 4.5, "ring": 4.2, "glow": 5.0}[
        style.front_emphasis
    ]


def scatter_points(
    ax, d: pd.DataFrame, cols: dict[str, str], x_col: str, y_col: str, style: Style
) -> None:
    emph = style.front_emphasis
    front_s = style.s_front * (1.12 if emph == "thick" else 1.0)
    front_lw = {"thick": 2.2}.get(emph, style.edge_front)
    for _, r in d.iterrows():
        on_front = bool(r["is_pareto"])
        xy = ([float(r[x_col])], [float(r[y_col])])
        marker = MARKERS.get(str(r["variant"]), "o")
        if on_front and emph == "halo":
            ax.scatter(
                *xy,
                marker=marker,
                s=_grown(front_s, 4.5),
                color="white",
                edgecolors="none",
                zorder=3.5,
            )
        if on_front and emph == "glow":
            tint = 0.3 * np.array(mcolors.to_rgb(cols[str(r["model"])])) + 0.7
            ax.scatter(
                *xy,
                marker="o",
                s=_grown(front_s, 5.0),
                color=[tint],
                edgecolors="none",
                zorder=3.5,
            )
        if on_front and emph == "ring":
            ax.scatter(
                *xy,
                marker=marker,
                s=_grown(front_s, 4.2),
                color="0.15",
                edgecolors="none",
                zorder=3.4,
            )
            ax.scatter(
                *xy,
                marker=marker,
                s=_grown(front_s, 2.8),
                color="white",
                edgecolors="none",
                zorder=3.6,
            )
        ax.scatter(
            *xy,
            c=[cols[str(r["model"])]],
            marker=marker,
            s=front_s if on_front else style.s_other,
            edgecolors="black" if on_front else "0.25",
            linewidths=front_lw if on_front else style.edge_other,
            alpha=style.alpha_front if on_front else style.alpha_other,
            zorder=(4 if on_front else 3)
            + style.marker_z.get(str(r.get("competitor", "")), 0.0),
        )


def draw_staircase_front(
    ax,
    front: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    style: Style,
) -> np.ndarray:
    f = front.sort_values(x_col)
    xs = f[x_col].to_numpy(dtype=float)
    ys = f[y_col].to_numpy(dtype=float)
    px, py = ([xs[0]], [ylim[0]])
    for i in range(len(xs)):
        px.append(xs[i])
        py.append(ys[i])
        if i + 1 < len(xs):
            px.append(xs[i + 1])
            py.append(ys[i])
    px.append(xlim[1])
    py.append(ys[-1])
    ax.plot(
        px,
        py,
        color=style.front_color,
        linewidth=style.front_lw,
        linestyle=style.front_dash,
        dash_capstyle="butt",
        zorder=2,
        clip_on=True,
    )
    return np.column_stack([px, py])


OPTIMAL_CUES = ("block", "outline", "line", "compass", "star", "gradient")


def draw_optimal_cue(
    ax, *, style: Style, kind: str | None = None, color: str | None = None
) -> np.ndarray:
    kind = kind or style.arrow_kind
    color = color or style.arrow_color
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    fs = style.arrow_text
    renderer = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "Optimal", fontsize=fs, fontweight="bold", transform=None)
    text_w, text_h = (
        probe.get_window_extent(renderer).width,
        probe.get_window_extent(renderer).height,
    )
    probe.remove()
    bb = ax.get_window_extent()
    inset = 0.02 * min(bb.width, bb.height)
    corner = np.array([bb.x0 + inset, bb.y1 - inset])
    unit = np.array([-1.0, 1.0]) / math.sqrt(2.0)

    def to_axes(q):
        return ((q[0] - bb.x0) / bb.width, (q[1] - bb.y0) / bb.height)

    def box(points, pad):
        pts = np.asarray(points)
        lo, hi = (pts.min(0) - pad, pts.max(0) + pad)
        return np.array([lo[0], lo[1], hi[0], hi[1]])

    if kind in ("block", "outline"):
        tw, hw, hl = (1.75 * fs, style.arrow_head_w * fs, 2.4 * fs)
        head_px = hl * px_per_pt
        length = text_w + head_px + 0.9 * fs * px_per_pt
        tip = corner
        tail = tip - unit * length
        filled = kind == "block"
        ax.add_patch(
            FancyArrowPatch(
                posA=to_axes(tail),
                posB=to_axes(tip),
                transform=ax.transAxes,
                arrowstyle=f"simple,tail_width={tw},head_width={hw},head_length={hl}",
                mutation_scale=1.0,
                facecolor=color if filled else "white",
                edgecolor="white" if filled else color,
                linewidth=1.2 if filled else 1.8,
                zorder=7,
                clip_on=False,
            )
        )
        mid = tail + unit * (length - head_px) * 0.5
        ax.text(
            *to_axes(mid),
            "Optimal",
            transform=ax.transAxes,
            rotation=-45,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color="white" if filled else color,
            zorder=8,
        )
        return box([tip, tail], hw * px_per_pt * 0.5)
    if kind == "line":
        length = 1.15 * text_w
        tip = corner
        tail = tip - unit * length
        ax.annotate(
            "",
            xy=to_axes(tip),
            xytext=to_axes(tail),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.7,head_width=0.38",
                color=color,
                lw=2.4,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=fs * 1.3,
            ),
            zorder=8,
        )
        t_anchor = tail + np.array([0.25 * fs * px_per_pt, -0.15 * fs * px_per_pt])
        ax.text(
            *to_axes(t_anchor),
            "Optimal",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fs,
            fontweight="bold",
            color=color,
            zorder=8,
        )
        far = t_anchor + np.array([text_w, -text_h])
        return box([tip, tail, far], 0.5 * fs * px_per_pt)
    if kind == "compass":
        arm = 1.25 * text_w
        origin = corner + np.array([arm, -arm * 0.85])
        left_tip = np.array([corner[0], origin[1]])
        up_tip = np.array([origin[0], corner[1]])
        for tip in (left_tip, up_tip):
            ax.annotate(
                "",
                xy=to_axes(tip),
                xytext=to_axes(origin),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.7,head_width=0.38",
                    color=color,
                    lw=2.2,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=fs * 1.2,
                ),
                zorder=8,
            )
        ax.add_patch(
            Circle(
                to_axes(origin),
                radius=2.6 * px_per_pt / bb.width,
                transform=ax.transAxes,
                color=color,
                zorder=8,
            )
        )
        ax.text(
            *to_axes((left_tip + origin) / 2 + np.array([0, -0.35 * fs * px_per_pt])),
            "faster",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fs * 0.95,
            color=color,
            fontweight="bold",
            zorder=8,
        )
        ax.text(
            *to_axes((up_tip + origin) / 2 + np.array([0.35 * fs * px_per_pt, 0])),
            "better",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fs * 0.95,
            color=color,
            fontweight="bold",
            zorder=8,
        )
        return box(
            [corner, origin + np.array([0.4 * fs * px_per_pt + text_w, -1.4 * text_h])],
            4,
        )
    if kind == "star":
        star = corner + np.array([0.9 * fs, -0.9 * fs]) * px_per_pt
        ax.scatter(
            *to_axes(star),
            transform=ax.transAxes,
            marker="*",
            s=(2.4 * fs) ** 2,
            color=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=8,
            clip_on=False,
        )
        t_anchor = star + np.array([1.3 * fs * px_per_pt, 0])
        ax.text(
            *to_axes(t_anchor),
            "Optimal",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=color,
            zorder=8,
        )
        return box([corner, t_anchor + np.array([text_w, -text_h])], 4)
    if kind == "gradient":
        ny, nx = (120, 200)
        yy, xx = np.mgrid[0 : 1 : ny * 1j, 0 : 1 : nx * 1j]
        dist = np.hypot(xx, 1 - yy)
        rgba = np.zeros((ny, nx, 4))
        rgba[..., :3] = mcolors.to_rgb(color)
        rgba[..., 3] = 0.22 * np.clip(1 - dist, 0, 1) ** 1.6
        ax.imshow(
            rgba,
            extent=(0, 0.42, 0.55, 1.0),
            transform=ax.transAxes,
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            zorder=0.5,
        )
        ax.text(
            *to_axes(corner + np.array([0.3 * fs, -0.3 * fs]) * px_per_pt),
            "Optimal",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fs,
            fontweight="bold",
            fontstyle="italic",
            color=color,
            zorder=8,
        )
        return box(
            [
                corner,
                corner
                + np.array(
                    [text_w + 0.6 * fs * px_per_pt, -text_h - 0.6 * fs * px_per_pt]
                ),
            ],
            4,
        )
    raise ValueError(f"unknown optimal cue {kind!r}; choose from {OPTIMAL_CUES}")


def _box_overlap(B: np.ndarray, O: np.ndarray) -> np.ndarray:
    if len(O) == 0:
        return np.zeros((len(B), 0))
    dx = np.minimum(B[:, None, 2], O[None, :, 2]) - np.maximum(
        B[:, None, 0], O[None, :, 0]
    )
    dy = np.minimum(B[:, None, 3], O[None, :, 3]) - np.maximum(
        B[:, None, 1], O[None, :, 1]
    )
    return np.clip(dx, 0, None) * np.clip(dy, 0, None)


def _box_point_dist(B: np.ndarray, P: np.ndarray) -> np.ndarray:
    dx = np.maximum.reduce(
        [
            B[:, None, 0] - P[None, :, 0],
            np.zeros((len(B), len(P))),
            P[None, :, 0] - B[:, None, 2],
        ]
    )
    dy = np.maximum.reduce(
        [
            B[:, None, 1] - P[None, :, 1],
            np.zeros((len(B), len(P))),
            P[None, :, 1] - B[:, None, 3],
        ]
    )
    return np.hypot(dx, dy)


def _seg_box_hit(P: np.ndarray, Q: np.ndarray, O: np.ndarray) -> np.ndarray:
    if len(O) == 0 or len(P) == 0:
        return np.zeros((len(P), len(O)), dtype=bool)
    d = (Q - P)[:, None, :]
    p0 = P[:, None, :]
    t0 = np.zeros((len(P), len(O)))
    t1 = np.ones((len(P), len(O)))
    ok = np.ones((len(P), len(O)), dtype=bool)
    for axis, (lo_i, hi_i) in enumerate(((0, 2), (1, 3))):
        dd = d[..., axis]
        lo = O[None, :, lo_i] - p0[..., axis]
        hi = O[None, :, hi_i] - p0[..., axis]
        par = np.abs(dd) < 1e-12
        ok &= ~(par & ((lo > 0) | (hi < 0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = np.where(par, -np.inf, lo / dd)
            tb = np.where(par, np.inf, hi / dd)
        t0 = np.maximum(t0, np.minimum(ta, tb))
        t1 = np.minimum(t1, np.maximum(ta, tb))
    return ok & (t0 <= t1)


def _seg_seg_hit(
    P: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray
) -> np.ndarray:
    if len(R) == 0 or len(P) == 0:
        return np.zeros((len(P), len(R)), dtype=bool)

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (
            b[..., 1] - a[..., 1]
        ) * (c[..., 0] - a[..., 0])

    p, q = (P[:, None, :], Q[:, None, :])
    r, s = (R[None, :, :], S[None, :, :])
    o1, o2 = (orient(p, q, r), orient(p, q, s))
    o3, o4 = (orient(r, s, p), orient(r, s, q))
    return (o1 * o2 < 0) & (o3 * o4 < 0)


@dataclass
class _Cands:
    boxes: np.ndarray
    anchors: np.ndarray
    starts: np.ndarray
    r_pt: np.ndarray
    ha: list
    va: list
    base: np.ndarray


@dataclass
class Placed:
    idx: int
    text: str
    box: np.ndarray
    anchor: np.ndarray
    start: np.ndarray
    r_pt: float
    ha: str
    va: str
    leader: bool
    k: int = -1


def _make_candidates(p, w, h, marker_r, px_per_pt, radii_pt):
    boxes, anchors, starts, rs, has, vas = ([], [], [], [], [], [])
    for deg in range(0, 360, 15):
        th = math.radians(deg)
        c, s = (math.cos(th), math.sin(th))
        ha_opts = (
            ["left"]
            if c > 0.4
            else ["right"] if c < -0.4 else ["center", "left", "right"]
        )
        va_opts = ["bottom"] if s > 0.4 else ["top"] if s < -0.4 else ["center"]
        for r in radii_pt:
            rr = r * px_per_pt + marker_r
            ax_, ay_ = (p[0] + c * rr, p[1] + s * rr)
            for ha in ha_opts:
                for va in va_opts:
                    x0 = (
                        ax_
                        if ha == "left"
                        else ax_ - w if ha == "right" else ax_ - w / 2
                    )
                    y0 = (
                        ay_
                        if va == "bottom"
                        else ay_ - h if va == "top" else ay_ - h / 2
                    )
                    boxes.append((x0, y0, x0 + w, y0 + h))
                    anchors.append((ax_, ay_))
                    starts.append((p[0] + c * marker_r, p[1] + s * marker_r))
                    rs.append(r)
                    has.append(ha)
                    vas.append(va)
    return (
        np.asarray(boxes),
        np.asarray(anchors),
        np.asarray(starts),
        np.asarray(rs, dtype=float),
        has,
        vas,
    )


def _stair_height(xq: np.ndarray, front_xy: np.ndarray) -> np.ndarray:
    xs, ys = (front_xy[:, 0], front_xy[:, 1])
    k = np.searchsorted(xs, xq, side="right") - 1
    out = np.full(len(xq), -np.inf)
    ok = k >= 0
    out[ok] = np.maximum.accumulate(ys)[k[ok]]
    return out


def place_labels(
    ax,
    points_disp: np.ndarray,
    point_radii: np.ndarray,
    labels: list[tuple[int, str]],
    sizes: list[tuple[float, float]],
    *,
    front_xy_disp: np.ndarray,
    front_poly_disp: np.ndarray,
    obstacles: list[np.ndarray],
    radii_pt: tuple[float, ...],
    leader_min_pt: float,
    leader_cost: float = 0.0,
    line_cost: float = 3000.0,
    pins: dict | None = None,
    passes: int = 8,
    sweeps: int = 90,
    restarts: int = 3,
    t_start: float = 400.0,
    t_end: float = 2.0,
) -> list[Placed]:
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    axb = ax.get_window_extent()
    inner = np.array([axb.x0 + 4, axb.y0 + 4, axb.x1 - 4, axb.y1 - 4])
    pt_boxes = np.column_stack(
        [points_disp - point_radii[:, None], points_disp + point_radii[:, None]]
    )
    seg_p, seg_q = (front_poly_disp[:-1], front_poly_disp[1:])
    seg_boxes = np.column_stack(
        [np.minimum(seg_p, seg_q) - 1.5, np.maximum(seg_p, seg_q) + 1.5]
    )
    obs = np.asarray(obstacles).reshape(-1, 4) if obstacles else np.zeros((0, 4))
    front_sorted = front_xy_disp[np.argsort(front_xy_disp[:, 0])]
    front_mask = np.zeros(len(points_disp), dtype=bool)
    front_mask[[i for i, _ in labels]] = True
    cands: list[_Cands] = []
    pins = pins or {}
    for j, ((idx, _), (w, h)) in enumerate(zip(labels, sizes)):
        if j in pins:
            dx, dy, ha_, va_ = pins[j][:4]
            with_leader = len(pins[j]) > 4 and pins[j][4] == "leader"
            ay_ = points_disp[idx][1] + dy * px_per_pt
            if dx == "left_edge":
                ha_, ax_ = ("left", inner[0] + 2)
            elif dx == "right_edge":
                ha_, ax_ = ("right", inner[2] - 2)
            else:
                ax_ = points_disp[idx][0] + dx * px_per_pt
            x0 = ax_ if ha_ == "left" else ax_ - w if ha_ == "right" else ax_ - w / 2
            y0 = ay_ if va_ == "bottom" else ay_ - h if va_ == "top" else ay_ - h / 2
            u = np.array([ax_, ay_]) - points_disp[idx]
            u = u / max(np.hypot(*u), 1e-09)
            cands.append(
                _Cands(
                    np.array([[x0, y0, x0 + w, y0 + h]]),
                    np.array([[ax_, ay_]]),
                    (points_disp[idx] + u * point_radii[idx])[None, :],
                    np.array([1000.0 if with_leader else 0.0]),
                    [ha_],
                    [va_],
                    np.zeros(1),
                )
            )
            continue
        B, A, S, R, ha, va = _make_candidates(
            points_disp[idx], w, h, point_radii[idx], px_per_pt, radii_pt
        )
        leader = R >= leader_min_pt
        base = R * 1.0
        outside = (
            (B[:, 0] < inner[0])
            | (B[:, 1] < inner[1])
            | (B[:, 2] > inner[2])
            | (B[:, 3] > inner[3])
        )
        base = base + outside * 1000000.0
        ov = _box_overlap(B, pt_boxes)
        base = base + (ov > 0).sum(1) * 1500 + ov.sum(1) * 10
        base = base + (_box_overlap(B + [-3, -2, 3, 2], obs) > 0).sum(1) * 100000.0
        base = (
            base + (_box_overlap(B + [-4, -3, 4, 3], seg_boxes) > 0).sum(1) * line_cost
        )
        depth = _stair_height(B[:, 2] - 1, front_sorted) - B[:, 1]
        base = base + (depth > 0) * (260 + 1.0 * np.clip(depth, 0, None) / px_per_pt)
        other_pts = np.ones(len(pt_boxes), dtype=bool)
        other_pts[idx] = False
        d_own = _box_point_dist(B, points_disp[idx][None, :])[:, 0]
        d_oth = (
            _box_point_dist(B, points_disp[other_pts]) - point_radii[other_pts][None, :]
        )
        close = d_oth < (d_own - point_radii[idx])[:, None] + 10.0
        w_amb = np.where(front_mask[other_pts], 2500.0, 500.0)[None, :]
        base = base + (close * w_amb).sum(1) * np.where(leader, 0.35, 1.0)
        lp = _seg_box_hit(S, A, pt_boxes[other_pts]).sum(1)
        ls = _seg_box_hit(S, A, seg_boxes).sum(1)
        base = base + leader * (lp * 900 + ls * 450 + R * 0.2 + leader_cost)
        cands.append(_Cands(B, A, S, R, ha, va, base))

    def total_cost(ci: int, others: list[Placed]) -> np.ndarray:
        c = cands[ci]
        cost = c.base.copy()
        if not others:
            return cost
        ob = np.array([o.box for o in others])
        ov = _box_overlap(c.boxes + [-4, -2, 4, 2], ob)
        cost += (ov > 0).sum(1) * 10000.0 + ov.sum(1) * 50
        leader = c.r_pt >= leader_min_pt
        cost += leader * _seg_box_hit(c.starts, c.anchors, ob).sum(1) * 6000
        lead_o = [o for o in others if o.leader]
        if lead_o:
            R = np.array([o.start for o in lead_o])
            S = np.array([o.anchor for o in lead_o])
            cost += leader * _seg_seg_hit(c.starts, c.anchors, R, S).sum(1) * 4000
            cost += _seg_box_hit(R, S, c.boxes).T.sum(1) * 6000
        return cost

    for c in cands:
        keep = np.argsort(c.base, kind="stable")[
            : max(60, min(320, int(np.sum(c.base < 100000.0))))
        ]
        c.boxes, c.anchors, c.starts, c.r_pt, c.base = (
            c.boxes[keep],
            c.anchors[keep],
            c.starts[keep],
            c.r_pt[keep],
            c.base[keep],
        )
        c.ha, c.va = ([c.ha[i] for i in keep], [c.va[i] for i in keep])

    def placed_at(ci: int, k: int) -> Placed:
        c = cands[ci]
        idx, text = labels[ci]
        return Placed(
            idx,
            text,
            c.boxes[k],
            c.anchors[k],
            c.starts[k],
            float(c.r_pt[k]),
            c.ha[k],
            c.va[k],
            bool(c.r_pt[k] >= leader_min_pt),
            k,
        )

    n = len(labels)

    def greedy_polish(state: dict[int, Placed], order) -> None:
        for _ in range(passes):
            changed = False
            for j in order:
                k = int(np.argmin(total_cost(j, [state[o] for o in state if o != j])))
                if k != state[j].k:
                    changed = True
                state[j] = placed_at(j, k)
            if not changed:
                break

    def energy(state: dict[int, Placed]) -> float:
        return float(
            sum(
                (
                    total_cost(j, [state[o] for o in state if o != j])[state[j].k]
                    for j in state
                )
            )
        )

    dens = np.array(
        [
            np.sum(
                np.hypot(
                    *(points_disp[[i for i, _ in labels]] - points_disp[labels[j][0]]).T
                )
                < 160 * px_per_pt
            )
            for j in range(n)
        ]
    )
    order = list(np.argsort(-dens, kind="stable"))
    state: dict[int, Placed] = {}
    for j in order:
        state[j] = placed_at(j, int(np.argmin(total_cost(j, list(state.values())))))
    greedy_polish(state, order)
    best_state, best_e = (dict(state), energy(state))
    rng = np.random.default_rng(0)
    steps = sweeps * n
    for _ in range(restarts):
        state = dict(best_state)
        for it in range(steps):
            temp = t_start * (t_end / t_start) ** (it / max(1, steps - 1))
            j = int(rng.integers(n))
            cost = total_cost(j, [state[o] for o in state if o != j])
            w = np.exp(-(cost - cost.min()) / temp)
            state[j] = placed_at(j, int(rng.choice(len(cost), p=w / w.sum())))
        greedy_polish(state, order)
        e = energy(state)
        if e < best_e:
            best_state, best_e = (dict(state), e)
    return [best_state[j] for j in range(n)]


def annotate_placed(
    ax,
    items: list[Placed],
    data_xy: np.ndarray,
    *,
    colors: list,
    fontsize: float,
    badge: bool,
    marker_radii_px: np.ndarray,
) -> None:
    px_per_pt = ax.figure.dpi / 72.0
    for it in items:
        p_disp = ax.transData.transform(data_xy[it.idx])
        off = (it.anchor - p_disp) / px_per_pt
        kw = dict(
            xy=tuple(data_xy[it.idx]),
            xytext=tuple(off),
            textcoords="offset points",
            ha=it.ha,
            va=it.va,
            fontsize=fontsize,
            fontweight="bold",
            color=colors[it.idx],
            annotation_clip=False,
            zorder=9,
        )
        if badge:
            kw["bbox"] = dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="black",
                linewidth=1.1,
            )
        else:
            kw["path_effects"] = [pe.withStroke(linewidth=4.5, foreground="white")]
        if it.leader:
            relpos = (
                {"left": 0.0, "center": 0.5, "right": 1.0}[it.ha],
                {"bottom": 0.0, "center": 0.5, "top": 1.0}[it.va],
            )
            kw["arrowprops"] = dict(
                arrowstyle="-",
                color="0.3",
                lw=1.1,
                shrinkA=0,
                shrinkB=marker_radii_px[it.idx] / px_per_pt + 0.5,
                relpos=relpos,
            )
        ax.annotate(it.text, **kw)


def measure_text(ax, text: str, fontsize: float, *, badge: bool) -> tuple[float, float]:
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    t = ax.text(0, 0, text, fontsize=fontsize, fontweight="bold", transform=None)
    bb = t.get_window_extent(renderer)
    t.remove()
    w, h = (bb.width, bb.height)
    if badge:
        pad = 2 * 0.22 * fontsize * fig.dpi / 72.0 + 2.0
        return (w + pad, h + pad)
    return (w + 5.0, h + 2.0)


def variant_handles(style: Style) -> list[Line2D]:
    ms = 12 * (style.variant_legend or style.legend) / style.legend
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.25",
            linestyle="None",
            markersize=ms,
            label="Default (D)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="0.25",
            linestyle="None",
            markersize=ms,
            label="Tuned (T)",
        ),
        Line2D(
            [0],
            [0],
            color=style.front_color,
            linestyle=style.front_dash,
            linewidth=style.front_lw,
            label="Pareto front",
        ),
    ]


def model_handles(cols: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            linestyle="None",
            label=m,
            markerfacecolor=c,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=13,
        )
        for m, c in cols.items()
    ]


def front_key(
    ax,
    front: pd.DataFrame,
    cols: dict[str, str],
    style: Style,
    *,
    obstacles,
    points_disp,
    point_radii,
    front_poly_disp,
    default_only,
):
    """In-panel key (marker shape = the variant on the front), placed where it hides no point.

    Tries the full key font in one column first, then two columns, then slightly
    smaller fonts, and keeps the first layout that covers no marker, no front line
    and no arrow.
    """
    fig = ax.figure
    axb = ax.get_window_extent()
    pt_boxes = np.column_stack(
        [points_disp - point_radii[:, None], points_disp + point_radii[:, None]]
    )
    seg_p, seg_q = (front_poly_disp[:-1], front_poly_disp[1:])
    seg_boxes = np.column_stack(
        [np.minimum(seg_p, seg_q) - 1.5, np.maximum(seg_p, seg_q) + 1.5]
    )
    obs = np.asarray(obstacles).reshape(-1, 4) if obstacles else np.zeros((0, 4))
    rows = front.reset_index(drop=True)

    def build(fs: float, ncol: int):
        handles = [
            Line2D(
                [0],
                [0],
                marker=MARKERS.get(str(r["variant"]), "o"),
                color="none",
                linestyle="None",
                markersize=fs * 0.7,
                markerfacecolor=cols[str(r["model"])],
                markeredgecolor="black",
                markeredgewidth=0.8,
                label=f"{i + 1}  {front_label(r['model'], r['variant'], default_only)}",
            )
            for i, r in rows.iterrows()
        ]
        leg = ax.legend(
            handles=handles,
            title="Pareto front",
            loc="lower left",
            bbox_to_anchor=(0, 0),
            ncol=ncol,
            fontsize=fs,
            title_fontsize=fs + 1.0,
            framealpha=0.96,
            borderpad=0.45,
            labelspacing=0.24,
            handletextpad=0.4,
            handlelength=1.1,
            columnspacing=1.0,
        )
        leg.set_zorder(10)
        fig.canvas.draw()
        lb = leg.get_window_extent()
        w, h = (lb.width, lb.height)
        m = 6.0
        xs = np.linspace(axb.x0 + m, axb.x1 - m - w, 70)
        ys = np.linspace(axb.y0 + m, axb.y1 - m - h, 60)
        gx, gy = np.meshgrid(xs, ys)
        B = np.column_stack([gx.ravel(), gy.ravel(), gx.ravel() + w, gy.ravel() + h])
        cost = (_box_overlap(B + [-4, -4, 4, 4], pt_boxes) > 0).sum(1) * 10000.0
        cost = cost + (_box_overlap(B, seg_boxes) > 0).sum(1) * 2000
        cost = cost + (_box_overlap(B, obs) > 0).sum(1) * 1000000.0
        cost = cost + 0.2 * (axb.x1 - B[:, 2] + (B[:, 1] - axb.y0)) / (fig.dpi / 72.0)
        k = int(np.argmin(cost))
        leg.remove()
        return (B[k], float(cost[k]))

    best = None
    done = False
    for fs in np.arange(style.key, style.key - 3.01, -0.5):
        for ncol in (1, 2) if len(rows) > 6 else (1,):
            box, c = build(float(fs), ncol)
            if best is None or c < best[0]:
                best = (c, float(fs), ncol, box)
            if c < 2000:
                done = True
                break
        if done:
            break
    _, fs, ncol, box = best
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS.get(str(r["variant"]), "o"),
            color="none",
            linestyle="None",
            markersize=fs * 0.7,
            markerfacecolor=cols[str(r["model"])],
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=f"{i + 1}  {front_label(r['model'], r['variant'], default_only)}",
        )
        for i, r in rows.iterrows()
    ]
    leg = ax.legend(
        handles=handles,
        title="Pareto front",
        loc="lower left",
        bbox_to_anchor=((box[0] - axb.x0) / axb.width, (box[1] - axb.y0) / axb.height),
        ncol=ncol,
        fontsize=fs,
        title_fontsize=fs + 1.0,
        framealpha=0.96,
        borderpad=0.45,
        labelspacing=0.24,
        handletextpad=0.4,
        handlelength=1.1,
        columnspacing=1.0,
    )
    leg.set_zorder(10)
    fig.canvas.draw()
    print(f"[key] font {fs:g}, {ncol} col, placement cost {best[0]:.0f}")
    return leg


def _legend(
    fig,
    handles,
    *,
    title,
    ncol,
    fs,
    style: Style,
    spacing: float,
    anchor,
    loc="upper left",
    borderpad=0.4,
):
    leg = fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        fontsize=fs,
        title=title,
        title_fontsize=style.legend_title,
        frameon=style.legend_frame,
        framealpha=1.0,
        borderpad=borderpad,
        labelspacing=0.38,
        handletextpad=0.3,
        columnspacing=spacing,
        handlelength=1.0,
        borderaxespad=0.0,
    )
    leg._legend_box.align = "center"
    return leg


def _centred_last_entry(fig, lg, model: str, color: str, fs: float):
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea

    centres = [
        ((t.get_window_extent().y0 + t.get_window_extent().y1) / 2, t)
        for t in lg.get_texts()
    ]
    rows = sorted({round(c, 1) for c, _ in centres}, reverse=True)
    pitch_pt = (rows[0] - rows[1]) * 72.0 / fig.dpi
    last = min(centres, key=lambda ct: ct[0])[1]
    da = DrawingArea(1.0 * fs, fs, 0, 0)
    da.add_artist(
        Line2D(
            [0.5 * fs],
            [0.5 * fs],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=13,
        )
    )
    box = HPacker(
        children=[da, TextArea(model, textprops=dict(fontsize=fs))],
        sep=0.3 * fs,
        pad=0,
        align="center",
    )
    ab = AnnotationBbox(
        box,
        (0.5, 0.5),
        xycoords=(lg, last),
        xybox=(0, -pitch_pt),
        boxcoords="offset points",
        box_alignment=(0.5, 0.5),
        frameon=False,
        pad=0,
        annotation_clip=False,
    )
    ab.set_zorder(lg.get_zorder() + 1)
    fig.add_artist(ab)
    return ab


def place_top_legends(
    fig,
    cols: dict[str, str],
    style: Style,
    *,
    y_top: float,
    groups: list[tuple[str, list[str]]] | None = None,
) -> float:
    fw = fig.get_figwidth() * fig.dpi
    fh = fig.get_figheight() * fig.dpi
    leg_v = fig.legend(
        handles=variant_handles(style),
        loc="upper center",
        bbox_to_anchor=(0.5, y_top),
        ncol=3,
        fontsize=style.variant_legend or style.legend,
        frameon=style.legend_frame,
        framealpha=1.0,
        borderpad=0.3,
        handletextpad=0.5,
        handlelength=2.4,
        columnspacing=2.4,
        borderaxespad=0.0,
    )
    fig.canvas.draw()
    if style.variant_below:
        y_model = y_top
    else:
        y_model = y_top - leg_v.get_window_extent().height / fh - style.variant_gap
    blocks = (
        [(t, [m for m in ms if m in cols]) for t, ms in groups]
        if groups
        else [("Model", list(cols))]
    )
    gap_px = 0.45 * fig.dpi
    spacings = (
        (style.legend_colspacing,)
        if style.legend_colspacing is not None
        else (1.5, 1.2, 0.9)
    )
    if style.legend_fit == "axes":
        attempts = [(3 if groups is None else 4, spacings[0], style.legend)]
    else:
        attempts = [
            (rows, spacing, fs)
            for rows in ((3,) if groups is None else (3, 4))
            for fs in np.arange(style.legend, style.legend - 2.51, -0.5)
            for spacing in spacings
        ]
    for rows, spacing, fs in attempts:
        legs, extras = ([], [])
        for title, members in blocks:
            ncol = math.ceil(len(members) / rows)
            lone = None
            if style.legend_center_last and ncol == 2 and (len(members) % 2 == 1):
                k = math.ceil(len(members) / 2) - 1
                lone, members = (members[k], members[:k] + members[k + 1 :])
            handles = model_handles({m: cols[m] for m in members})
            moved = [m for m in members if m in style.legend_right_col]
            if moved and ncol == 2:
                rest = [m for m in members if m not in moved]
                left, right = (
                    rest[: math.ceil(len(rest) / 2)],
                    rest[math.ceil(len(rest) / 2) :] + moved,
                )
                n = max(len(left), len(right))
                blank = Line2D([], [], linestyle="None", marker="None", label=" ")
                handles = (
                    model_handles({m: cols[m] for m in left})
                    + [blank] * (n - len(left))
                    + model_handles({m: cols[m] for m in right})
                    + [blank] * (n - len(right))
                )
            legs.append(
                _legend(
                    fig,
                    handles,
                    title=title,
                    ncol=ncol,
                    fs=float(fs),
                    style=style,
                    spacing=spacing,
                    anchor=(0, y_model),
                )
            )
            extras.append(lone)
        fig.canvas.draw()
        widths = [lg.get_window_extent().width for lg in legs]
        total = sum(widths) + gap_px * (len(legs) - 1)
        if (
            style.legend_fit == "axes"
            or total / fw <= 0.985
            or (rows, spacing, fs) == attempts[-1]
        ):
            break
        for lg in legs:
            lg.remove()
    if style.legend_fit == "axes":
        m = style.margins
        span = m["right"] - m["left"]
        if total > span * fw:
            fig.set_figwidth(total / span / fig.dpi * 1.004)
            fw = fig.get_figwidth() * fig.dpi
            fig.canvas.draw()
            widths = [lg.get_window_extent().width for lg in legs]
            total = sum(widths) + gap_px * (len(legs) - 1)
        x = m["left"] * fw + (span * fw - total) / 2.0
        x_centre = m["left"] + span / 2
        leg_v.set_bbox_to_anchor((x_centre, y_top), transform=fig.transFigure)
    else:
        x = (fw - total) / 2.0
        x_centre = 0.5
    for lg, _, w in zip(legs, extras, widths):
        lg.set_bbox_to_anchor((x / fw, y_model), transform=fig.transFigure)
        if groups is not None:
            lg.get_title().set_fontweight("semibold")
        x += w + gap_px
    fig.canvas.draw()
    boxes = [
        _centred_last_entry(fig, lg, lone, cols[lone], float(fs))
        for lg, lone in zip(legs, extras)
        if lone
    ]
    if boxes:
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = (
        min(
            [lg.get_window_extent().y0 for lg in legs]
            + [b.get_window_extent(renderer).y0 for b in boxes]
        )
        / fh
    )
    if style.variant_below:
        leg_v.set_bbox_to_anchor(
            (x_centre, bottom - style.variant_gap), transform=fig.transFigure
        )
        fig.canvas.draw()
        bottom = leg_v.get_window_extent().y0 / fh
    return bottom


@dataclass
class Panel:
    summary: pd.DataFrame
    x_col: str
    x_label: str


def plot_pareto_pair(
    panels: list[Panel],
    out_png: Path,
    *,
    y_col: str,
    y_label: str,
    metric: str,
    mode: str,
    style: Style | None = None,
    arrow: bool = True,
    colors: dict[str, str] | None = None,
    legend_groups: list[tuple[str, list[str]]] | None = None,
) -> list[pd.DataFrame]:
    style = style or Style()
    if mode not in ("key", "labels"):
        raise ValueError(mode)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
        }
    )
    if style.font_serif:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": list(style.font_serif),
                "mathtext.fontset": "stix",
                "text.usetex": False,
            }
        )
    default_only = default_only_models(panels)
    models = sorted(set().union(*[set(p.summary["model"].astype(str)) for p in panels]))
    cols = dict(colors) if colors is not None else model_colors(models)
    if legend_groups is None:
        cols = {m: cols[m] for m in sorted(cols) if m in models}
    else:
        cols = {m: cols[m] for _, ms in legend_groups for m in ms if m in models}
    text_cols = {
        m: readable_text_color(c, style.label_min_contrast) for m, c in cols.items()
    }
    text_cols.update(
        {m: mcolors.to_rgb(c) for m, c in style.label_colors.items() if m in text_cols}
    )
    all_y = np.concatenate([p.summary[y_col].to_numpy(dtype=float) for p in panels])
    if mode == "labels":
        headroom = 0.12 if metric == "norm" else 0.24
    else:
        headroom = 0.0 if metric == "norm" else 0.1
    ylim, yticks = shared_y_config(metric, all_y, headroom=headroom)
    fig, axes = plt.subplots(1, len(panels), figsize=style.figsize, sharey=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(**style.margins)
    legend_bottom = place_top_legends(
        fig, cols, style, y_top=0.992, groups=legend_groups
    )
    fig.subplots_adjust(**{**style.margins, "top": legend_bottom - style.legend_gap})
    fronts, polys = ([], [])
    for k, (ax, p) in enumerate(zip(axes, panels)):
        d = p.summary
        if metric == "norm":
            apply_compressed_normalized_y(ax, ylim)
        else:
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(FixedLocator(yticks))
        pad_l = style.pad_left[k] if style.pad_left else 0.45
        xlim = log_x_limits(
            d[p.x_col].to_numpy(dtype=float), pad_left=pad_l, pad_right=0.22
        )
        ax.set_xlim(*xlim)
        scatter_points(ax, d, cols, p.x_col, y_col, style)
        front = d[d["is_pareto"]].sort_values(p.x_col).reset_index(drop=True)
        fronts.append(front)
        polys.append(
            draw_staircase_front(
                ax, front, p.x_col, y_col, xlim=xlim, ylim=ylim, style=style
            )
        )
        style_axes(
            ax, x_label=p.x_label, y_label=y_label if k == 0 else None, style=style
        )
        if k > 0:
            ax.tick_params(axis="y", labelleft=False)
    fig.canvas.draw()
    x0 = axes[0].yaxis.label.get_window_extent().x0
    if x0 < 0:
        left = style.margins["left"] + (4 - x0) / (fig.get_figwidth() * fig.dpi)
        fig.subplots_adjust(
            **{**style.margins, "left": left, "top": legend_bottom - style.legend_gap}
        )
        fig.canvas.draw()
    placements = []
    for k, (ax, p, front, poly) in enumerate(zip(axes, panels, fronts, polys)):
        d = p.summary.reset_index(drop=True)
        xy = d[[p.x_col, y_col]].to_numpy(dtype=float)
        disp = ax.transData.transform(xy)
        radii = np.where(
            d["is_pareto"].to_numpy(),
            math.sqrt(style.s_front) / 2.0 + front_pad_pt(style),
            math.sqrt(style.s_other) / 2.0,
        )
        radii = radii * fig.dpi / 72.0 + 1.0
        poly_disp = ax.transData.transform(poly)
        front_idx = [
            int(np.flatnonzero((d["competitor"] == c).to_numpy())[0])
            for c in front["competitor"]
        ]
        obstacles = []
        if arrow:
            obstacles.append(draw_optimal_cue(ax, style=style))
        if mode == "key":
            leg = front_key(
                ax,
                front,
                cols,
                style,
                obstacles=obstacles,
                points_disp=disp,
                point_radii=radii,
                front_poly_disp=poly_disp,
                default_only=default_only,
            )
            lb = leg.get_window_extent()
            obstacles.append(np.array([lb.x0, lb.y0, lb.x1, lb.y1]))
            texts = [str(n + 1) for n in range(len(front_idx))]
            fs, badge, radii_pt, leader_pt = (
                style.badge,
                True,
                (3, 7, 12, 18, 26, 36, 48, 62, 80),
                15,
            )
            colors = ["black"] * len(d)
        else:
            texts = [
                front_label(d.at[i, "model"], d.at[i, "variant"], default_only)
                for i in front_idx
            ]
            fs, badge, leader_pt = (style.label, False, 13)
            radii_pt = (2, 6, 11, 17, 24, 32, 42, 54, 68, 84, 102, 122, 145, 170, 200)
            colors = [text_cols[str(m)] for m in d["model"]]
        sizes = [measure_text(ax, t, fs, badge=badge) for t in texts]
        items = place_labels(
            ax,
            disp,
            radii,
            list(zip(front_idx, texts)),
            sizes,
            front_xy_disp=disp[front_idx],
            front_poly_disp=poly_disp,
            obstacles=obstacles,
            radii_pt=radii_pt,
            leader_min_pt=leader_pt,
            leader_cost=style.leader_cost,
            line_cost=style.line_cost,
            pins={
                j: tuple(style.label_pins[k, t])
                for j, t in enumerate(texts)
                if (k, t) in style.label_pins
            },
        )
        annotate_placed(
            ax,
            items,
            xy,
            colors=colors,
            fontsize=fs,
            badge=badge,
            marker_radii_px=radii,
        )
        ppt = fig.dpi / 72.0
        placements.append(
            {
                it.text: [
                    round(float((it.anchor - disp[it.idx])[0] / ppt), 1),
                    round(float((it.anchor - disp[it.idx])[1] / ppt), 1),
                    it.ha,
                    it.va,
                ]
                + (["leader"] if it.leader else [])
                for it in items
            }
        )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {}
    if style.center_frames:
        fig.canvas.draw()
        x0 = axes[0].get_window_extent().x0 / fig.dpi
        x1 = axes[-1].get_window_extent().x1 / fig.dpi
        width = max(fig.get_figwidth(), x1 + x0)
        save_kw["bbox_inches"] = Bbox.from_extents(0, 0, width, fig.get_figheight())
    fig.savefig(out_png, dpi=300, **save_kw)
    fig.savefig(out_png.with_suffix(".pdf"), **save_kw)
    import json

    out_png.with_suffix(".labels.json").write_text(
        json.dumps(dict(zip(("train", "inference"), placements)), indent=1) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return fronts
