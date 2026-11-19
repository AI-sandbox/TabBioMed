"""Binary metrics analysis."""

from tabbiomed.analysis.paths import PLAN, PROCESSED, RESULTS

from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, ROUND_HALF_UP
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-binary-metrics-mpl")
import pandas as pd
from tabbiomed.analysis.common import load_benchmark_models_config
from tabbiomed.analysis.binary_metrics import ALT_METRICS, METRICS, score_pair
from tabbiomed.analysis.binary_ranks import agreement_tables, mean_rank_table


def plot_imbalance(detail, path, threshold=2.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0), sharey=True)
        for ax, spec in zip(axes.ravel(), ALT_METRICS):
            part = detail[detail.metric.eq(spec["key"])].dropna(
                subset=["imbalance_ratio", "spearman_rho"]
            )
            part = part[part.imbalance_ratio.gt(0)]
            ax.scatter(
                part.imbalance_ratio,
                part.spearman_rho,
                s=46,
                color="#0072B2",
                alpha=0.85,
                linewidths=0,
            )
            ax.axvline(threshold, color="#666666", linestyle="--", linewidth=1.2)
            ax.set_xscale("log")
            ax.set_title(f"ROC-AUC vs {spec['label']}", fontweight="bold", pad=10)
            ax.set_xlabel("Imbalance ratio")
            ax.set_ylim(-0.5, 1.05)
            ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
            ax.grid(False, axis="x")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color(".15")
            ax.tick_params(axis="both", width=1.1, length=4)
        for ax in axes[:, 0]:
            ax.set_ylabel("Spearman correlation of ranks")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def median_text(value):
    text = f"{value:.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def ranking_rows(ranks):
    best = {s["key"]: ranks[f"mean_rank_{s['key']}"].min() for s in METRICS}
    rows = []
    for row in ranks.to_dict("records"):
        cells = [row["model"], str(int(row["n_roc_auc"]))]
        for spec in METRICS:
            key = spec["key"]
            value = row[f"mean_rank_{key}"]
            mean = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            cell = f"{mean} / {median_text(row[f'median_rank_{key}'])}"
            cells.append("\\textbf{" + cell + "}" if value == best[key] else cell)
        rows.append(" & ".join(cells) + " \\\\")
    return rows


def agreement_rows(summary):
    summary = summary.set_index("metric").loc[[s["key"] for s in ALT_METRICS]]
    rows = []

    def add(label, value):
        rows.append(
            " & ".join([label] + [value(r) for r in summary.to_dict("records")])
            + " \\\\"
        )

    add("Median Spearman $\\rho$", lambda r: f"{r['median_spearman']:.3f}")
    add("Median Kendall $\\tau$", lambda r: f"{r['median_kendall']:.3f}")
    add("Top-1 changes", lambda r: f"{r['top1_flips']}/{r['n_datasets']}")
    add("Top-1 change rate (\\%)", lambda r: f"{100 * r['top1_flip_rate']:.1f}")
    add("Median top-3 Jaccard similarity", lambda r: f"{r['median_top_k_jaccard']:.3f}")
    for subset in ("balanced", "imbalanced"):
        condition = "\\leq" if subset == "balanced" else ">"
        title = "Relatively balanced" if subset == "balanced" else "Imbalanced"
        count = int(summary[f"n_{subset}"].iloc[0])
        rows.append("\\midrule")
        rows.append(
            "\\multicolumn{5}{l}{\\textit{"
            + title
            + f" datasets ($\\mathrm{{IR}}{condition} 2$, $n={count}$)"
            + "}} \\\\"
        )
        add(
            "Median Spearman $\\rho$", lambda r: f"{r[f'{subset}_median_spearman']:.3f}"
        )
        add(
            "Top-1 changes", lambda r: f"{r[f'{subset}_top1_flips']}/{r[f'n_{subset}']}"
        )
        add(
            "Top-1 change rate (\\%)",
            lambda r: f"{100 * r[f'{subset}_top1_flip_rate']:.1f}",
        )
    return rows


def write_tables(ranks, summary, output):
    tables = {
        "binary_metric_ranks": ("lcccccc", ranking_rows(ranks)),
        "binary_metric_agreement": ("lcccc", agreement_rows(summary)),
    }
    headers = {
        "binary_metric_ranks": "Model & $N$ & ROC-AUC & PR-AUC & Log loss & Brier score & ECE \\\\",
        "binary_metric_agreement": "Statistic & PR-AUC & Log loss & Brier score & ECE \\\\",
    }
    for name, (columns, rows) in tables.items():
        text = [
            f"\\begin{{tabular}}{{{columns}}}",
            "\\toprule",
            headers[name],
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
        (output / f"{name}.tex").write_text("\n".join(text) + "\n")
    return tables


def verify_paper(paper, tables, figure):
    text = paper.read_text()

    def normalize(value):
        return re.sub("\\s+", "", value)

    result = {}
    for name, (_, rows) in tables.items():
        label = f"\\label{{tab:{name}}}"
        position = text.index(label)
        start = text.rfind("\\begin{table", 0, position)
        end = text.index("\\end{table", position)
        block = normalize(text[start:end])
        mismatches = [
            r
            for r in rows
            if "&" in r
            and (not r.startswith("\\multicolumn"))
            and (normalize(r) not in block)
        ]
        result[name] = {"matches": not mismatches, "mismatched_rows": mismatches}
    target = paper.parent / "vis/rank_agreement_vs_imbalance.pdf"
    if not target.is_file():
        raise FileNotFoundError(target)
    if not shutil.which("pdftoppm"):
        raise RuntimeError("Paper figure verification requires pdftoppm")
    from PIL import Image

    def pixels(path):
        data = subprocess.check_output(
            ["pdftoppm", "-singlefile", "-scale-to", "1400", "-png", str(path)]
        )
        with Image.open(BytesIO(data)) as im:
            return (im.size, im.convert("RGB").tobytes())

    result["rank_agreement_vs_imbalance"] = {
        "pixels_identical": pixels(target) == pixels(figure)
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-paper", type=Path)
    args = parser.parse_args()
    ensure_results(args.results_dir)
    if args.workers < 1:
        parser.error("--workers must be positive")
    config = load_benchmark_models_config()
    models = sorted(config.expected_models_binary_classification)
    plan = load_benchmark(args.plan, args.processed_path)
    datasets = sorted(
        (d for d, info in plan["datasets"].items() if info["task"] == "binary")
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pairs = [(m, d) for m in models for d in datasets]

    def load(pair):
        m, d = pair
        return score_pair(
            m, d, args.results_dir, args.processed_path, info=plan["datasets"][d]
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scored = list(pool.map(load, pairs))
    scores = pd.DataFrame([row for row, _ in scored])
    scores.to_csv(args.output / "scores_long.csv", index=False)
    pd.DataFrame([note for _, notes in scored for note in notes]).to_csv(
        args.output / "score_notes.csv", index=False
    )
    scores = pd.read_csv(args.output / "scores_long.csv")
    for spec in METRICS:
        if not scores[f"{spec['key']}_source"].eq(scores.selected_variant).all():
            raise ValueError(
                "All metrics must use the same complete evaluation variant"
            )
    ranks = mean_rank_table(scores)
    summary, detail = agreement_tables(scores)
    for name, frame in (
        ("mean_ranks", ranks),
        ("agreement_summary", summary),
        ("per_dataset_rank_corr", detail),
    ):
        frame.to_csv(args.output / f"{name}.csv", index=False)
    tables = write_tables(ranks, summary, args.output)
    figure = args.output / "rank_agreement_vs_imbalance.pdf"
    plot_imbalance(detail, figure)
    manifest = dict(
        results_root=str(args.results_dir.resolve()),
        n_models=len(models),
        n_datasets=len(datasets),
        n_pairs=len(pairs),
        policy="Main results only, complete Tuned then complete Default, one variant for all five metrics, equal outer-fold means, no imputation",
        selected_variants=scores.selected_variant.value_counts().to_dict(),
        ece_bins=15,
        imbalance_threshold=2,
        top_k=3,
    )
    if args.verify_paper:
        manifest["paper_comparison"] = verify_paper(args.verify_paper, tables, figure)
    (args.output / "analysis.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if args.verify_paper and any(
        (
            not check.get("matches", check.get("pixels_identical", False))
            for check in manifest["paper_comparison"].values()
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
