"""Pca analysis."""

from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.paths import RESULTS, TEMPLATES
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tabbiomed.analysis.common import PRIMARY_METRIC
from tabbiomed.analysis.pca import dataset_inventory, eligible_datasets
from tabbiomed.results.selection import test_available

PCA_KEYS = {
    "apply_pca",
    "pca_max_features",
    "pca_fit",
    "pca_impute_missing",
    "pca_imputation_strategy",
}
TASK_ABBREVIATIONS = {
    "binary": "Bin",
    "classification": "MultiC",
    "regression": "Reg",
    "multiregression": "MultiReg",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_score(report, variant="best"):
    folds = report.get("cv_fold_metrics")
    if folds and (len(folds) != 5 or {u["fold_id"] for u in folds} != set(range(5))):
        raise ValueError("PCA comparisons require all five stored outer folds")
    metric = report["primary_metric"]
    blocks = [unit.get(variant) for unit in folds or [report]]
    if not all(test_available(block, metric) for block in blocks):
        return None
    return sum(block["test_metrics"][metric] for block in blocks) / len(blocks)


def selected_score(report):
    for variant, label in [("best", "tuned"), ("default", "default")]:
        value = test_score(report, variant)
        if value is not None:
            return (value, label)
    return (None, None)


def oriented_difference(pca, baseline, metric):
    if metric not in ("roc_auc", "log_loss", "mse"):
        raise ValueError("Unexpected primary metric")
    return (pca - baseline) * (1 if metric == "roc_auc" else -1)


def validate_pair(main, pca, model, dataset, task, component_cap=500, components=None):
    for field, expected in [
        ("model", model),
        ("dataset", dataset),
        ("primary_metric", PRIMARY_METRIC[task]),
    ]:
        if main.get(field) != expected or pca.get(field) != expected:
            raise ValueError(f"Mismatched PCA comparison: {model}/{dataset}/{field}")
    if main.get("seed") != pca.get("seed"):
        raise ValueError("PCA and Main seeds differ")
    if bool(main.get("cv_fold_metrics")) != bool(pca.get("cv_fold_metrics")):
        raise ValueError("PCA and Main outer evaluation schemes differ")
    recorded = pca["search_space"].get("pca_max_features", {}).get("choices")
    if recorded != [component_cap]:
        if (
            not components
            or recorded != [components.get("recorded_fit_component_cap")]
            or components.get("comparison_component_cap") != component_cap
            or (components.get("component_caps_equivalent") is not True)
        ):
            raise ValueError("Recorded PCA cap lacks verified component equivalence")
        for key in ("inner_fit_components", "final_fit_components"):
            width = components.get(key, {}).get("max")
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or (not 0 < width <= component_cap)
            ):
                raise ValueError("Equivalent PCA fits exceed the comparison cap")
    for key, value in [("apply_pca", True), ("pca_fit", "lowrank")]:
        if pca["search_space"].get(key, {}).get("choices") != [value]:
            raise ValueError("Unexpected fixed PCA search settings")
    spaces = [
        {
            k: v
            for k, v in report["search_space"].items()
            if k not in PCA_KEYS | {"custom_model_dir"}
        }
        for report in (main, pca)
    ]
    if spaces[0] != spaces[1]:
        raise ValueError("PCA and Main search spaces differ beyond fixed PCA settings")


def collect(root, inventory, models, component_cap=500):
    rows, sources = ([], {})

    def read(relative):
        path = root / relative
        raw = path.read_bytes()
        sources[str(relative)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for dataset in sorted(
        eligible_datasets(inventory), key=lambda d: (inventory[d]["num_features"], d)
    ):
        info = inventory[dataset]
        for model in models:
            parent = Path(model) / dataset
            directory = parent / f"pca_{component_cap}"
            main, pca = (read(parent / "report.json"), read(directory / "report.json"))
            if component_cap == 500 and (root / directory / "pca_components.json").is_file():
                components = read(directory / "pca_components.json")
            elif component_cap == 500:
                eff = min(
                    component_cap,
                    int(info["num_samples"]),
                    int(info["num_features"]),
                )
                components = dict(
                    recorded_fit_component_cap=component_cap,
                    comparison_component_cap=component_cap,
                    component_caps_equivalent=True,
                    inner_fit_components=dict(min=1, max=eff),
                    final_fit_components=dict(min=1, max=eff),
                )
            else:
                components = None
            validate_pair(
                main, pca, model, dataset, info["task"], component_cap, components
            )
            baseline, main_variant = selected_score(main)
            reduced, pca_variant = selected_score(pca)
            row = dict(
                dataset=dataset,
                model=model,
                task=info["task"],
                num_samples=info["num_samples"],
                num_features=info["num_features"],
                metric=main["primary_metric"],
                variant="tuned_with_default_fallback",
                main_variant=main_variant,
                pca_variant=pca_variant,
                outer_folds=5 if main.get("cv_fold_metrics") else 1,
                main_score=baseline,
                pca_score=reduced,
                delta=None,
                reference=None,
                reference_score=None,
                reference_delta=None,
            )
            if baseline is not None and reduced is not None:
                row["delta"] = oriented_difference(reduced, baseline, row["metric"])
                row["outcome"] = (
                    "better"
                    if row["delta"] > 0
                    else "worse"
                    if row["delta"] < 0
                    else "tied"
                )
            elif reduced is not None:
                row["outcome"] = "pca_only"
                rf = read(Path("RandomForest") / dataset / "report.json")
                if rf["primary_metric"] != row["metric"]:
                    raise ValueError("RandomForest reference uses a different metric")
                score = test_score(rf, "default")
                if score is not None:
                    row.update(
                        reference="RandomForest Default",
                        reference_score=score,
                        reference_delta=oriented_difference(
                            reduced, score, row["metric"]
                        ),
                    )
            else:
                row["outcome"] = "main_only" if baseline is not None else "neither"
            rows.append(row)
    return (rows, sources)


def summarize(rows, models):
    output = []
    for model in models:
        part = [r for r in rows if r["model"] == model]
        count = lambda outcome: sum((r["outcome"] == outcome for r in part))
        output.append(
            dict(
                model=model,
                eligible=len(part),
                pca_available=sum((r["pca_score"] is not None for r in part)),
                main_available=sum((r["main_score"] is not None for r in part)),
                paired=sum((r["delta"] is not None for r in part)),
                better=count("better"),
                worse=count("worse"),
                tied=count("tied"),
                pca_only=count("pca_only"),
                main_only=count("main_only"),
                neither=count("neither"),
            )
        )
    return output


def pfn_direction_agreement(rows):
    models = ("TabPFNv2.5", "RealTabPFNv2.5")
    if not set(models) <= {row["model"] for row in rows}:
        return None
    paired = {
        m: {
            r["dataset"]: r["outcome"]
            for r in rows
            if r["model"] == m and r["delta"] is not None
        }
        for m in models
    }
    shared = paired[models[0]].keys() & paired[models[1]].keys()
    return dict(
        shared_paired_datasets=len(shared),
        same_direction=sum(
            (paired[models[0]][d] == paired[models[1]][d] for d in shared)
        ),
    )


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tex_name(name):
    return name.replace("_", "\\_")


def delta_cell(row):
    value = row["delta"] if row["delta"] is not None else row["reference_delta"]
    if value is None:
        return "--"
    text = f"{value:.4f}" if value == 0 or abs(value) >= 5e-05 else f"{value:.2e}"
    return text + ("*" if row["delta"] is None else "")


def write_tables(output, rows, summary, models):
    template = TEMPLATES / "pca_deltas.tex"
    body = []
    for dataset in dict.fromkeys((r["dataset"] for r in rows)):
        part = {r["model"]: r for r in rows if r["dataset"] == dataset}
        features = part[models[0]]["num_features"]
        samples = part[models[0]]["num_samples"]
        task = TASK_ABBREVIATIONS[part[models[0]]["task"]]
        display_name = {"tahoe_batch_integration_14plates": "tahoe_batch_integr."}.get(
            dataset, dataset
        )
        label = "\\texttt{" + tex_name(display_name) + "}"
        body.append(
            label
            + " & "
            + task
            + " & "
            + f"{samples:,} & {features:,}"
            + " & "
            + " & ".join((delta_cell(part[m]) for m in models))
            + " \\\\"
        )
    headers = "Dataset & Task & Samples & Features &\n" + " &\n".join(
        "\\makecell{" + tex_name(model) + "~$\\Delta$}" for model in models
    )
    table = (
        template.read_text()
        .replace("PCA_TABLE_COLUMNS", "lcrr" + "r" * len(models))
        .replace("PCA_TABLE_HEADERS", headers)
        .replace("PCA_TABLE_SPAN", str(4 + len(models)))
        .replace("PCA_TABLE_ROWS", "\n".join(body))
    )
    (output / "tfm_pcs_table.tex").write_text(table)
    lines = [
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & PCA results & Paired & PCA better & PCA worse & Tied \\\\",
        "\\midrule",
    ]
    for row in summary:
        lines.append(
            " & ".join(
                (
                    tex_name(str(row[k]))
                    for k in [
                        "model",
                        "pca_available",
                        "paired",
                        "better",
                        "worse",
                        "tied",
                    ]
                )
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (output / "tfms_pca_summary.tex").write_text("\n".join(lines))


def plot_outcomes(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, max(3.6, 1.8 + 0.6 * len(summary))))
    categories = [
        ("better", "PCA better", "#2166ac"),
        ("worse", "PCA worse", "#d95f02"),
        ("tied", "Tied", "#777777"),
        ("pca_only", "PCA only", "#1b9e77"),
        ("no_pca", "No PCA test result", "#dedede"),
    ]
    left = [0] * len(summary)
    for key, label, color in categories:
        values = [
            r["main_only"] + r["neither"] if key == "no_pca" else r[key]
            for r in summary
        ]
        if not any(values):
            continue
        ax.barh(
            range(len(summary)),
            values,
            left=left,
            label=label,
            color=color,
            height=0.54,
            edgecolor="white",
            linewidth=0.8,
        )
        for i, (start, width) in enumerate(zip(left, values)):
            if width:
                ax.text(
                    start + width / 2,
                    i,
                    str(width),
                    ha="center",
                    va="center",
                    color="white" if key in ("better", "worse") else "#111111",
                    fontsize=10,
                )
        left = [a + b for a, b in zip(left, values)]
    ax.set_yticks(range(len(summary)), [r["model"] for r in summary])
    ax.invert_yaxis()
    ax.set_xlim(0, max((r["eligible"] for r in summary)))
    ax.set_xticks(range(0, max((r["eligible"] for r in summary)) + 1, 6))
    ax.set_xlabel("Datasets")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.96, bottom=0.43)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(
            output / f"pca_paired_outcomes.{suffix}", bbox_inches="tight", dpi=220
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["TabICLv2", "TabPFNv2.5", "RealTabPFNv2.5"]
    )
    parser.set_defaults(component_cap=500)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Model names must be unique")
    ensure_results(args.results_root)
    inventory = dataset_inventory()
    rows, sources = collect(
        args.results_root, inventory, args.models, args.component_cap
    )
    summary = summarize(rows, models=args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    write_csv(args.output / "comparisons.csv", rows)
    write_csv(
        args.output / "paired_comparisons.csv",
        [r for r in rows if r["delta"] is not None],
        fields,
    )
    write_csv(
        args.output / "rf_reference_comparisons.csv",
        [r for r in rows if r["reference"] is not None],
        fields,
    )
    write_csv(args.output / "summary.csv", summary)
    write_tables(args.output, rows, summary, models=args.models)
    plot_outcomes(args.output, summary)
    for path, expected in sources.items():
        if digest(args.results_root / path) != expected:
            raise ValueError("Consolidated evaluation changed during PCA analysis")
    manifest = dict(
        scope="all_current_datasets_above_2000_features",
        variant="tuned_with_default_fallback",
        variant_policy="Tuned when available, otherwise the same model Default, independently in each arm",
        component_cap=args.component_cap,
        model_dataset_pairs=len(rows),
        models=list(args.models),
        summaries=summary,
        pfn_direction_agreement=pfn_direction_agreement(rows),
        cv_policy="mean_of_all_five_stored_outer_test_scores_before_dataset_comparison",
        improvement_policy="strict_oriented_difference_with_zero_as_tie",
        rf_policy="Default_only_when_the_corresponding_TFM_has_no_usable_Main_result",
        a100_retries_included=False,
        source_reports_sha256=sources,
    )
    (args.output / "analysis.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
