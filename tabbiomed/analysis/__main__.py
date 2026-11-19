"""Compute benchmark analyses and generate figures and tables."""

import argparse
from importlib import import_module
import os
from pathlib import Path
import sys

COMMANDS = {
    "dataset-tables": ("tables.dataset_inventory", "--output", "dataset_tables"),
    "dataset-figures": (
        "figures.dataset_inventory",
        "--output-dir",
        "dataset_inventory",
    ),
    "model-summary": ("tables.model_summary", "--output", "model_summary"),
    "elo": ("figures.elo", "--output", "elo"),
    "cd": ("figures.cd", "--plots-dir", "cd"),
    "elo-regimes": ("figures.elo_regimes", "--output-dir", "elo_regimes"),
    "elo-domains": ("figures.elo_domains", "--output-dir", "elo_domains"),
    "elo-features": ("figures.elo_features", "--output-dir", "elo_features"),
    "source-dependence": (
        "figures.source_dependence",
        "--output-dir",
        "source_dependence",
    ),
    "pareto": ("figures.pareto", "--output", "pareto"),
    "runtime": ("figures.runtime", "--output-dir", "runtime"),
    "upset": ("figures.upset", "--output-dir", "upset"),
    "win-rates": ("figures.win_rates", "--output-dir", "win_rates"),
    "calibration": ("figures.calibration", "--output-dir", "calibration"),
    "pretraining": ("studies.pretraining", "--output", "pretraining"),
    "pretraining-figures": ("figures.pretraining", "--output", "pretraining_figures"),
    "cross-benchmark": ("studies.cross_benchmark", "--output-dir", "cross_benchmark"),
    "binary-metrics": ("studies.binary_metrics", "--output", "binary_metrics"),
    "failures": ("studies.failures", "--output", "failures"),
    "a100": ("studies.a100", "--output", "a100"),
    "pca": ("studies.pca", "--output", "pca"),
    "metafeatures": ("studies.metafeatures", "--output", "metafeatures"),
    "hpo-history": ("studies.hpo_history", "--output", "hpo_budgets"),
    "hpo-sensitivity": ("studies.hpo_sensitivity", None, "hpo_budgets"),
    "hpo-budgets": ("figures.hpo_budgets", None, "hpo_budgets"),
    "reproduce": ("reproduce", "--output", ""),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    show_help = not argv or argv in (["--help"], ["-h"])
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=show_help, allow_abbrev=False
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--results", type=Path, help="Consolidated benchmark results")
    parser.add_argument("--plan", type=Path, help="Dataset and model comparison scope")
    parser.add_argument(
        "--model-families", type=Path, help="Model families and task eligibility"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Destination for this analysis"
    )
    if show_help:
        parser.print_help()
        return 0
    settings, remaining = parser.parse_known_args(argv)
    for value, name in (
        (settings.results, "TBM_RESULTS"),
        (settings.plan, "TBM_ANALYSIS_PLAN"),
        (settings.model_families, "TBM_MODEL_FAMILIES"),
    ):
        if value is not None:
            os.environ[name] = str(value.expanduser().resolve())
    module, output_flag, subdirectory = COMMANDS[settings.command]
    base = Path(os.environ.get("TBM_OUTPUT", "outputs")).expanduser().resolve()
    if settings.output is not None and output_flag is None:
        parser.error(f"{settings.command} writes beside its --input files, use --input")
    if output_flag is not None and not any(
        item == output_flag or item.startswith(output_flag + "=") for item in remaining
    ):
        destination = settings.output or base / subdirectory
        remaining += [output_flag, str(destination.expanduser().resolve())]
    if output_flag is None and not any(
        item == "--input" or item.startswith("--input=") for item in remaining
    ):
        remaining += ["--input", str(base / subdirectory)]
    implementation = import_module("tabbiomed.analysis." + module)
    previous = sys.argv
    try:
        sys.argv = ["tabbiomed analyze " + settings.command, *remaining]
        return implementation.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
