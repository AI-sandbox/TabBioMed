"""Generate the paper analyses in dependency order."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def paper_commands(output):
    return [
        (
            "tabbiomed.analysis.tables.dataset_inventory",
            "--output",
            output / "dataset_tables",
        ),
        ("tabbiomed.analysis.figures.dataset_inventory",),
        ("tabbiomed.analysis.tables.model_summary",),
        ("tabbiomed.analysis.figures.cd",),
        ("tabbiomed.analysis.figures.elo", "--output", output / "elo"),
        ("tabbiomed.analysis.figures.elo_regimes",),
        ("tabbiomed.analysis.figures.elo_domains",),
        ("tabbiomed.analysis.figures.elo_features",),
        ("tabbiomed.analysis.figures.source_dependence",),
        ("tabbiomed.analysis.figures.pareto",),
        ("tabbiomed.analysis.figures.runtime",),
        ("tabbiomed.analysis.figures.upset",),
        ("tabbiomed.analysis.figures.win_rates",),
        ("tabbiomed.analysis.figures.calibration", "--name", "sexcheck50k"),
        ("tabbiomed.analysis.studies.pretraining",),
        ("tabbiomed.analysis.figures.pretraining",),
        ("tabbiomed.analysis.studies.cross_benchmark",),
        (
            "tabbiomed.analysis.studies.binary_metrics",
            "--output",
            output / "binary_metrics",
        ),
        ("tabbiomed.analysis.studies.failures",),
        ("tabbiomed.analysis.studies.a100",),
        ("tabbiomed.analysis.studies.pca", "--output", output / "pca"),
        ("tabbiomed.analysis.studies.metafeatures",),
        ("tabbiomed.analysis.studies.hpo_history", "--output", output / "hpo_budgets"),
        (
            "tabbiomed.analysis.studies.hpo_sensitivity",
            "--input",
            output / "hpo_budgets",
        ),
        ("tabbiomed.analysis.figures.hpo_budgets", "--input", output / "hpo_budgets"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("paper",), default="paper")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.results is not None:
        os.environ["TBM_RESULTS"] = str(args.results.expanduser().resolve())
    if args.output is not None:
        os.environ["TBM_OUTPUT"] = str(args.output.expanduser().resolve())
    output = Path(os.environ.get("TBM_OUTPUT", "outputs")).expanduser().resolve()
    from .paths import CONFIG

    environment = {
        **os.environ,
        "TBM_ANALYSIS_PLAN": str(CONFIG / "benchmark.json"),
        "TBM_MODEL_FAMILIES": str(CONFIG / "model_families.json"),
    }
    for command in paper_commands(output):
        print(command[0], flush=True)
        subprocess.run(
            [sys.executable, "-m", *map(str, command)], check=True, env=environment
        )


if __name__ == "__main__":
    main()
