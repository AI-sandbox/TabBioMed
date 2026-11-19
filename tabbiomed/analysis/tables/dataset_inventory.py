"""Dataset domain counts and pretraining overlap tables."""

from tabbiomed.analysis.paths import PLAN, CONFIG, PROCESSED, TEMPLATES

import argparse
import json
from pathlib import Path

import pandas as pd

from tabbiomed.analysis.common import canon_domain
from tabbiomed.analysis.data_tables import domain_rows, overlap_rows, render
from tabbiomed.analysis.datasets import load_benchmark, load_metadata
from tabbiomed.analysis.pretraining_overlap import pretraining_overlaps


def generate(frame, domains, overlaps, templates):
    domains = [
        {**domain, "summary_domain": canon_domain(domain["domain"])}
        for domain in domains
    ]
    if set(frame.domain) - {domain["summary_domain"] for domain in domains}:
        raise ValueError("Unexpected dataset domain")
    return {
        "tab:dataset_domains": render(
            templates / "domains.tex", rows="\n".join(domain_rows(frame, domains))
        ),
        "tab:pretraining_overlap": render(
            templates / "overlap.tex", rows="\n".join(overlap_rows(overlaps))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED)
    args = parser.parse_args()
    plan = load_benchmark(args.plan, args.processed_path)
    names = list(plan["datasets"])
    frame = pd.DataFrame(
        {
            "dataset": name,
            "task": plan["datasets"][name]["task"],
            "domain": canon_domain(load_metadata(name, args.processed_path)["label"]),
        }
        for name in names
    )
    overlaps = pretraining_overlaps()
    if not set(overlaps["all"]) <= set(names):
        raise ValueError("Overlap inventory contains datasets outside this scope")
    domains = json.loads((CONFIG / "dataset_domains.json").read_text())
    tables = generate(frame, domains, overlaps, TEMPLATES / "dataset_inventory")
    args.output.mkdir(parents=True, exist_ok=True)
    for label, table in tables.items():
        filename = label.replace(":", "_") + ".tex"
        path = args.output / filename
        path.write_text(table)
        print(path)


if __name__ == "__main__":
    main()
