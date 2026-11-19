"""Hpo history analysis."""

from __future__ import annotations
from tabbiomed.analysis.paths import PLAN, RESULTS
from tabbiomed.analysis.downloads import ensure_results
from tabbiomed.analysis.datasets import load_benchmark
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tabbiomed.results.hpo_read import read_history
from tabbiomed.results.schema import metric_direction


CAPS = (25, 50, 100, 200)


def read_case(record, datasets):
    return read_history(
        RESULTS, record["model"], record["dataset"], datasets[record["dataset"]]["task"]
    )


def dataset_milestones(frame):
    rows = []
    for (model, dataset, cap), group in frame.groupby(["model", "dataset", "cap"]):
        first = group.iloc[0]
        complete = len(group) == first.n_folds and group.status.eq("success").all()
        row = dict(
            model=model,
            dataset=dataset,
            cap=int(cap),
            task=first.task,
            metric=first.metric,
            complete=bool(complete),
            validation=float(group.validation.mean()) if complete else np.nan,
            test=float(group.test.mean()) if complete else np.nan,
            actual_attempts=float(group.actual_attempts.mean()),
            trajectory_available=not group.status.eq(
                "selection_clock_unavailable"
            ).any(),
            cap_reached=bool(len(group) == first.n_folds and group.cap_reached.all()),
        )
        for key in ("validation_log_loss", "test_log_loss"):
            row[key] = (
                float(group[key].mean())
                if complete and key in group and group[key].notna().all()
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(frame):
    rows, pairs = ([], [])
    for model, group in frame.groupby("model"):
        val = group.pivot(index="dataset", columns="cap", values="validation").reindex(
            columns=CAPS
        )
        test = group.pivot(index="dataset", columns="cap", values="test").reindex(
            columns=CAPS
        )
        common = val.notna().all(axis=1) & test.notna().all(axis=1)
        if group.groupby("dataset").metric.nunique(dropna=False).ne(1).any():
            raise ValueError("HPO budgets must use the same metric within each dataset")
        directions = group.groupby("dataset").metric.first().map(
            lambda metric: 1 if metric_direction(metric) == "maximize" else -1
        )
        for cap in CAPS:
            valid = (
                val[25].notna()
                & val[cap].notna()
                & test[25].notna()
                & test[cap].notna()
            )
            for dataset in val.index[valid]:
                direction = directions[dataset]
                dv = direction * (val.loc[dataset, cap] - val.loc[dataset, 25])
                dt = direction * (test.loc[dataset, cap] - test.loc[dataset, 25])
                pairs.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        cap=cap,
                        validation_improves=dv > 1e-12,
                        test_worsens=dt < -1e-12,
                        overtuning=dv > 1e-12 and dt < -1e-12,
                    )
                )
            matched = val.index[common]
            direction = directions.reindex(matched)
            dv = direction * (val.loc[matched, cap] - val.loc[matched, 25])
            dt = direction * (test.loc[matched, cap] - test.loc[matched, 25])
            rows.append(
                dict(
                    model=model,
                    cap=cap,
                    n=len(matched),
                    validation_net_win=100
                    * ((dv > 1e-12).mean() - (dv < -1e-12).mean())
                    if len(matched)
                    else np.nan,
                    test_net_win=100 * ((dt > 1e-12).mean() - (dt < -1e-12).mean())
                    if len(matched)
                    else np.nan,
                )
            )
    return (pd.DataFrame(rows), pd.DataFrame(pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    ensure_results()
    plan = load_benchmark(PLAN)
    datasets = plan["datasets"]
    records = [
        dict(model=model, dataset=dataset)
        for model in plan["models"]
        for dataset in datasets
        if (RESULTS / model / dataset / "report.json").is_file()
    ]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    official, budgets, hashes, issues, errors = ([], [], {}, [], [])

    def retain(result):
        a, b, h, e = result
        official.extend(a)
        budgets.extend(b)
        hashes.update(h)
        issues.extend(e)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(read_case, record, datasets): record for record in records
        }
        for i, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append(
                    dict(
                        model=record["model"],
                        dataset=record["dataset"],
                        error=repr(exc),
                    )
                )
                print("Input error:", errors[-1], flush=True)
                continue
            retain(result)
            if i % 100 == 0:
                print(f"Read {i}/{len(records)} model/dataset reports", flush=True)
    official, budgets = (pd.DataFrame(official), pd.DataFrame(budgets))
    frame = dataset_milestones(budgets)
    curves, pairs = paired_statistics(frame)
    official.to_csv(output / "realized_searches.csv", index=False)
    budgets.to_csv(output / "milestones_by_fold.csv", index=False)
    frame.to_csv(output / "milestones_by_dataset.csv", index=False)
    curves.to_csv(output / "budget_curves.csv", index=False)
    pairs.to_csv(output / "overtuning_pairs.csv", index=False)
    (output / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (output / "issues.json").write_text(json.dumps(issues, indent=2) + "\n")
    (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    print("Issues:", len(issues), flush=True)
    print("Fold statuses:", budgets.status.value_counts().to_dict(), flush=True)
    print(
        "Complete four-cap cohorts:",
        curves.loc[curves.cap == 25, ["model", "n"]].to_dict("records"),
        flush=True,
    )
    if errors:
        raise SystemExit(f"Review {len(errors)} input errors before using the analysis")


if __name__ == "__main__":
    main()
