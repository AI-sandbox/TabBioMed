import hashlib
import json
from pathlib import Path

import numpy as np

from ..budgets import completed_within_deadline
from .schema import metric_direction

CAPS = (25, 50, 100, 200)


def units(report):
    folds = report.get("cv_fold_metrics")
    if not folds:
        return {None: report}
    result = {u["fold_id"]: u for u in folds}
    if len(result) != len(folds) or set(result) != set(range(report["n_outer_folds"])):
        raise ValueError("CV results must contain every stored outer fold exactly once")
    return result


def load_trials(path, folds):
    result = {fold: {} for fold in folds}
    with Path(path).open() as stream:
        for line in stream:
            value = json.loads(line)
            fold, trial = value["fold_id"], value["trial_id"]
            if (
                fold not in result
                or type(trial) is not int
                or trial < -1
                or trial in result[fold]
            ):
                raise ValueError("Invalid or duplicate HPO trial identity")
            result[fold][trial] = value
    return result


def parameter_key(parameters):
    return json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def finite(value):
    return (
        isinstance(value, (int, float))
        and (not isinstance(value, bool))
        and np.isfinite(value)
    )


def select_prefix(trials, candidate, metric, cap, cutoff):
    direction = 1 if metric_direction(metric) == "maximize" else -1
    eligible = [
        t
        for t in trials
        if t["trial_id"] < cap
        and t["status"] == "success"
        and finite(t.get("finished_seconds"))
        and t["finished_seconds"] <= cutoff
        and finite(t["validation_metrics"].get(metric))
    ]
    best = min(
        eligible,
        key=lambda t: (-direction * t["validation_metrics"][metric], t["trial_id"]),
        default=None,
    )
    value = (candidate.get("validation_metrics") or {}).get(metric)
    source_finished = candidate["finished_seconds"] if "finished_seconds" in candidate else 0.0
    if (
        candidate.get("status") == "success"
        and completed_within_deadline(source_finished, cutoff)
        and finite(value)
    ):
        if (
            best is None
            or direction * value >= direction * best["validation_metrics"][metric]
        ):
            return candidate
    return best


def valid_test(block, metric):
    return (
        isinstance(block, dict)
        and not block.get("failure")
        and not block.get("error")
        and not block.get("post_hpo_test_error")
        and block.get("fixed_split_test_evaluated")
        and finite((block.get("test_metrics") or {}).get(metric))
    )


def read_history(root, model, dataset, task):
    path = Path(root) / model / dataset / "report.json"
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    public = read(path)
    if not public["hpo_enabled"]:
        return [], [], {}, []
    timing_path = path.with_name("timing.json")
    timing = read(timing_path)
    history = path.parent / "hpo"
    selections = read(history / "selections.json")
    if (
        selections["schema_version"] != 1
        or selections["model"] != public["model"]
        or selections["dataset"] != dataset
        or selections["primary_metric"] != public["primary_metric"]
        or selections["report_sha256"] != hashes[str(path)]
        or timing["report_sha256"] != hashes[str(path)]
        or selections["timing_sha256"] != hashes[str(timing_path)]
    ):
        raise ValueError("HPO export, timing and Main report identities differ")
    pu = units(public)
    all_trials = load_trials(history / "trials.jsonl", pu)
    hashes[str(history / "trials.jsonl")] = hashlib.sha256(
        (history / "trials.jsonl").read_bytes()
    ).hexdigest()
    selected_units = {u["fold_id"]: u for u in selections["units"]}
    if len(selected_units) != len(selections["units"]) or set(selected_units) != set(
        pu
    ):
        raise ValueError("HPO selections and Main have different outer folds")
    official, milestones = [], []
    metric = public["primary_metric"]
    for clock in timing["units"]:
        fold = clock["fold_id"]
        values = all_trials[fold]
        candidate = values.get(-1, {})
        trials = [v for i, v in sorted(values.items()) if i >= 0]
        hpo = clock["hpo"]
        prefix = dict(
            model=model,
            dataset=dataset,
            task=task,
            metric=metric,
            fold_id=-1 if fold is None else fold,
            n_folds=len(pu),
        )
        cutoff = hpo["search_cutoff_seconds"]
        entries = {
            entry["attempt_cap"]: entry for entry in selected_units[fold]["budgets"]
        }
        if len(entries) != 4 or set(entries) != set(CAPS):
            raise ValueError("Expected exactly the four recorded configuration caps")
        if not finite(cutoff):
            count = hpo.get("attempted_configurations")
            official.append(
                dict(
                    prefix,
                    attempted=count,
                    scored=0 if count == 0 else None,
                    official_cap=hpo["official_attempt_cap"],
                    budget_seconds=hpo.get("budget_seconds"),
                    selection_seconds=hpo.get("selection_wall_seconds"),
                    recorded_attempts=None,
                )
            )
            for cap in CAPS:
                if entries[cap]["status"] != "selection_clock_unavailable":
                    raise ValueError(
                        "Missing selection clock has a scored budget result"
                    )
                milestones.append(
                    dict(
                        prefix,
                        cap=cap,
                        actual_attempts=None,
                        cap_reached=False,
                        validation=None,
                        test=None,
                        selected_trial_id=None,
                        status="selection_clock_unavailable",
                    )
                )
            continue
        if (
            candidate.get("duration_seconds") is not None
            and candidate["duration_seconds"] > hpo["budget_seconds"]
        ):
            candidate = dict(candidate, status="timeout")
        attempted = [
            t
            for t in trials
            if (
                t["started_seconds"]
                if t["started_seconds"] is not None
                else t.get("finished_seconds", np.inf)
            )
            < cutoff
        ]
        successful = [
            t
            for t in attempted
            if t["status"] == "success"
            and finite(t.get("finished_seconds"))
            and t["finished_seconds"] <= cutoff
            and finite(t["validation_metrics"].get(metric))
        ]
        official.append(
            dict(
                prefix,
                attempted=hpo["attempted_configurations"],
                scored=sum(
                    t["trial_id"] < hpo["official_attempt_cap"] for t in successful
                ),
                official_cap=hpo["official_attempt_cap"],
                budget_seconds=hpo["budget_seconds"],
                selection_seconds=hpo.get("selection_wall_seconds"),
                recorded_attempts=len(attempted),
            )
        )
        for cap in CAPS:
            item = entries[cap]
            selected = select_prefix(trials, candidate, metric, cap, cutoff)
            expected_id = selected["trial_id"] if selected else None
            if (
                item["selected_trial_id"] != expected_id
                or item["actual_attempts"] != min(cap, len(attempted))
                or item["cap_reached"] != (len(attempted) >= cap)
            ):
                raise ValueError(
                    "Stored budget selection differs from its validation trajectory"
                )
            row = dict(
                prefix,
                cap=cap,
                actual_attempts=item["actual_attempts"],
                cap_reached=item["cap_reached"],
                validation=None,
                test=None,
                selected_trial_id=expected_id,
                status=item["status"],
                validation_log_loss=None,
                test_log_loss=None,
            )
            if selected:
                if parameter_key(item["validation_metrics"]) != parameter_key(
                    selected["validation_metrics"]
                ):
                    raise ValueError(
                        "Selected validation metrics differ from their trial"
                    )
                row["validation"] = selected["validation_metrics"][metric]
                if item["status"] == "success":
                    row.update(
                        test=item["test_metrics"][metric],
                        validation_log_loss=selected["validation_metrics"].get(
                            "log_loss"
                        ),
                        test_log_loss=item["test_metrics"].get("log_loss"),
                    )
                if cap == hpo["official_attempt_cap"] and valid_test(
                    pu[fold].get("best"), metric
                ):
                    best = pu[fold]["best"]
                    if parameter_key(best.get("parameters")) != parameter_key(
                        selected.get("parameters")
                    ):
                        raise ValueError(
                            "Official HPO configuration differs from Main Tuned"
                        )
                    if row["test"] is not None and not np.isclose(
                        row["test"],
                        best["test_metrics"][metric],
                        rtol=1e-10,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            "Official HPO test metric differs from Main Tuned"
                        )
            milestones.append(row)
    return official, milestones, hashes, []
