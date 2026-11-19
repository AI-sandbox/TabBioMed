"""Validation-only selection checks and compact evaluation outcomes."""

import math
import re

from ..budgets import completed_within_deadline
from .io import local_file
from .schema import clean, compact_failure, metric_direction
from .timing import seconds, selection_timing, trial_records


def test_available(block, metric):
    if not isinstance(block, dict) or not isinstance(block.get("test_metrics"), dict):
        return False
    value = block["test_metrics"].get(metric)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and block.get("fixed_split_test_evaluated") is True
        and not block.get("failure")
        and not block.get("error")
        and not block.get("post_hpo_test_error")
    )


def failure_record(value):
    if not isinstance(value, dict) or not value.get("reason"):
        raise ValueError("Failure records require a reason")
    result = compact_failure(value)
    if set(value) - {"reason", "memory_limit_gib"}:
        raise ValueError("Failure records only accept reason and memory_limit_gib")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", result["reason"]):
        raise ValueError("Failure reasons must be short identifiers")
    if "memory_limit_gib" in result and seconds(result["memory_limit_gib"]) is None:
        raise ValueError("Invalid failure memory limit")
    return result


def inferred_failure(block):
    block = block or {}
    if block.get("failure"):
        return failure_record(compact_failure(block["failure"]))
    if block.get("status") == "timeout":
        return {"reason": "time_limit"}
    error = str(block.get("error") or block.get("post_hpo_test_error") or "").lower()
    if any(token in error for token in ("sigsegv", "segmentation fault", "signal 11")):
        return {"reason": "segmentation_fault"}
    if "out of memory" in error or "out_of_memory" in error:
        return {
            "reason": "gpu_oom"
            if any(t in error for t in ("cuda", "gpu"))
            else "out_of_memory"
        }
    if any(
        token in error for token in ("timeout", "time limit", "reached its deadline")
    ):
        return {"reason": "time_limit"}
    return {"reason": "evaluation_failed" if error else "not_available"}


def read_selection(run, fold, evidence):
    directory = run / "selection" / ("fixed" if fold is None else f"fold_{fold}")
    result = []
    for name in ("reference.json", "result.json"):
        path = evidence.optional(directory / name)
        result.append(
            evidence.read(local_file(run, str(path.relative_to(run))))
            if path is not None
            else {}
        )
    return result


def verify_selection(unit, candidate, selection, events, fold, metric, config):
    best = unit.get("best") or {}
    if test_available(best, metric):
        evaluation = selection.get("evaluation") or {}
        if not selection or not candidate:
            raise ValueError(
                "Successful Tuned evaluation lacks saved selection evidence"
            )
        for key in ("parameters", "validation_metrics", "test_metrics", "trial_id"):
            if clean(evaluation.get(key)) != clean(best.get(key)):
                raise ValueError(f"Selection and report disagree on {key}")
    if not selection:
        return {}, {}
    budget = config.get("cv_time_limit_per_fold" if fold is not None else "time_limit")
    if selection.get("selection_budget_seconds", budget) != budget:
        raise ValueError("Saved selection budget differs from the configured budget")
    if (
        candidate.get("validation_seconds") is not None
        and selection.get("validation_seconds", candidate["validation_seconds"])
        != candidate["validation_seconds"]
    ):
        raise ValueError("Default validation time differs between selection records")
    parameters = (unit.get("default") or {}).get("parameters")
    if parameters is not None and clean(candidate.get("parameters")) != clean(
        parameters
    ):
        raise ValueError(
            "Default candidate parameters differ from the recorded Default"
        )
    hpo, _ = selection_timing(events, fold, {}, config, selection, candidate)
    if not selection.get("selected"):
        if test_available(best, metric):
            raise ValueError(
                "Successful Tuned result has no validation-selected configuration"
            )
        return hpo, {}
    if not events and not (
        selection.get("default_selected") is True
        and selection.get("retained_attempts") == 0
    ):
        raise ValueError("A sampled selection requires its saved HPO history")
    records, _, _ = trial_records(events, fold, selection.get("search_id"))
    cap, cutoff = hpo["official_attempt_cap"], hpo["search_cutoff_seconds"]
    if cutoff is None:
        raise ValueError("Selection is missing its time cutoff")
    eligible = [
        trial
        for trial in records
        if trial.get("status") == "success"
        and trial["trial_id"] < cap
        and seconds(trial.get("hpo_elapsed_seconds")) is not None
        and trial["hpo_elapsed_seconds"] <= cutoff
        and seconds(trial.get("started_at_hpo_seconds")) is not None
        and trial["started_at_hpo_seconds"] < cutoff
        and finite_metric(trial.get("validation_metrics"), metric)
    ]
    maximize = metric_direction(metric) == "maximize"
    sampled = min(
        eligible,
        key=lambda trial: (
            (-1 if maximize else 1) * trial["validation_metrics"][metric],
            trial["trial_id"],
        ),
        default=None,
    )
    charge = hpo["reference_validation_charge_seconds"]
    source_finished = (
        candidate.get("source_completion_seconds")
        if "source_trial_id" in candidate
        else 0.0
    )
    default_ok = (
        candidate.get("status") == "success"
        and charge is not None
        and hpo["budget_seconds"] is not None
        and charge <= hpo["budget_seconds"]
        and completed_within_deadline(source_finished, cutoff)
        and finite_metric(candidate.get("validation_metrics"), metric)
    )
    default_selected = default_ok and (
        sampled is None
        or (
            candidate["validation_metrics"][metric]
            >= sampled["validation_metrics"][metric]
            if maximize
            else candidate["validation_metrics"][metric]
            <= sampled["validation_metrics"][metric]
        )
    )
    expected = candidate if default_selected else sampled
    selected = selection["selected"]
    if expected is None or selection.get("default_selected") is not bool(
        default_selected
    ):
        raise ValueError("Saved selection disagrees with validation-only selection")
    trial_id = -1 if default_selected else expected["trial_id"]
    for key, value in (
        ("parameters", expected.get("parameters")),
        ("validation_metrics", expected.get("validation_metrics")),
        ("trial_id", trial_id),
    ):
        if clean(selected.get(key)) != clean(value):
            raise ValueError(
                f"Saved selection disagrees with eligible validation scores on {key}"
            )
        if test_available(best, metric) and clean(best.get(key)) != clean(value):
            raise ValueError(
                f"Tuned refit differs from its selected configuration on {key}"
            )
    successful = len(eligible)
    return hpo, {"n_successful_trials": successful}


def finite_metric(metrics, metric):
    value = (metrics or {}).get(metric)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def aggregate(folds, variant, field):
    values = [(fold.get(variant) or {}).get(field) or {} for fold in folds]
    common = set.intersection(*(set(value) for value in values))
    result = {}
    for metric in sorted(common):
        if all(finite_metric(value, metric) for value in values):
            numbers = [value[metric] for value in values]
            mean = sum(numbers) / len(numbers)
            result.update(
                {
                    metric: mean,
                    f"{metric}_std": math.sqrt(
                        sum((number - mean) ** 2 for number in numbers) / len(numbers)
                    ),
                    f"{metric}_n_folds": len(numbers),
                }
            )
    return result
