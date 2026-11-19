"""Measured selection and evaluation costs."""

import json
import math
from pathlib import Path

from .io import digest_bytes, read_json


def seconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return float(value)
    return None


def sum_known(*values):
    return sum(values) if all(v is not None for v in values) else None


def canonical(value, key=None):
    """Apply the checkpoint and nonfinite-value conversions used by the public export."""
    if isinstance(value, dict):
        return {k: canonical(v, k) for k, v in value.items() if k != "custom_model_dir"}
    if isinstance(value, list):
        return [canonical(v, key) for v in value]
    if key == "base_tabpfn_path" and isinstance(value, str):
        return Path(value).name
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def units(report, expected_folds=None):
    if report.get("cv_fold_metrics"):
        out = {u["fold_id"]: u for u in report["cv_fold_metrics"]}
        expected_folds = expected_folds or report.get("n_outer_folds") or len(out)
        if len(out) != len(report["cv_fold_metrics"]) or set(out) != set(
            range(expected_folds)
        ):
            raise ValueError(
                "CV results must contain every stored outer fold exactly once"
            )
        return out
    return {None: report}


def fitted_rows(block, counts):
    merged = (block or {}).get("train_val_merged_for_fit")
    return (
        counts["train"] + (counts["validation"] if merged else 0)
        if isinstance(merged, bool)
        else None
    )


def load_history(path):
    if not path.is_file():
        return [], None, False
    raw = path.read_bytes()
    lines = raw.splitlines()
    events, truncated = [], False
    for number, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise ValueError(
                    f"Invalid interior history record: {path}:{number + 1}"
                )
            truncated = True
    return events, digest_bytes(raw), truncated


def trial_records(events, fold_id, search_id=None):
    starts = [
        e
        for e in events
        if e.get("event") == "search_started" and e.get("fold_id") == fold_id
    ]
    if not starts:
        return [], [], None
    search_id = search_id or starts[-1]["search_id"]
    if search_id not in {s["search_id"] for s in starts}:
        raise ValueError("Selection points to a different HPO search")
    chosen = [e for e in events if e.get("search_id") == search_id]
    records = {}
    for event in chosen:
        if "trial_id" not in event:
            continue
        trial_id = event["trial_id"]
        record = records.setdefault(trial_id, {"trial_id": trial_id})
        if event["event"] == "trial_started":
            record["started_at_hpo_seconds"] = event["hpo_elapsed_seconds"]
        record.update(event)
    return [records[k] for k in sorted(records)], chosen, search_id


def selection_timing(events, fold_id, public_unit, config, selection, candidate):
    """Charge the attempted official prefix, including a partially consumed deadline trial."""
    cap = selection.get("attempted_configuration_cap", config.get("n_trials"))
    if type(cap) is not int or cap <= 0:
        raise ValueError("Official attempt cap must be a positive integer")
    configured_cap = config.get("n_trials")
    if configured_cap is not None and cap > configured_cap:
        raise ValueError("Official attempt cap exceeds the configured search")
    budget = seconds(selection.get("selection_budget_seconds"))
    if budget is None:
        budget = seconds(
            config.get(
                "cv_time_limit_per_fold" if fold_id is not None else "time_limit"
            )
        )
    charge = seconds(
        selection.get("validation_seconds", candidate.get("validation_seconds"))
    )
    cutoff = seconds(selection.get("hpo_cutoff_seconds"))
    if cutoff is None and budget is not None and charge is not None:
        cutoff = max(0.0, budget - charge)
    if budget is not None and charge is not None and cutoff is not None:
        if not math.isclose(cutoff, max(0.0, budget - charge), abs_tol=1e-6):
            raise ValueError("Default reservation and official search cutoff disagree")
    records, chosen, search_id = trial_records(
        events, fold_id, selection.get("search_id")
    )
    prefix = [t for t in records if t["trial_id"] < cap]
    attempted = [
        t
        for t in prefix
        if cutoff is not None
        and seconds(t.get("started_at_hpo_seconds")) is not None
        and t["started_at_hpo_seconds"] < cutoff
    ]
    expected_count = selection.get(
        "retained_attempts",
        (public_unit.get("tuning") or {}).get("n_attempted_trials"),
    )
    if (
        records
        and cutoff is not None
        and expected_count is not None
        and len(attempted) != expected_count
    ):
        raise ValueError(
            f"Official attempted prefix disagrees: {len(attempted)} != {expected_count}"
        )

    boundary = next(
        (
            t
            for t in prefix
            if t["trial_id"] == cap - 1 and t.get("event") == "trial_finished"
        ),
        None,
    )
    finishes = [e for e in chosen if e.get("event") == "search_finished"]
    endpoint = seconds(boundary.get("hpo_elapsed_seconds")) if boundary else None
    if endpoint is None and finishes:
        if all(t["trial_id"] < cap for t in records):
            endpoint = seconds(finishes[-1].get("hpo_elapsed_seconds"))
    if cutoff == 0:
        elapsed = 0.0
    elif endpoint is not None and cutoff is not None:
        elapsed = min(endpoint, cutoff)
    else:
        elapsed = None

    def complete_sum(field):
        if not attempted:
            return 0.0 if elapsed == 0 else None
        if any(
            t.get("event") != "trial_finished"
            or seconds(t.get(field)) is None
            or t["hpo_elapsed_seconds"] > cutoff
            for t in attempted
        ):
            return None
        return sum(float(t[field]) for t in attempted)

    measured_charge = charge
    observed = [seconds(t.get("hpo_elapsed_seconds")) for t in prefix]
    observed = [t for t in observed if t is not None]
    lower_bound = (
        elapsed
        if elapsed is not None
        else min(max(observed), cutoff)
        if observed and cutoff is not None
        else None
    )
    details = dict(
        official_attempt_cap=cap,
        budget_seconds=budget,
        search_cutoff_seconds=cutoff,
        attempted_configurations=len(attempted)
        if records and cutoff is not None
        else expected_count,
        reference_validation_charge_seconds=charge,
        reference_validation_measured_seconds=measured_charge,
        official_search_wall_seconds=elapsed,
        official_search_lower_bound_seconds=lower_bound,
        search_clock_status=(
            "available"
            if elapsed is not None
            else "incomplete_history"
            if chosen
            else "missing_history"
        ),
        search_fit_seconds=complete_sum("train_wall_time_seconds"),
        search_validation_prediction_seconds=complete_sum(
            "val_inference_wall_time_seconds"
        ),
        search_trial_wall_seconds=complete_sum("trial_wall_time_seconds"),
        selection_wall_seconds=sum_known(elapsed, measured_charge),
        attempts_beyond_cap=sum(t["trial_id"] >= cap for t in records),
        timing_status="available"
        if elapsed is not None and measured_charge is not None
        else "unavailable",
    )
    return details, search_id


def evaluation_timing(block, primary_metric, tuned=False):
    block = block or {}
    field = (
        "post_hpo_train_wall_time_seconds"
        if tuned and block.get("post_hpo_train_wall_time_seconds") is not None
        else "train_wall_time_seconds"
    )
    fit = seconds(block.get(field))
    val_field = (
        "post_hpo_val_inference_wall_time_seconds"
        if field == "post_hpo_train_wall_time_seconds" and block.get("inner_cv_metrics")
        else "val_inference_wall_time_seconds"
    )
    validation = seconds(block.get(val_field))
    if block.get("train_val_merged_for_fit") is True:
        validation = 0.0
    metric = (block.get("test_metrics") or {}).get(primary_metric)
    success = isinstance(metric, (int, float)) and math.isfinite(metric)

    result = dict(
        evaluation_status="success" if success else "failed",
        final_fit_seconds=fit,
        final_validation_prediction_seconds=validation,
        final_fit_and_validation_seconds=sum_known(fit, validation),
        test_prediction_seconds=seconds(block.get("test_inference_wall_time_seconds")),
        train_val_merged_for_fit=block.get("train_val_merged_for_fit"),
        training_cost_seconds=None,
    )
    return result


def read_selection(run, fold_id):
    stage = run / "selection" / ("fixed" if fold_id is None else f"fold_{fold_id}")
    evidence = {}
    values = []
    for name in ("reference.json", "result.json"):
        path = stage / name
        if not path.is_file():
            values.append({})
        else:
            value, sha = read_json(path)
            evidence[str(path)] = sha
            values.append(value)
    candidate, selection = values
    return candidate, selection, evidence
