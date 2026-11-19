"""Small, durable HPO trajectories and validation-selected budget evaluations."""

import copy
import datetime
import json
import math
import os
from pathlib import Path
import time
import tempfile
import uuid

import numpy as np
from optuna.trial import TrialState

from .artifacts import save_predictions, selected_study
from .cli_setup import _set_reproducibility
from .train_eval import (
    _primary_metric_value_ok,
    apply_fixed_split_test_eval_for_best_trial,
)


class _OfficialStudy:
    """Read-only reporting view of the official attempted-configuration prefix."""

    def __init__(self, study, cap):
        self.trials = [t for t in study.trials if t.number < cap]
        self.direction = study.direction

    @property
    def best_trial(self):
        candidates = [
            t
            for t in self.trials
            if t.state == TrialState.COMPLETE
            and t.value is not None
            and math.isfinite(t.value)
        ]
        if not candidates:
            raise ValueError("No successful official HPO trial")
        sign = -1 if self.direction.name == "MAXIMIZE" else 1
        return min(candidates, key=lambda t: (sign * t.value, t.number))


def official_hpo_result(study, records, cap):
    if study is None or not isinstance(cap, int):
        return study, records
    return _OfficialStudy(study, cap), [r for r in records if r["trial_id"] < cap]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported trajectory value: {type(value).__name__}")


def _json_safe(value):
    if isinstance(value, (np.ndarray, np.generic)):
        value = _json_default(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save_budget_report(payload, path):
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(_json_safe(payload), stream, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_test_labels(run_dir, dataset, fold_id=None):
    suffix = "fixed" if fold_id is None else f"fold_{fold_id}"
    path = Path(run_dir) / f"test_labels_{suffix}.npy"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, dataset.y_test)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


class HPOHistory:
    def __init__(self, run_dir, model, dataset, fold_id, seed, metric, origin=None):
        self.path = Path(run_dir) / "hpo_trials.jsonl"
        self.origin = time.monotonic() if origin is None else origin
        if self.path.exists():
            with self.path.open("rb+") as stream:
                data = stream.read()
                if data and not data.endswith(b"\n"):
                    tail = data[data.rfind(b"\n") + 1 :]
                    try:
                        json.loads(tail)
                    except json.JSONDecodeError:
                        stream.truncate(data.rfind(b"\n") + 1)
                    else:
                        stream.write(b"\n")
        self.identity = dict(
            schema_version=1,
            search_id=uuid.uuid4().hex,
            model=model,
            dataset=dataset,
            fold_id=fold_id,
            seed=seed,
            primary_metric=metric,
        )
        self.write("search_started")

    def write(self, event, **fields):
        row = {
            **self.identity,
            "event": event,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hpo_elapsed_seconds": time.monotonic() - self.origin,
            **fields,
        }

        with self.path.open("a") as stream:
            stream.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _load_search_events(run_dir, fold_id):
    path = Path(run_dir) / "hpo_trials.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if stream.read().strip():
                    raise
    searches = [
        r["search_id"]
        for r in rows
        if r.get("event") == "search_started" and r.get("fold_id") == fold_id
    ]
    if not searches:
        return []
    return [row for row in rows if row.get("search_id") == searches[-1]]


def load_hpo_trials(run_dir, fold_id=None):
    """Latest search for this fold; unfinished starts remain interrupted attempts."""
    trials = {}
    for row in _load_search_events(run_dir, fold_id):
        if "trial_id" not in row:
            continue
        tid = row["trial_id"]
        trials[tid] = {**trials.get(tid, {}), **row}
    for row in trials.values():
        if row["event"] != "trial_finished":
            row.update(
                status="interrupted", error="Process stopped before trial completion"
            )
    return [trials[k] for k in sorted(trials)]


def select_budget_trial(trials, metric, attempt_cap, seconds=None):
    """Best successful completed trial within an attempted-config/time prefix."""
    eligible = [
        r
        for r in trials
        if r.get("status") == "success"
        and r["trial_id"] < attempt_cap
        and (seconds is None or r["hpo_elapsed_seconds"] <= seconds)
        and _primary_metric_value_ok(r.get("validation_metrics"), metric)
    ]
    if not eligible:
        return None
    maximize = metric.lower() in ("accuracy", "auc", "roc_auc", "f1", "r2")
    return min(
        eligible,
        key=lambda r: (
            (-1 if maximize else 1) * r["validation_metrics"][metric],
            r["trial_id"],
        ),
    )


def budget_results_path(run_dir, fold_id=None):
    suffix = "" if fold_id is None else f"_fold_{fold_id}"
    return Path(run_dir) / f"hpo_budget_results{suffix}.json"


def budget_results_complete(run_dir, budgets, fold_id=None):
    if not budgets:
        return True
    path = budget_results_path(run_dir, fold_id)
    if not path.exists():
        return False
    entries = json.loads(path.read_text()).get("budgets", [])
    return set(budgets) == {r["attempt_cap"] for r in entries} and all(
        r.get("status") in ("success", "no_successful_trial") for r in entries
    )


def evaluate_hpo_budgets(
    run_dir,
    budgets,
    metric,
    model_class,
    model_args,
    dataset,
    best_rep=None,
    fold_id=None,
):
    if not budgets:
        return
    if any(type(b) is not int or b <= 0 for b in budgets) or len(set(budgets)) != len(
        budgets
    ):
        raise ValueError(
            "HPO evaluation budgets must be distinct positive attempt counts"
        )
    from .selection import select, test_ok

    run_dir = Path(run_dir)
    trials = load_hpo_trials(run_dir, fold_id)
    events = _load_search_events(run_dir, fold_id)
    if not events:
        raise ValueError("Budget evaluation requires the saved HPO trajectory")
    path = budget_results_path(run_dir, fold_id)
    prefix = "" if fold_id is None else f"fold_{fold_id}_"
    labels_path = save_test_labels(run_dir, dataset, fold_id)
    prior = json.loads(path.read_text()) if path.exists() else {}
    cache = {
        r["selected_trial_id"]: r
        for r in prior.get("budgets", [])
        if r.get("status") == "success"
        and not r.get("error")
        and _primary_metric_value_ok(r.get("test_metrics"), metric)
        and r.get("predictions")
        and (run_dir / r["predictions"]).is_file()
    }
    actual = max((r["trial_id"] for r in trials), default=-1) + 1
    context = events[0]
    payload = dict(
        schema_version=1,
        model=context["model"],
        dataset=context["dataset"],
        fold_id=fold_id,
        seed=model_args.seed,
        primary_metric=metric,
        search_id=context["search_id"],
        actual_attempts=actual,
        hpo_elapsed_seconds=events[-1]["hpo_elapsed_seconds"],
        stop_reason=events[-1].get("stop_reason", "interrupted"),
        test_labels=labels_path.name,
        budgets=[],
    )
    if getattr(dataset, "hpo_validation", None):
        payload["hpo_validation"] = dataset.hpo_validation
    if getattr(dataset, "selection_context", None):
        ctx = dataset.selection_context
        payload["selection"] = dict(
            protocol="validation_selection_v1",
            validation_seconds=ctx["candidate"]["validation_seconds"],
            selection_budget_seconds=ctx["budget"],
        )
    if prior.get("search_id") != payload["search_id"]:
        cache = {}

    for cap in sorted(budgets):
        selected = select_budget_trial(trials, metric, cap)
        context_default = getattr(dataset, "selection_context", None)
        if context_default:
            selected = select(
                trials,
                context_default["candidate"],
                metric,
                cap,
                context_default["budget"],
            )["selected"]
        entry = dict(
            attempt_cap=cap,
            cap_reached=actual >= cap,
            actual_attempts=min(cap, actual),
            selected_trial_id=None,
            status="no_successful_trial",
            test_metrics={},
        )
        if selected is not None:
            entry.update(
                selected_trial_id=selected["trial_id"],
                parameters=selected["parameters"],
                validation_metrics=selected["validation_metrics"],
                selected_at_hpo_seconds=selected["hpo_elapsed_seconds"],
                status="pending",
            )
            if selected.get("inner_cv_metrics"):
                entry["inner_cv_metrics"] = selected["inner_cv_metrics"]
        payload["budgets"].append(entry)
    _save_budget_report(payload, path)
    for entry in payload["budgets"]:
        tid = entry["selected_trial_id"]
        if tid is None:
            continue
        if tid in cache and cache[tid].get("parameters") == entry["parameters"]:
            evaluation = cache[tid]
        else:
            if tid == -1 and context_default:
                selected = {**context_default["candidate"], "trial_id": -1}
            else:
                selected = next(r for r in trials if r["trial_id"] == tid)
            rep = copy.deepcopy(selected)
            rep.pop("error", None)
            predictions = []
            prediction_file = f"predictions_{prefix}budget_trial_{tid}.pt"
            baseline = context_default.get("baseline") if context_default else None
            baseline_file = f"predictions_{prefix}default.pt"
            best_file = (
                (best_rep or {}).get("predictions") or f"predictions_{prefix}best.pt"
            )
            if (
                tid == -1
                and test_ok(baseline, metric)
                and baseline.get("parameters") == selected["parameters"]
                and (run_dir / baseline_file).is_file()
            ):
                rep = copy.deepcopy(baseline)
                prediction_file = baseline_file
            elif (
                test_ok(best_rep, metric)
                and best_rep.get("trial_id") == tid
                and best_rep.get("parameters") == selected["parameters"]
                and (run_dir / best_file).is_file()
            ):
                rep = copy.deepcopy(best_rep)
                prediction_file = best_file
            else:
                _set_reproducibility(model_args.seed)
                apply_fixed_split_test_eval_for_best_trial(
                    selected_study(rep),
                    [rep],
                    dataset,
                    model_class,
                    model_args,
                    predictions,
                )
                if _primary_metric_value_ok(rep.get("test_metrics"), metric):
                    save_predictions(
                        run_dir, f"{prefix}budget_trial_{tid}", predictions
                    )
            ok = test_ok(rep, metric) and (run_dir / prediction_file).is_file()
            evaluation = dict(
                parameters=copy.deepcopy(selected["parameters"]),
                status="success" if ok else "failed",
                test_metrics=rep.get("test_metrics", {}),
                refit_seconds=rep.get("post_hpo_train_wall_time_seconds"),
                test_inference_seconds=rep.get("test_inference_wall_time_seconds"),
                train_val_merged_for_fit=rep.get("train_val_merged_for_fit"),
                predictions=prediction_file if ok else None,
                error=rep.get("post_hpo_test_error"),
            )
            cache[tid] = evaluation
        for key in (
            "status",
            "test_metrics",
            "refit_seconds",
            "test_inference_seconds",
            "train_val_merged_for_fit",
            "predictions",
            "error",
        ):
            if key in evaluation:
                entry[key] = evaluation[key]
        _save_budget_report(payload, path)
