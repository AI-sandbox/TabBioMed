"""Portable search histories and validation-selected budget evaluations."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from ..budgets import completed_within_deadline
from .io import digest, local_file, write_json
from .predictions import validate_predictions
from .schema import clean, pick, INNER_KEYS, metric_direction
from .selection import inferred_failure, test_available
from .sources import Evidence
from .timing import (
    canonical,
    load_history,
    read_selection,
    seconds,
    trial_records,
    units,
)

CAPS = (25, 50, 100, 200)


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def parameter_key(parameters):
    return json.dumps(canonical(parameters), sort_keys=True, separators=(",", ":"))


def select_prefix(trials, candidate, metric, cap, cutoff):
    direction = 1 if metric_direction(metric) == "maximize" else -1
    eligible = [
        t
        for t in trials
        if t["trial_id"] < cap
        and t.get("status") == "success"
        and finite(t.get("hpo_elapsed_seconds"))
        and t["hpo_elapsed_seconds"] <= cutoff
        and finite((t.get("validation_metrics") or {}).get(metric))
    ]
    best = min(
        eligible,
        key=lambda t: (-direction * t["validation_metrics"][metric], t["trial_id"]),
        default=None,
    )
    value = (candidate.get("validation_metrics") or {}).get(metric)
    source_finished = (
        candidate.get("source_completion_seconds")
        if "source_trial_id" in candidate
        else 0.0
    )
    if (
        candidate.get("status") == "success"
        and completed_within_deadline(source_finished, cutoff)
        and finite(value)
    ):
        if (
            best is None
            or direction * value >= direction * best["validation_metrics"][metric]
        ):
            return dict(candidate, trial_id=-1)
    return best


def valid_test(block, metric):
    return test_available(block, metric)


def normalized_trial(record, fold, *, default=False):
    out = dict(
        fold_id=fold,
        trial_id=-1 if default else record["trial_id"],
        parameters=canonical(record.get("parameters")),
        status="interrupted"
        if record.get("status") in (None, "running")
        else record["status"],
        validation_metrics=canonical(record.get("validation_metrics") or {}),
    )
    fields = {
        "trial_wall_time_seconds": "duration_seconds",
        "train_wall_time_seconds": "fit_seconds",
        "val_inference_wall_time_seconds": "validation_prediction_seconds",
    }
    if default:
        out["duration_seconds"] = seconds(record.get("validation_seconds"))
        if "source_trial_id" in record:
            out["finished_seconds"] = seconds(record.get("source_completion_seconds"))
    else:
        out.update(
            started_seconds=seconds(record.get("started_at_hpo_seconds")),
            finished_seconds=seconds(record.get("hpo_elapsed_seconds"))
            if record.get("event") != "trial_started"
            else None,
        )
    for raw, public in fields.items():
        if raw in record:
            out[public] = seconds(record[raw])
    if record.get("inner_cv_metrics"):
        out["inner_cv_metrics"] = [
            pick(v, INNER_KEYS) for v in record["inner_cv_metrics"]
        ]
    if out["status"] not in ("success", "running"):
        out["failure"] = (
            {"reason": "interrupted"}
            if out["status"] == "interrupted" and not record.get("error")
            else inferred_failure(record)
        )
    return clean(out)


@dataclass
class HistoryExport:
    trials: list
    selections: dict
    artifacts: dict
    evidence: Evidence


def prepare_history(
    public_path,
    source_path,
    *,
    split_sha256,
    metadata,
    search_ids=None,
    expected_source_sha256=None,
    expected_report_sha256=None,
):
    public_path, source_path = Path(public_path).resolve(), Path(source_path).resolve()
    evidence = Evidence()
    public, source = evidence.read(public_path), evidence.read(source_path)
    evidence.pin(source_path, expected_source_sha256)
    evidence.pin(public_path, expected_report_sha256)
    timing_path = public_path.with_name("timing.json")
    timing = evidence.read(timing_path)
    report_sha = evidence.files[public_path]
    if timing["report_sha256"] != report_sha:
        raise ValueError("Timing belongs to another Main report")
    for key in ("dataset", "seed"):
        if public.get(key) != source.get(key):
            raise ValueError(f"Search and Main differ on {key}")
    if (
        source.get("primary_metric") is not None
        and source["primary_metric"] != public["primary_metric"]
    ):
        raise ValueError("Search and Main use different primary metrics")
    if str(public["model"]).lower() != str(source["model"]).lower():
        raise ValueError("Search and Main use different models")
    if timing["dataset_split_sha256"] != split_sha256:
        raise ValueError("Search and Main use different stored splits")
    if not public.get("hpo_enabled"):
        raise ValueError("Default-only evaluations have no HPO history")
    run = source_path.parent
    history_path = run / "hpo_trials.jsonl"
    if evidence.optional(history_path):
        events, history_sha, _ = load_history(history_path)
        evidence.pin(history_path, history_sha)
    else:
        events = []
    for event in events:
        for key in ("dataset", "seed", "primary_metric"):
            if key in event and event[key] != public[key]:
                raise ValueError(f"Trial history differs from Main on {key}")
    pu, su = units(public), units(source)
    if set(pu) != set(su) or {u["fold_id"] for u in timing["units"]} != set(pu):
        raise ValueError("Search, timing and Main have different outer folds")
    metric = public["primary_metric"]
    rows, exported, artifacts = [], [], {}
    known_predictions = {}
    for filename in sorted(public_path.parent.glob("predictions_*.pt")):
        known_predictions.setdefault(evidence.pin(filename), "../" + filename.name)

    def prediction_reference(block, directory, fold, trial_id, alternatives):
        candidates = [(block, directory), *alternatives]
        for value, base in candidates:
            pointer = value.get("predictions")
            if not pointer:
                continue
            if Path(pointer).is_absolute():
                pointer = str(Path(pointer).resolve().relative_to(base.resolve()))
            path = base / pointer
            if not evidence.optional(path):
                continue
            path = local_file(base, pointer)
            sha = evidence.pin(path)
            label = (
                f"test_labels_{'fixed' if fold is None else 'fold_' + str(fold)}.npy"
            )
            public_labels = public_path.parent / label
            if not public_labels.is_file():
                raise ValueError("Saved budget predictions lack Main test labels")
            label_sha = evidence.pin(public_labels)
            raw_labels = run / label
            if evidence.optional(raw_labels) and evidence.pin(raw_labels) != label_sha:
                raise ValueError("Budget and Main test labels differ")
            validate_predictions(
                path, public_labels, metadata, block["test_metrics"], metric
            )
            if sha not in known_predictions:
                prefix = "fixed" if fold is None else f"fold_{fold}"
                suffix = "default" if trial_id == -1 else f"trial_{trial_id}"
                name = f"predictions/{prefix}_{suffix}.pt"
                if name in artifacts and digest(artifacts[name]) != sha:
                    raise ValueError(
                        "One selected configuration has conflicting predictions"
                    )
                artifacts[name] = path
                known_predictions[sha] = name
            return known_predictions[sha]
        return None

    for clock in timing["units"]:
        fold = clock["fold_id"]
        public_unit, source_unit = pu[fold], su[fold]
        hpo = clock["hpo"]
        candidate, selection, selected_files = read_selection(run, fold)
        for path, sha in selected_files.items():
            evidence.pin(path, sha)
        trials, _, sid = trial_records(
            events,
            fold,
            (search_ids or {}).get(str(fold)) or selection.get("search_id"),
        )
        if candidate:
            rows.append(normalized_trial(candidate, fold, default=True))
        rows.extend(normalized_trial(t, fold) for t in trials)
        cutoff = hpo["search_cutoff_seconds"]
        if (candidate.get("validation_seconds") or 0) > (
            hpo.get("budget_seconds") or 0
        ):
            candidate = dict(candidate, status="timeout")
        budget_path = run / (
            "hpo_budget_results.json"
            if fold is None
            else f"hpo_budget_results_fold_{fold}.json"
        )
        budget = evidence.read(budget_path) if evidence.optional(budget_path) else {}
        if budget and budget.get("search_id") != sid:
            raise ValueError("Budget evaluations belong to a different search")
        pool = []
        for block, base, variant in [
            (source_unit.get("default"), run, "default"),
            (source_unit.get("best"), run, "best"),
            (public_unit.get("best"), public_path.parent, "best"),
            (selection.get("evaluation"), run, None),
        ]:
            if valid_test(block, metric):
                if variant is not None and not block.get("predictions"):
                    prefix = "" if fold is None else f"fold_{fold}_"
                    block = dict(block, predictions=f"predictions_{prefix}{variant}.pt")
                pool.append(
                    (
                        block.get("trial_id", -1),
                        parameter_key(block.get("parameters")),
                        block,
                        base,
                    )
                )
        for entry in budget.get("budgets", []):
            if entry.get("status") == "success" and finite(
                (entry.get("test_metrics") or {}).get(metric)
            ):
                pool.append(
                    (
                        entry["selected_trial_id"],
                        parameter_key(entry.get("parameters")),
                        entry,
                        run,
                    )
                )
        def _trial_start_seconds(t):
            started = t.get("started_at_hpo_seconds")
            if started is not None:
                return started
            elapsed = t.get("hpo_elapsed_seconds")
            return elapsed if elapsed is not None else math.inf

        attempted = (
            [t for t in trials if _trial_start_seconds(t) < cutoff]
            if finite(cutoff)
            else None
        )
        if finite(cutoff) and valid_test(public_unit.get("best"), metric):
            chosen = select_prefix(
                trials, candidate, metric, hpo["official_attempt_cap"], cutoff
            )
            best = public_unit["best"]
            if chosen is None or parameter_key(
                chosen.get("parameters")
            ) != parameter_key(best.get("parameters")):
                raise ValueError("Recorded official selection differs from Main Tuned")
        entries = []
        for cap in CAPS:
            item = dict(
                attempt_cap=cap,
                actual_attempts=min(cap, len(attempted))
                if attempted is not None
                else None,
                cap_reached=bool(attempted is not None and len(attempted) >= cap),
                selected_trial_id=None,
                status="no_eligible_configuration",
                validation_metrics={},
                test_metrics={},
                predictions=None,
            )
            if not finite(cutoff):
                item["status"] = "selection_clock_unavailable"
                entries.append(item)
                continue
            selected = select_prefix(trials, candidate, metric, cap, cutoff)
            if selected is not None:
                trial_id, key = (
                    selected["trial_id"],
                    parameter_key(selected.get("parameters")),
                )
                item.update(
                    selected_trial_id=trial_id,
                    status="selected_test_unavailable",
                    validation_metrics=canonical(selected["validation_metrics"]),
                )
                matches = [
                    (block, base)
                    for tid, params, block, base in pool
                    if tid == trial_id and params == key
                ]
                if matches:
                    block, base = matches[0]
                    compatible = [
                        (b, d)
                        for b, d in matches[1:]
                        if canonical(b["test_metrics"])
                        == canonical(block["test_metrics"])
                    ]
                    pred_ref = prediction_reference(
                        block, base, fold, trial_id, compatible
                    )
                    if pred_ref is None:
                        raise ValueError(
                            f"Saved budget evaluation for cap {cap} is missing its prediction file"
                        )
                    item.update(
                        test_metrics=canonical(block["test_metrics"]),
                        status="success",
                        predictions=pred_ref,
                    )
                    for raw, name in [
                        ("refit_seconds", "refit_seconds"),
                        ("post_hpo_train_wall_time_seconds", "refit_seconds"),
                        ("test_inference_seconds", "test_prediction_seconds"),
                        ("test_inference_wall_time_seconds", "test_prediction_seconds"),
                    ]:
                        if raw in block:
                            item[name] = seconds(block[raw])
                    if "train_val_merged_for_fit" in block:
                        item["train_val_merged_for_fit"] = block[
                            "train_val_merged_for_fit"
                        ]
                if cap == hpo["official_attempt_cap"] and valid_test(
                    public_unit.get("best"), metric
                ):
                    best = public_unit["best"]
                    if parameter_key(best.get("parameters")) != key:
                        raise ValueError(
                            f"Official validation selection differs from Main: {public['model']}/{public['dataset']}/{fold}"
                        )
                    if item["test_metrics"] and not math.isclose(
                        item["test_metrics"][metric],
                        best["test_metrics"][metric],
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            "Official selected test metric differs from Main"
                        )
                if not candidate:
                    item["test_metrics"] = {}
                    item["predictions"] = None
                    item["status"] = "default_validation_unavailable"
            entries.append(item)
        exported.append(dict(fold_id=fold, budgets=entries))
    selections = clean(
        dict(
            schema_version=1,
            model=public["model"],
            dataset=public["dataset"],
            primary_metric=metric,
            report_sha256=report_sha,
            timing_sha256=evidence.files[timing_path],
            units=exported,
        )
    )
    return HistoryExport(rows, selections, artifacts, evidence)


def stage_history(export, destination):
    from .io import copy_verified

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "trials.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in export.trials
        )
    )
    write_json(destination / "selections.json", export.selections)
    for name, source in export.artifacts.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        copy_verified(source, path)
