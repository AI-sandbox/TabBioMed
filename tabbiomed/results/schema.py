"""Public evaluation schema shared by result publishers."""

from copy import deepcopy
import math
from pathlib import Path
import re


def metric_direction(metric):
    return (
        "maximize"
        if metric.lower() in ("roc_auc", "auc", "accuracy", "f1", "r2")
        else "minimize"
    )


CONFIG_KEYS = (
    "n_trials",
    "trajectory_n_trials",
    "hpo_evaluation_budgets",
    "time_limit",
    "cv_time_limit_per_fold",
    "cv_outer_folds",
    "cv_inner_folds",
    "early_stopping_rounds",
    "data_parallel",
    "batch_size",
    "val_batch_size",
)

BLOCK_KEYS = (
    "parameters",
    "validation_metrics",
    "test_metrics",
    "trial_id",
    "fold_id",
    "train_wall_time_seconds",
    "val_inference_wall_time_seconds",
    "test_inference_wall_time_seconds",
    "post_hpo_train_wall_time_seconds",
    "fixed_split_test_evaluated",
    "train_val_merged_for_fit",
    "inner_cv_metrics",
)

INNER_KEYS = (
    "inner_fold_id",
    "train_samples",
    "validation_samples",
    "validation_metrics",
    "train_wall_time_seconds",
    "val_inference_wall_time_seconds",
)

UNIT_KEYS = (
    "actual_configurations",
    "trial_completion",
    "n_completed_trials",
    "n_successful_trials",
    "unsuccessful_trial_ids",
)


def checkpoint_reference(value):
    if isinstance(value, dict):
        return {k: checkpoint_reference(v) for k, v in value.items()}
    if isinstance(value, list):
        return [checkpoint_reference(v) for v in value]
    if isinstance(value, str):
        return Path(value).name
    return value


def clean(value):
    if isinstance(value, dict):
        return {
            k: clean(
                checkpoint_reference(v)
                if k in ("base_tabpfn_path", "checkpoint_path", "model_path")
                else v
            )
            for k, v in value.items()
            if k not in ("custom_model_dir", "disk_offload_dir")
        }
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and (
        value.startswith("/")
        or any(t in value for t in ("/private/", "/home/", "/tmp/", "/scratch/"))
        or re.search(r"run_\d{4}_\d{2}_\d{2}", value)
    ):
        raise ValueError("Nonpublic path or run identifier in export")
    return value


def pick(value, keys):
    return {k: deepcopy(value[k]) for k in keys if k in value}


def public_block(value):
    if value is None:
        return None
    out = pick(value, BLOCK_KEYS)
    if out.get("fold_id") is None:
        out.pop("fold_id", None)
    if value.get("inner_cv_metrics"):
        out["inner_cv_metrics"] = [
            pick(x, INNER_KEYS) for x in value["inner_cv_metrics"]
        ]
    for key in ("error", "post_hpo_test_error"):
        if value.get(key):
            text = str(value[key]).lower()
            out[key] = (
                "out_of_memory" if "out of memory" in text else "evaluation_failed"
            )
    return clean(out)


BOOKKEEPING = {
    "validation_seconds",
    "hpo_cutoff_seconds",
    "reach_best_trial_seconds",
    "hpo_finished",
    "actual_configurations",
    "trial_completion",
    "n_completed_trials",
    "unsuccessful_trial_ids",
}


def compact_failure(value):
    result = {
        k: deepcopy(value[k]) for k in ("reason", "memory_limit_gib") if k in value
    }
    return result


def remove_bookkeeping(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in BOOKKEEPING or key.startswith("elapsed_seconds"):
                continue
            if key == "failure" and isinstance(item, dict):
                item = compact_failure(item)
            result[key] = remove_bookkeeping(item)
        return result
    if isinstance(value, list):
        return [remove_bookkeeping(v) for v in value]
    return value


def normalize_report_schema(report):
    out = deepcopy(report)
    variants = ["default", "best"] if out["hpo_enabled"] else ["default"]
    units = out.get("cv_fold_metrics") or [out]
    cv = bool(out.get("cv_fold_metrics"))
    for unit in units:
        if out.get("hpo_enabled"):
            tuning = unit.setdefault("tuning", {})
            completion = unit.get("trial_completion") or {}
            for name, raw_name in (
                ("n_attempted_trials", "n_completed_trials"),
                ("n_successful_trials", "n_successful_trials"),
            ):
                value = unit.get(raw_name, completion.get(raw_name))
                if name == "n_attempted_trials" and value is None:
                    value = unit.get("actual_configurations")
                tuning.setdefault(name, value)
            attempted = tuning["n_attempted_trials"]
            successful = tuning["n_successful_trials"]
            if (
                attempted is not None
                and successful is not None
                and not 0 <= successful <= attempted
            ):
                raise ValueError(
                    "Successful sampled attempts exceed the recorded eligible attempts"
                )
        for key in UNIT_KEYS:
            unit.pop(key, None)
    if cv:
        for key in UNIT_KEYS:
            out.pop(key, None)

    def normalize_variants(value):
        for variant in ("default", "best"):
            value.setdefault(variant, None)

    for unit in units:
        normalize_variants(unit)
    for field in ("cv_aggregated_metrics", "cv_aggregated_timing"):
        if field in out:
            normalize_variants(out[field])
    if "completion_errors" not in out:
        errors = []
        for unit in units:
            prefix = f"Fold {unit['fold_id']}: " if cv else ""
            for variant in variants:
                metric = ((unit.get(variant) or {}).get("test_metrics") or {}).get(
                    out["primary_metric"]
                )
                if not isinstance(metric, (int, float)) or not math.isfinite(metric):
                    errors.append(
                        f"{prefix}{variant} has no successful test evaluation for {out['primary_metric']}"
                    )
        out["completion_errors"] = errors
    if not isinstance(out["completion_errors"], list):
        raise ValueError("completion_errors must be a list")
    return out


def summarize(report, selections=None):
    out = deepcopy(report)
    out.update(artifact_type="evaluation_summary", schema_version=1)
    units = out.get("cv_fold_metrics") or [out]
    cv = bool(out.get("cv_fold_metrics"))
    for unit in units:
        fold = unit["fold_id"] if cv else None
        recorded = (selections or {}).get(fold, unit.get("selection", {}))
        unit.pop("selection", None)
        if out.get("hpo_enabled"):
            config = out["config"]
            defaults = dict(
                metric=out["primary_metric"],
                direction=metric_direction(out["primary_metric"]),
                budget_seconds=recorded.get(
                    "selection_budget_seconds",
                    config["cv_time_limit_per_fold"] if cv else config["time_limit"],
                ),
                sampled_attempt_cap=recorded.get(
                    "attempted_configuration_cap", config["n_trials"]
                ),
                n_attempted_trials=recorded.get(
                    "retained_attempts",
                    unit.get(
                        "n_completed_trials",
                        unit.get(
                            "actual_configurations",
                            (unit.get("trial_completion") or {}).get(
                                "n_completed_trials"
                            ),
                        ),
                    ),
                ),
            )
            tuning = deepcopy(unit.get("tuning", {}))
            for key, value in defaults.items():
                tuning.setdefault(key, value)
            completion = unit.get("trial_completion") or {}
            tuning.setdefault(
                "n_successful_trials",
                unit.get("n_successful_trials", completion.get("n_successful_trials")),
            )
            attempts = tuning["n_attempted_trials"]
            successful = tuning["n_successful_trials"]
            if (
                attempts is not None
                and successful is not None
                and not 0 <= successful <= attempts
            ):
                raise ValueError(
                    "Successful sampled attempts exceed the recorded eligible attempts"
                )
            unit["tuning"] = tuning
    return clean(normalize_report_schema(remove_bookkeeping(out)))
