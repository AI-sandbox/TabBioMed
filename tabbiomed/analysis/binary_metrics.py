from __future__ import annotations
from tabbiomed.analysis.datasets import load_metadata
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Callable
import numpy as np

ECE_N_BINS = 15
METRICS = (
    {
        "key": "roc_auc",
        "label": "ROC-AUC",
        "higher_is_better": True,
        "from_report": True,
    },
    {
        "key": "pr_auc",
        "label": "PR-AUC",
        "higher_is_better": True,
        "from_report": False,
    },
    {
        "key": "log_loss",
        "label": "log loss",
        "higher_is_better": False,
        "from_report": True,
    },
    {"key": "brier", "label": "Brier", "higher_is_better": False, "from_report": False},
    {"key": "ece", "label": "ECE", "higher_is_better": False, "from_report": False},
)
ALT_METRICS = tuple((spec for spec in METRICS if spec["key"] != "roc_auc"))
_FOLD_FILE = re.compile("predictions_fold_(\\d+)_(best|default)\\.pt")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def positive_scores(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 1:
        raise ValueError(f"Invalid binary probability shape: {probabilities.shape}")
    return probabilities[:, 1 if probabilities.shape[1] >= 2 else 0].astype(float)


def pr_auc_binary(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(
        average_precision_score(
            np.asarray(labels).ravel(), positive_scores(probabilities)
        )
    )


def brier_binary(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss

    return float(
        brier_score_loss(np.asarray(labels).ravel(), positive_scores(probabilities))
    )


def ece_binary(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int = ECE_N_BINS
) -> float:
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    labels = np.asarray(labels, dtype=float).ravel()
    scores = np.clip(positive_scores(probabilities), 0.0, 1.0)
    if not len(labels) or len(labels) != len(scores):
        raise ValueError("Labels and probabilities must have the same nonzero length")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise ValueError("Labels and probabilities must be finite")
    from .calibration import ece

    return ece(labels, scores, n_bins=n_bins, strategy="uniform")


def _test_metric(block: Any, metric: str) -> float | None:
    if (
        not isinstance(block, dict)
        or block.get("fixed_split_test_evaluated") is not True
    ):
        return None
    if any((block.get(key) for key in ("error", "post_hpo_test_error", "failure"))):
        return None
    values = block.get("test_metrics") or {}
    if not isinstance(values, dict):
        return None
    for name in (metric, "auc") if metric == "roc_auc" else (metric,):
        value = values.get(name)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and finite(value)
        ):
            return float(value)
    return None


def _test_units(report: dict[str, Any], expected_folds: int) -> list[dict[str, Any]]:
    if not expected_folds:
        return [report]
    folds = report.get("cv_fold_metrics") or []
    if not isinstance(folds, list):
        raise ValueError("cv_fold_metrics must be a list")
    indexed = {}
    for fold in folds:
        fold_id = fold.get("fold_id") if isinstance(fold, dict) else None
        if (
            not isinstance(fold_id, int)
            or isinstance(fold_id, bool)
            or fold_id not in range(expected_folds)
            or (fold_id in indexed)
        ):
            raise ValueError("Invalid or duplicate outer test fold")
        indexed[fold_id] = fold
    return [indexed.get(index, {}) for index in range(expected_folds)]


def _recorded_score(
    report: dict[str, Any] | None, variant: str, metric: str, expected_folds: int
) -> tuple[float | None, float | None, int, str]:
    if report is None:
        return (None, None, 0, "missing_report")
    if variant not in report.get("evaluated_variants", ["best", "default"]):
        return (None, None, 0, "variant_not_evaluated")
    units = _test_units(report, expected_folds)
    values = [_test_metric(unit.get(variant), metric) for unit in units]
    valid = [value for value in values if value is not None]
    if len(valid) != len(units):
        return (
            None,
            None,
            len(valid),
            f"recorded_test_folds_{len(valid)}_of_{len(units)}",
        )
    return (
        float(np.mean(valid)),
        float(np.std(valid)) if len(valid) > 1 else None,
        len(valid),
        "ok",
    )


def _expected_folds(report: dict[str, Any] | None, info: dict[str, Any] | None) -> int:
    cv = report is not None and any(
        (key in report for key in ("cv_fold_metrics", "cv_aggregated_metrics"))
    )
    expected = info.get("n_folds") if info is not None else None
    if expected is None and cv:
        expected = report.get("n_outer_folds") or (report.get("config") or {}).get(
            "cv_outer_folds"
        )
        if expected is None:
            raise ValueError(
                "CV evaluation requires an explicit expected outer-fold count"
            )
    if expected is None:
        expected = 0
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        raise ValueError("Expected fold count must be a nonnegative integer")
    if cv and (not expected):
        raise ValueError("CV report cannot be evaluated as a fixed split")
    return expected


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_prediction(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, (tuple, list)):
        payload = next(
            (
                entry
                for entry in reversed(payload)
                if isinstance(entry, dict) and entry.get("y_proba") is not None
            ),
            None,
        )
    if not isinstance(payload, dict) or payload.get("y_proba") is None:
        return None
    return _as_numpy(payload["y_proba"])


def prediction_pairs(
    directory: Path, variant: str, *, expected_folds: int | None = None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if variant not in ("best", "default"):
        raise ValueError(variant)
    files = sorted(
        (
            (int(match[1]), path)
            for path in directory.glob("predictions_fold_*.pt")
            if (match := _FOLD_FILE.fullmatch(path.name)) and match[2] == variant
        )
    )
    if files:
        if expected_folds == 0:
            raise ValueError(f"CV predictions in fixed-split evaluation: {directory}")
        if expected_folds is not None and any((k >= expected_folds for k, _ in files)):
            raise ValueError(f"Unexpected outer-fold prediction: {directory}")
        units = [
            (f"fold_{fold}", path, directory / f"test_labels_fold_{fold}.npy")
            for fold, path in files
        ]
    elif expected_folds:
        return ([], "missing_fold_predictions")
    else:
        units = [
            (
                "fixed",
                directory / f"predictions_{variant}.pt",
                directory / "test_labels_fixed.npy",
            )
        ]
    pairs = []
    notes = []
    for name, predictions, labels_path in units:
        try:
            proba = load_prediction(predictions)
            if proba is None:
                raise ValueError("missing_proba")
            if not labels_path.is_file():
                raise ValueError("missing_labels")
            labels = np.asarray(np.load(labels_path))
            if labels.ndim == 2 and labels.shape[1] == 1:
                labels = labels[:, 0]
            if labels.ndim != 1 or not len(labels) or len(labels) != len(proba):
                raise ValueError("labels_probability_shape_mismatch")
            scores = positive_scores(proba)
            if not np.isfinite(proba).all() or not np.isfinite(labels).all():
                raise ValueError("nonfinite_labels_or_probabilities")
            if (
                not np.isin(labels, [0, 1]).all()
                or (scores < 0).any()
                or (scores > 1).any()
            ):
                raise ValueError("invalid_binary_labels_or_probabilities")
            pairs.append((labels, proba))
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            EOFError,
            pickle.UnpicklingError,
        ) as exc:
            notes.append(f"{name}:{exc}")
    if expected_folds and len(pairs) != expected_folds:
        notes.append(f"available_folds_{len(pairs)}_of_{expected_folds}")
    return (pairs, ",".join(notes) or ("ok" if pairs else "no_predictions"))


def score_folds(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    calculator: Callable,
    *,
    expected_count: int,
) -> tuple[float | None, float | None, int, str]:
    values = []
    notes = []
    for index, (labels, probabilities) in enumerate(pairs):
        try:
            value = float(calculator(labels, probabilities))
            if not finite(value):
                raise ValueError("nonfinite_metric")
            values.append(value)
        except (ValueError, TypeError, RuntimeError) as exc:
            notes.append(f"fold_{index}:{exc}")
    if len(values) != expected_count:
        notes.append(f"scored_test_folds_{len(values)}_of_{expected_count}")
    if notes or not values:
        return (None, None, len(values), ",".join(notes) or "no_scores")
    return (
        float(np.mean(values)),
        float(np.std(values)) if len(values) > 1 else None,
        len(values),
        "ok",
    )


def score_pair(
    model: str,
    dataset: str,
    results_dir: Path,
    processed_path: Path,
    *,
    n_bins: int = ECE_N_BINS,
    info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    directory = Path(results_dir) / model / dataset
    metadata = load_metadata(dataset, processed_path)
    if metadata.get("task") != "binary" or (
        info is not None and info.get("task") != "binary"
    ):
        raise ValueError(f"Expected binary task: {dataset}")
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    if report is not None and (
        report.get("artifact_type") != "evaluation_summary"
        or report.get("schema_version") != 1
        or str(report.get("model", "")).lower() != model.lower()
        or (report.get("dataset") != dataset)
    ):
        raise ValueError(f"Invalid consolidated Main report: {report_path}")
    expected = _expected_folds(report, info)
    count = expected or 1
    row: dict[str, Any] = dict(
        model=model,
        dataset=dataset,
        imbalance_ratio=(
            float(metadata["imbalance_ratio"])
            if finite(metadata.get("imbalance_ratio"))
            else None
        ),
        split_strategy=metadata.get("split_strategy") or "",
        ece_n_bins=n_bins,
        expected_test_folds=count,
    )
    notes = []
    for key in ("roc_auc", "log_loss"):
        for variant in ("best", "default"):
            mean, std, actual, note = _recorded_score(report, variant, key, expected)
            row.update(
                {
                    f"{key}_{variant}": mean,
                    f"{key}_{variant}_std": std,
                    f"{key}_n_folds_{variant}": actual,
                }
            )
            if note != "ok":
                notes.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        metric=key,
                        variant=variant,
                        note=note,
                    )
                )
    pairs = {
        variant: (
            prediction_pairs(directory, variant, expected_folds=expected)
            if report is not None
            else ([], "missing_report")
        )
        for variant in ("best", "default")
    }
    calculators = {
        "pr_auc": pr_auc_binary,
        "brier": brier_binary,
        "ece": lambda y, p: ece_binary(y, p, n_bins),
    }
    for key, calculator in calculators.items():
        for variant, (predictions, prediction_note) in pairs.items():
            mean, std, actual, metric_note = score_folds(
                predictions, calculator, expected_count=count
            )
            note = ",".join((n for n in (prediction_note, metric_note) if n != "ok"))
            if prediction_note != "ok":
                mean, std = (None, None)
            row.update(
                {
                    f"{key}_{variant}": mean,
                    f"{key}_{variant}_std": std,
                    f"{key}_n_folds_{variant}": actual,
                }
            )
            if note:
                notes.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        metric=key,
                        variant=variant,
                        note=note,
                    )
                )
    for variant in ("best", "default"):
        row[f"{variant}_complete"] = all(
            (finite(row[f"{spec['key']}_{variant}"]) for spec in METRICS)
        )
    source = (
        "best"
        if report is not None and report["hpo_enabled"] and row["best_complete"]
        else "default" if row["default_complete"] else "missing"
    )
    row["selected_variant"] = source
    for spec in METRICS:
        key = spec["key"]
        row.update(
            {
                key: row[f"{key}_{source}"] if source != "missing" else None,
                f"{key}_source": source,
                f"{key}_note": "" if source != "missing" else "no_complete_variant",
            }
        )
    return (row, notes)
