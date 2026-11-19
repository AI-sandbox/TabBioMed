"""Validation of saved test predictions before publication."""

import math

import numpy as np
import torch

from ..evaluator.metrics import compute_metrics


def numeric_array(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return array


def validate_predictions(path, labels_path, metadata, test_metrics, primary_metric):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot read prediction file: {path}") from exc
    if not isinstance(payload, dict) or payload.get("y_pred") is None:
        raise ValueError(f"Prediction file must contain y_pred: {path}")
    labels = numeric_array(np.load(labels_path, allow_pickle=False), "Test labels")
    predictions = numeric_array(payload["y_pred"], "Predictions")
    if labels.ndim not in (1, 2) or not labels.size:
        raise ValueError("Test labels must have a nonempty sample/output shape")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} differs from test labels "
            f"{labels.shape}: {path}"
        )
    task = metadata["task"]
    probabilities = None
    if task in ("binary", "classification"):
        classes = metadata["num_classes"]
        if (
            type(classes) is not int
            or classes < 2
            or (task == "binary" and classes != 2)
            or (task == "classification" and classes <= 2)
        ):
            raise ValueError("Classification metadata has an invalid class count")
        if labels.ndim != 1 or any(
            np.any((array < 0) | (array >= classes) | (array != np.floor(array)))
            for array in (labels, predictions)
        ):
            raise ValueError(
                "Class labels must be integers within the dataset class range"
            )
        if payload.get("y_proba") is None:
            raise ValueError(f"Classification predictions must contain y_proba: {path}")
        probabilities = numeric_array(payload["y_proba"], "Class probabilities")
        if probabilities.shape != (len(labels), classes):
            raise ValueError(
                "Probability shape must match test rows and dataset classes"
            )
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Class probabilities must lie between zero and one")
        tolerance = (
            max(1e-5, 2 * np.finfo(probabilities.dtype).eps)
            if probabilities.dtype.kind == "f"
            else 1e-5
        )
        if not np.allclose(probabilities.sum(axis=1), 1, rtol=tolerance, atol=1e-7):
            raise ValueError("Class probability rows must sum to one")
    elif task in ("regression", "multiregression"):
        if payload.get("y_proba") is not None:
            raise ValueError(
                "Regression predictions must not contain class probabilities"
            )
    else:
        raise ValueError(f"Unknown prediction task: {task}")
    recorded = test_metrics.get(primary_metric)
    if (
        not isinstance(recorded, (int, float))
        or isinstance(recorded, bool)
        or not math.isfinite(recorded)
    ):
        raise ValueError("Saved primary test metric must be finite")
    metric = primary_metric.lower()
    if metric == "auc":
        metric = "roc_auc"
    try:
        computed = compute_metrics(labels, predictions, probabilities, objective=task)[
            metric
        ]
    except Exception as exc:
        raise ValueError(
            f"Cannot recompute primary test metric {primary_metric}: {path}"
        ) from exc
    if not math.isfinite(computed) or not math.isclose(
        computed, recorded, rel_tol=1e-6, abs_tol=1e-7
    ):
        raise ValueError(
            f"Saved primary test metric disagrees with predictions: {path}"
        )
