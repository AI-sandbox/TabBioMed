from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from tabbiomed.evaluator.metrics import compute_metrics


@dataclass(frozen=True)
class SplitEvalResult:
    metrics: dict
    inference_wall_time_seconds: float
    y_pred: Any
    y_proba: Any


def evaluate_split(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    objective: str,
    compute_proba: bool,
) -> SplitEvalResult:
    """
    Evaluate a fitted model on one split (val or test).
    """
    t0 = time.perf_counter()

    if compute_proba and hasattr(model, "prediction_probabilities"):
        model.prediction_probabilities = None
    y_pred = model.predict(X)
    y_proba = None
    if compute_proba:
        y_proba = getattr(model, "prediction_probabilities", None)
        if y_proba is None:
            y_proba = model.predict_proba(X)
        if y_proba is None:
            raise ValueError(
                "Probability prediction returned None during split evaluation"
            )
    inf = time.perf_counter() - t0
    metrics = compute_metrics(y, y_pred, y_proba, objective)
    return SplitEvalResult(
        metrics=metrics, inference_wall_time_seconds=inf, y_pred=y_pred, y_proba=y_proba
    )
