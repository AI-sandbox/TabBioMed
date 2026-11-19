import numpy as np

from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None, objective=None):
    """
    Compute metrics for a model

    Args:
        y_true: True labels
        y_pred: Predicted labels (class predictions)
        y_proba: Predicted probabilities (for classification tasks)
        objective: Task objective (classification, binary, or regression)

    Returns:
        Dictionary of metrics
    """

    metrics = {}

    if objective in ["classification", "binary"]:
        metrics.update(
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    average="binary" if objective == "binary" else "weighted",
                ),
            }
        )

        if y_proba is not None:
            try:
                n_classes = y_proba.shape[1]
                labels = np.arange(n_classes)
                if objective == "binary":
                    if n_classes == 2:
                        auc = roc_auc_score(
                            y_true, y_proba[:, 1], labels=labels
                        )
                    else:
                        auc = roc_auc_score(
                            y_true, y_proba[:, 0], labels=labels
                        )
                else:
                    auc = roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=labels,
                    )
                metrics["roc_auc"] = float(auc) if np.isfinite(auc) else None
            except Exception as e:
                logger.warning(f"Could not compute roc_auc: {e}")

        if y_proba is not None:
            try:
                labels = np.arange(y_proba.shape[1])
                metrics["log_loss"] = log_loss(y_true, y_proba, labels=labels)
            except Exception as e:
                logger.warning(f"Could not compute log_loss: {e}")

    else:
        metrics.update(
            {
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2": r2_score(y_true, y_pred),
            }
        )

    return metrics
