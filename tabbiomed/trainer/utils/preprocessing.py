from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer


@dataclass(frozen=True)
class ScalingPlan:
    cat_indices: set[int]
    num_indices: list[int]
    scale_indices_in_num: list[int]
    scale_indices: list[int]
    binary_indices_in_num: list[int]
    binary_indices: list[int]


def _detect_binary_columns(num_cols_train: np.ndarray) -> list[int]:
    """Return indices (in the numeric-submatrix) that look binary (0/1)."""
    binary: list[int] = []
    for col_idx in range(num_cols_train.shape[1]):
        try:
            col = num_cols_train[:, col_idx]
            uniq = np.unique(col[~np.isnan(col)])
            if len(uniq) <= 2:
                is_binary = all(
                    np.isclose(v, 0, atol=1e-6) or np.isclose(v, 1, atol=1e-6)
                    for v in uniq
                )
                if is_binary:
                    binary.append(col_idx)
        except (TypeError, ValueError):
            continue
    return binary


def build_scaling_plan(
    X_train: np.ndarray,
    cat_idx: Iterable[int] | None,
) -> ScalingPlan:
    cat_indices = set(int(i) for i in (cat_idx or []))
    num_indices = [i for i in range(X_train.shape[1]) if i not in cat_indices]
    if not num_indices:
        return ScalingPlan(
            cat_indices=cat_indices,
            num_indices=[],
            scale_indices_in_num=[],
            scale_indices=[],
            binary_indices_in_num=[],
            binary_indices=[],
        )

    num_cols_train = X_train[:, num_indices].astype(float)
    binary_in_num = _detect_binary_columns(num_cols_train)
    scale_in_num = [i for i in range(num_cols_train.shape[1]) if i not in binary_in_num]
    scale_indices = [num_indices[i] for i in scale_in_num]
    binary_indices = [num_indices[i] for i in binary_in_num]
    return ScalingPlan(
        cat_indices=cat_indices,
        num_indices=num_indices,
        scale_indices_in_num=scale_in_num,
        scale_indices=scale_indices,
        binary_indices_in_num=binary_in_num,
        binary_indices=binary_indices,
    )


def scale_splits_fit_on_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    *,
    cat_idx: Iterable[int] | None,
    scaler_type: str = "standard",
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object | None, ScalingPlan]:
    """
    Fit a scaler on training rows only and transform train/val/test.
    Categorical columns (cat_idx) and detected binary numeric columns are excluded from scaling.
    """
    if scaler_type not in ("standard", "quantile"):
        raise ValueError(f"Unsupported scaler: {scaler_type!r}")
    plan = build_scaling_plan(X_train, cat_idx)
    if not plan.scale_indices:
        return X_train, X_val, X_test, None, plan

    num_cols_train = X_train[:, plan.num_indices].astype(float)
    cols_to_scale_train = num_cols_train[:, plan.scale_indices_in_num]

    if scaler_type == "quantile":
        scaler = QuantileTransformer(
            output_distribution="uniform", random_state=random_state
        )
    else:
        scaler = StandardScaler()

    scaler.fit(cols_to_scale_train)

    scaled_splits = []
    for split in (X_train, X_val, X_test):
        dtype = split.dtype
        if dtype.kind in "biu":
            dtype = float
        elif dtype.kind in "SU":
            dtype = object
        scaled = split.astype(dtype, copy=True)
        if split.shape[0] > 0:
            scaled[:, plan.scale_indices] = scaler.transform(
                split[:, plan.scale_indices].astype(float)
            )
        scaled_splits.append(scaled)

    return (*scaled_splits, scaler, plan)
