import hashlib
import json
import numbers
import numpy as np
import pandas as pd

METADATA_FIELDS = (
    "task",
    "num_samples",
    "num_features",
    "cat_idx",
    "label",
    "num_classes",
    "imbalance_ratio",
    "split_strategy",
    "split_group",
    "curation_protocol",
)


def json_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def text_columns(array):
    if array.dtype.kind not in "OUS":
        return []
    matrix = array.reshape(len(array), -1)
    result = []
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        observed = values[~pd.isna(values)]
        if observed.size and isinstance(observed[0], (str, np.str_)):
            result.append(column)
        elif observed.size and (not isinstance(observed[0], numbers.Real)):
            raise ValueError(f"Unsupported feature value type in column {column}")
    return result


def array_digest(array, strings=()):
    array = np.asarray(array)
    if array.ndim not in (1, 2) or not array.shape[0]:
        raise ValueError("Expected a nonempty vector or matrix")
    matrix = array.reshape(len(array), -1)
    strings = tuple(strings)
    if len(set(strings)) != len(strings) or any(
        (c < 0 or c >= matrix.shape[1] for c in strings)
    ):
        raise ValueError("Invalid text-column indices")
    numeric = [c for c in range(matrix.shape[1]) if c not in set(strings)]
    numeric_hash, text_hash = (hashlib.sha256(), hashlib.sha256())
    step = max(1, 1000000 // matrix.shape[1])
    for start in range(0, len(matrix), step):
        block = matrix[start : start + step]
        if numeric:
            values = np.asarray(block[:, numeric], dtype="<f8").copy(order="C")
            values[np.isnan(values)] = np.nan
            values[values == 0] = 0
            numeric_hash.update(values.tobytes())
        if strings:
            values = block[:, strings].astype(object)
            values[pd.isna(values)] = None
            rows = values.tolist()
            if any(
                (
                    v is not None and (not isinstance(v, str))
                    for row in rows
                    for v in row
                )
            ):
                raise ValueError("A text column contains a non-string value")
            text_hash.update(
                json.dumps(
                    rows, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            )
    return json_digest(
        dict(
            shape=list(array.shape),
            text_columns=list(strings),
            numeric=numeric_hash.hexdigest(),
            text=text_hash.hexdigest(),
        )
    )


def split_payload(splits, n_samples, n_folds):
    folds = splits["cv_folds"] if n_folds else [splits]
    if len(folds) != (n_folds or 1):
        raise ValueError("Unexpected number of outer folds")
    normalized = []
    for fold in folds:
        converted = {}
        for part in ("train", "val", "test"):
            raw = np.asarray(fold[part])
            if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
                raise ValueError(f"Invalid {part} row indices")
            indices = raw.astype(np.int64)
            if (
                np.any(indices < 0)
                or np.any(indices >= n_samples)
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError(f"Duplicate or out-of-range {part} row indices")
            converted[part] = indices.tolist()
        combined = np.concatenate([converted[p] for p in ("train", "val", "test")])
        if len(combined) != n_samples or len(np.unique(combined)) != n_samples:
            raise ValueError("Each outer split must partition all original rows")
        normalized.append(converted)
    if n_folds:
        tests = np.concatenate([fold["test"] for fold in normalized])
        if len(tests) != n_samples or len(np.unique(tests)) != n_samples:
            raise ValueError("Outer tests must cover every original row exactly once")
    groups = splits.get("groups")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != n_samples:
            raise ValueError("Groups must contain one identifier per original row")
        groups = groups.tolist()
    return {"folds": normalized, "groups": groups}


def split_digest(splits, n_samples, n_folds):
    return json_digest(split_payload(splits, n_samples, n_folds))
