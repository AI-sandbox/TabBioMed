import json
from pathlib import Path
import shutil
import tempfile

from loguru import logger
import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
from sklearn.preprocessing import LabelEncoder

from . import verify_dataset
from .catalog import dataset_info


def _check_data(X, y, task) -> None:
    assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
    assert isinstance(y, np.ndarray), "y must be an instance of np.ndarray"
    assert X.shape[0] == y.shape[0], "X and y must match along their 0-th dimensions"
    assert len(X.shape) == 2, "X must be 2-dimensional"
    if task in ("regression", "binary", "classification"):
        assert len(y.shape) == 1, (
            "y must be 1-dimensional for regression and classification tasks"
        )
    elif task == "multiregression":
        assert len(y.shape) == 2 and y.shape[1] >= 2, (
            "y must be 2-dimensional with at least 2 columns for multiregression tasks"
        )


def _drop_constant_and_empty_columns(
    X: np.ndarray, dataset_name: str = "", fit_rows: np.ndarray = None
) -> tuple:
    X_fit = X if fit_rows is None else X[fit_rows]
    n_rows, n_cols = X_fit.shape
    keep_cols = []
    for j in range(n_cols):
        col = X_fit[:, j]
        if np.issubdtype(col.dtype, np.number):
            valid = ~np.isnan(col.astype(float))
        else:
            valid = np.ones(n_rows, dtype=bool)
            for i in range(n_rows):
                v = col.flat[i]
                if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
                    valid[i] = False
        n_valid = np.sum(valid)
        if n_valid == 0:
            continue
        unique_vals = np.unique(col[valid])
        if len(unique_vals) <= 1:
            continue
        keep_cols.append(j)
    if len(keep_cols) < n_cols:
        n_dropped = n_cols - len(keep_cols)
        rows_desc = "training rows" if fit_rows is not None else "all rows"
        logger.info(
            f"{dataset_name}: Dropped {n_dropped} constant or all-empty column(s) on {rows_desc}. Remaining features: {len(keep_cols)}."
        )
        X = X[:, np.array(keep_cols)]
    return (X, keep_cols)


def _normalize_labels(y: np.ndarray, dataset_name: str = "") -> np.ndarray:
    y_flat = np.asarray(y).ravel()
    unique = np.unique(y_flat)
    n_classes = len(unique)
    expected = np.arange(n_classes)
    if n_classes == 0:
        return y_flat
    if np.issubdtype(unique.dtype, np.number) and np.array_equal(
        np.sort(unique), expected
    ):
        return y_flat.astype(np.int64)
    le = LabelEncoder()
    y_normalized = le.fit_transform(y_flat).astype(np.int64)
    name_str = f" ({dataset_name})" if dataset_name else ""
    logger.info(
        f"Labels normalized{name_str}: original values {unique.tolist()} -> 0..{n_classes - 1}. Use 0/1 for binary, 0..n-1 for multiclass."
    )
    return y_normalized


def _generate_nested_cv_folds_split_indices(
    *,
    n_samples: int,
    y: np.ndarray,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    indices = np.arange(int(n_samples), dtype=int)
    y_arr = np.asarray(y).ravel()
    if task in ("classification", "binary"):
        outer = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        outer_iter = outer.split(indices, y_arr)
    else:
        outer = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        outer_iter = outer.split(indices)
    cv_folds: list[dict] = []
    for fold_i, (pool_rel, test_rel) in enumerate(outer_iter):
        pool_idx = indices[np.asarray(pool_rel, dtype=int)]
        test_idx = indices[np.asarray(test_rel, dtype=int)]
        if len(pool_idx) <= 1:
            continue
        inner_rs = (int(random_state) + int(fold_i) * 10007) % 2**31
        val_ratio = 1.0 / 8.0
        if (
            task in ("classification", "binary")
            and len(np.unique(y_arr[pool_idx])) >= 2
        ):
            inner = StratifiedShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=inner_rs
            )
            try:
                tr_loc, va_loc = next(
                    inner.split(np.arange(len(pool_idx)), y_arr[pool_idx])
                )
            except ValueError:
                inner2 = ShuffleSplit(
                    n_splits=1, test_size=val_ratio, random_state=inner_rs
                )
                tr_loc, va_loc = next(inner2.split(np.arange(len(pool_idx))))
        else:
            inner2 = ShuffleSplit(
                n_splits=1, test_size=val_ratio, random_state=inner_rs
            )
            tr_loc, va_loc = next(inner2.split(np.arange(len(pool_idx))))
        train_idx = pool_idx[np.asarray(tr_loc, dtype=int)]
        val_idx = pool_idx[np.asarray(va_loc, dtype=int)]
        cv_folds.append(
            {
                "train": np.sort(train_idx).astype(int).tolist(),
                "val": np.sort(val_idx).astype(int).tolist(),
                "test": np.sort(test_idx).astype(int).tolist(),
            }
        )
    return {"cv_folds": cv_folds}


def prepare_dataset(
    name, unified_root="data/unified_format", output_root="data/processed"
):
    reference = dataset_info(name)
    source = Path(unified_root) / name
    output = Path(output_root)
    destination = output / name
    if destination.exists():
        verify_dataset(name, output)
        return destination
    raw_metadata = json.loads((source / "metadata.json").read_text())
    X = np.load(source / "X.npy", allow_pickle=True)
    y = np.load(source / "y.npy", allow_pickle=True)
    task = reference["metadata"]["task"]
    if raw_metadata.get("task") != task:
        raise ValueError(f"Source target task differs: {name}")
    if task in ("binary", "classification"):
        y = _normalize_labels(y, name)
    _check_data(X, y, task)
    split_file = source / "split_indices.npy"
    if split_file.is_file():
        splits = np.load(split_file, allow_pickle=True).item()
        if (
            reference["metadata"].get("split_strategy") == "cv"
            and "cv_folds" not in splits
        ):
            splits = _generate_nested_cv_folds_split_indices(
                n_samples=len(y), y=y, task=task
            )
    elif reference["metadata"].get("split_strategy") == "cv":
        if reference["metadata"].get("split_group"):
            raise ValueError(f"Grouped task requires prepared source splits: {name}")
        splits = _generate_nested_cv_folds_split_indices(
            n_samples=len(y), y=y, task=task
        )
    else:
        order = np.random.RandomState(42).permutation(len(y))
        n_test, n_val = int(len(y) * 0.15), int(len(y) * 0.15)
        splits = dict(
            train=order[n_test + n_val :].tolist(),
            val=order[n_test : n_test + n_val].tolist(),
            test=order[:n_test].tolist(),
        )
    fit_rows = None if "cv_folds" in splits else np.asarray(splits["train"], dtype=int)
    categories = raw_metadata.get("cat_idx", [])
    selection = reference.get("source_feature_indices")
    if selection is not None:
        if X.shape[1] != selection["input_width"]:
            raise ValueError(f"Source feature width differs: {name}")
        columns = selection["indices"]
        X = X[:, columns]
    else:
        X, columns = _drop_constant_and_empty_columns(
            X, dataset_name=name, fit_rows=fit_rows
        )
    metadata = {**raw_metadata, **reference["metadata"], "READY": True}
    metadata.update(
        num_samples=len(y),
        num_features=X.shape[1],
        cat_idx=[columns.index(i) for i in categories if i in columns],
    )
    splits.update(reference.get("split_metadata", {}))
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + name + "-", dir=output))
    try:
        directory = staging / name
        directory.mkdir()
        np.save(directory / "X.npy", X)
        np.save(directory / "y.npy", y)
        np.save(directory / "split_indices.npy", splits, allow_pickle=True)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        verify_dataset(name, staging)
        for filename in ("sample_metadata.csv", "feature_metadata.csv", "LICENSE"):
            if (source / filename).is_file():
                shutil.copyfile(source / filename, directory / filename)
        directory.rename(destination)
    finally:
        shutil.rmtree(staging)
    return destination
