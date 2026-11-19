# code from https://github.com/naszilla/tabzilla
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from loguru import logger
from tabbiomed.trainer.utils.preprocessing import scale_splits_fit_on_train
from tabbiomed.data.catalog import dataset_names


all_datasets = dataset_names()


def _validate_indices(indices, size, name):
    values = np.asarray(indices, dtype=object)
    if values.ndim != 1 or any(
        not isinstance(index, (int, np.integer))
        or isinstance(index, (bool, np.bool_))
        for index in values
    ):
        raise ValueError(f"{name} must be a one-dimensional sequence of integers")
    if any(index < 0 or index >= size for index in values):
        raise ValueError(f"{name} contains indices outside [0, {size})")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate indices")
    return np.asarray(values, dtype=np.intp)


class TabularDataset(object):
    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        task: str,
        num_classes: Optional[int] = None,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
        cat_dims: Optional[list] = None,
        split_indices: Optional[dict] = None,
        source: str | None = None,
        description: str | None = None,
        label: str | None = None,
        force_scale: bool = False,
        groups: Optional[np.ndarray] = None,
    ) -> None:
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert isinstance(y, np.ndarray), "y must be an instance of np.ndarray"
        assert X.shape[0] == y.shape[0], (
            "X and y must match along their 0-th dimensions"
        )
        assert len(X.shape) == 2, "X must be 2-dimensional"
        if task != "multiregression":
            assert len(y.shape) == 1, "y must be 1-dimensional"
        else:
            assert len(y.shape) == 2, "y must be 2-dimensional for multiregression"
        if num_samples is not None:
            assert X.shape[0] == num_samples, (
                f"first dimension of X must be equal to num_samples. X has shape {X.shape}"
            )
            assert y.shape[0] == num_samples, (
                f"first dimension of y must be equal to num_samples. y has shape {y.shape} and num_samples={num_samples}"
            )
        else:
            num_samples = X.shape[0]
        if num_features is not None:
            assert X.shape[1] == num_features, (
                f"second dimension of X must be equal to num_features. X has shape {X.shape}"
            )
        else:
            num_features = X.shape[1]
        _validate_indices(cat_idx, num_features, "cat_idx")
        assert task in ["regression", "classification", "binary", "multiregression"]
        if not isinstance(split_indices, dict):
            raise ValueError("split_indices must be a dictionary")
        folds = split_indices.get("cv_folds", [split_indices])
        if not isinstance(folds, list) or not folds:
            raise ValueError("cv_folds must be a nonempty list of dictionaries")
        for fold_id, fold in enumerate(folds):
            if not isinstance(fold, dict) or not all(
                part in fold for part in ("train", "val", "test")
            ):
                raise ValueError(f"Split {fold_id} must contain train, val and test")
            indices = {
                part: _validate_indices(
                    fold[part], num_samples, f"Split {fold_id} {part}"
                )
                for part in ("train", "val", "test")
            }
            combined = np.concatenate(list(indices.values()))
            if len(np.unique(combined)) != len(combined):
                raise ValueError(f"Split {fold_id} train, val and test must be disjoint")
        if task not in ["regression", "multiregression"] and num_classes is not None:
            assert len(set(list(y))) == num_classes, (
                "num_classes does not match the unique labels in y"
            )
        else:
            num_classes = (
                1 if task in ["regression", "multiregression"] else len(set(list(y)))
            )
        if task == "regression":
            assert num_classes == 1
        elif task == "binary":
            assert num_classes == 2
        elif task == "classification":
            assert num_classes > 2
        self.name = name
        self.source = source
        self.description = description
        self.biomed_label = label
        self.X = X
        self.y = y
        self.cat_idx = cat_idx
        self.task = task
        self.num_classes = num_classes
        self.num_features = num_features
        self.cat_dims = cat_dims
        self.num_samples = num_samples
        self.split_indices = split_indices
        self.num_outputs = y.shape[1] if task == "multiregression" else 1
        self.force_scale = force_scale
        self.groups = (
            np.asarray(groups)
            if groups is not None
            else np.asarray(split_indices["groups"])
            if "groups" in split_indices
            else None
        )
        if self.groups is not None and (
            self.groups.ndim != 1 or len(self.groups) != num_samples
        ):
            raise ValueError("Sample groups must have one value per dataset row")
        if "cv_folds" in split_indices:
            self.X_train = np.array([]).reshape(0, X.shape[1])
            self.X_val = np.array([]).reshape(0, X.shape[1])
            self.X_test = np.array([]).reshape(0, X.shape[1])
            self.y_train = np.array([])
            self.y_val = np.array([])
            self.y_test = np.array([])
        else:
            train_idx = indices["train"]
            val_idx = indices["val"]
            test_idx = indices["test"]
            self.X_train = X[train_idx]
            self.X_val = X[val_idx]
            self.X_test = (
                X[test_idx]
                if len(test_idx) > 0
                else np.array([]).reshape(0, X.shape[1])
            )
            self.y_train = y[train_idx]
            self.y_val = y[val_idx]
            self.y_test = y[test_idx] if len(test_idx) > 0 else np.array([])
        return

    def target_encode(self):
        """Normalize classification targets and refresh their stored partitions."""
        if self.task not in ("binary", "classification"):
            return
        normalized = np.issubdtype(self.y.dtype, np.number) and np.array_equal(
            np.unique(self.y), np.arange(self.num_classes)
        )
        if not normalized:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(self.y)
            assert self.num_classes == len(encoder.classes_), (
                "num_classes was set incorrectly."
            )
            self.y = encoded
            self._target_encoder = encoder
        if "cv_folds" not in self.split_indices:
            for part in ("train", "val", "test"):
                rows = np.asarray(self.split_indices[part], dtype=int)
                setattr(self, "y_" + part, self.y[rows])

    def cat_feature_encode(self, strategy="onehot"):
        if not self.cat_idx:
            return
        if not self.cat_dims is None:
            raise RuntimeError(
                "cat_dims is already set. Categorical feature encoding might be running twice."
            )
        binary_cat_idx = []
        non_binary_cat_idx = []
        for cat_idx_val in self.cat_idx:
            cat_col_train = self.X_train[:, cat_idx_val]
            try:
                try:
                    cat_col_numeric = cat_col_train.astype(float)
                    valid_mask = ~np.isnan(cat_col_numeric)
                    if valid_mask.sum() == 0:
                        non_binary_cat_idx.append(cat_idx_val)
                        continue
                    unique_vals = np.unique(cat_col_numeric[valid_mask])
                    is_numeric = True
                except (ValueError, TypeError):
                    non_missing = cat_col_train[~pd.isna(cat_col_train)]
                    if len(non_missing) == 0:
                        non_binary_cat_idx.append(cat_idx_val)
                        continue
                    try:
                        unique_vals = np.unique(non_missing)
                    except TypeError:
                        seen_strs = {}
                        for val in non_missing:
                            seen_strs.setdefault(str(val), val)
                        unique_vals = np.array(
                            [seen_strs[k] for k in sorted(seen_strs)], dtype=object
                        )
                    is_numeric = False
                if len(unique_vals) == 2:
                    if is_numeric:
                        if all(
                            (
                                np.isclose(val, 0, atol=1e-06)
                                or np.isclose(val, 1, atol=1e-06)
                                for val in unique_vals
                            )
                        ):
                            binary_cat_idx.append(cat_idx_val)
                        else:
                            non_binary_cat_idx.append(cat_idx_val)
                    else:
                        binary_cat_idx.append(cat_idx_val)
                else:
                    non_binary_cat_idx.append(cat_idx_val)
            except Exception:
                non_binary_cat_idx.append(cat_idx_val)
        encoding_dtype = self.X.dtype
        if self.X.dtype.kind in "uSU":
            encoding_dtype = object
        elif self.X.dtype.kind in "bi" and (
            self.X.dtype.kind == "b"
            or self.X_train.shape[0] - 1 > np.iinfo(self.X.dtype).max
        ):
            encoding_dtype = np.intp
        if binary_cat_idx or encoding_dtype != self.X.dtype:
            self.X = self.X.astype(encoding_dtype, copy=True)
            self.X_train = self.X_train.astype(encoding_dtype, copy=False)
            self.X_val = self.X_val.astype(encoding_dtype, copy=False)
            self.X_test = self.X_test.astype(encoding_dtype, copy=False)
        if binary_cat_idx:
            unknown_value = -1.0
            for cat_idx_val in binary_cat_idx:
                cat_col_train = self.X_train[:, cat_idx_val]
                try:
                    cat_col_numeric = cat_col_train.astype(float)
                    valid_mask = ~np.isnan(cat_col_numeric)
                    unique_vals = (
                        np.unique(cat_col_numeric[valid_mask])
                        if valid_mask.sum() > 0
                        else []
                    )
                except (ValueError, TypeError):
                    non_missing = cat_col_train[~pd.isna(cat_col_train)]
                    try:
                        unique_vals = (
                            np.unique(non_missing) if len(non_missing) > 0 else []
                        )
                    except TypeError:
                        seen_strs = {}
                        for val in non_missing:
                            seen_strs.setdefault(str(val), val)
                        unique_vals = [seen_strs[k] for k in sorted(seen_strs)]
                if len(unique_vals) == 2:
                    v0, v1 = (unique_vals[0], unique_vals[1])
                    val_to_num = {v0: 0.0, v1: 1.0}
                    try:
                        val_to_num[float(v0)], val_to_num[float(v1)] = (0.0, 1.0)
                    except (TypeError, ValueError):
                        pass

                    def convert_to_binary(arr):
                        result = np.full(len(arr), unknown_value, dtype=float)
                        for i, v in enumerate(arr):
                            try:
                                if isinstance(v, (int, float, np.number)) and np.isnan(
                                    v
                                ):
                                    result[i] = unknown_value
                                else:
                                    key = v
                                    if key not in val_to_num:
                                        try:
                                            key = float(v)
                                        except (TypeError, ValueError):
                                            try:
                                                key = int(v)
                                            except (TypeError, ValueError):
                                                key = (
                                                    v.item()
                                                    if hasattr(v, "item")
                                                    else v
                                                )
                                    result[i] = val_to_num.get(key, unknown_value)
                            except (TypeError, ValueError):
                                result[i] = val_to_num.get(v, unknown_value)
                        return result

                    self.X_train[:, cat_idx_val] = convert_to_binary(
                        self.X_train[:, cat_idx_val]
                    )
                    self.X_val[:, cat_idx_val] = convert_to_binary(
                        self.X_val[:, cat_idx_val]
                    )
                    self.X_test[:, cat_idx_val] = convert_to_binary(
                        self.X_test[:, cat_idx_val]
                    )
            if self.split_indices is not None and "cv_folds" not in self.split_indices:
                for cat_idx_val in binary_cat_idx:
                    self.X[:, cat_idx_val] = unknown_value
                    self.X[
                        np.array(self.split_indices["train"], dtype=int), cat_idx_val
                    ] = self.X_train[:, cat_idx_val]
                    self.X[
                        np.array(self.split_indices["val"], dtype=int), cat_idx_val
                    ] = self.X_val[:, cat_idx_val]
                    if (
                        "test" in self.split_indices
                        and len(self.split_indices["test"]) > 0
                    ):
                        self.X[
                            np.array(self.split_indices["test"], dtype=int), cat_idx_val
                        ] = self.X_test[:, cat_idx_val]
            else:
                for cat_idx_val in binary_cat_idx:
                    cat_col = self.X[:, cat_idx_val]
                    try:
                        cat_col_numeric = cat_col.astype(float)
                        valid_mask = ~np.isnan(cat_col_numeric)
                        unique_vals = (
                            np.unique(cat_col_numeric[valid_mask])
                            if valid_mask.sum() > 0
                            else []
                        )
                    except (ValueError, TypeError):
                        unique_vals = np.unique(cat_col)
                        try:
                            unique_vals = unique_vals[
                                ~np.isnan(unique_vals.astype(float))
                            ]
                        except:
                            pass
                    if len(unique_vals) == 2:
                        v0, v1 = (unique_vals[0], unique_vals[1])
                        val_to_num = {v0: 0.0, v1: 1.0}
                        try:
                            val_to_num[float(v0)], val_to_num[float(v1)] = (0.0, 1.0)
                        except (TypeError, ValueError):
                            pass

                        def _lookup(u):
                            if isinstance(u, (int, float, np.number)) and np.isnan(u):
                                return unknown_value
                            k = u
                            if k not in val_to_num:
                                try:
                                    k = float(u)
                                except (TypeError, ValueError):
                                    try:
                                        k = int(u)
                                    except (TypeError, ValueError):
                                        k = getattr(u, "item", lambda: u)()
                            return val_to_num.get(k, unknown_value)

                        self.X[:, cat_idx_val] = np.array(
                            [_lookup(v) for v in cat_col], dtype=float
                        )
        if not non_binary_cat_idx:
            self.cat_dims = [2] * len(binary_cat_idx)
            return
        if strategy == "onehot":
            num_idx = [i for i in range(self.num_features) if i not in self.cat_idx]
            cat_cols_train = self.X_train[:, non_binary_cat_idx]
            try:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
            enc.fit(cat_cols_train)
            cat_cols_train_encoded = enc.transform(cat_cols_train)
            if self.X_val.shape[0] > 0:
                cat_cols_val_encoded = enc.transform(self.X_val[:, non_binary_cat_idx])
            else:
                cat_cols_val_encoded = np.zeros(
                    (0, cat_cols_train_encoded.shape[1]),
                    dtype=cat_cols_train_encoded.dtype,
                )
            if self.X_test.shape[0] > 0:
                cat_cols_test_encoded = enc.transform(
                    self.X_test[:, non_binary_cat_idx]
                )
            else:
                cat_cols_test_encoded = np.zeros(
                    (0, cat_cols_train_encoded.shape[1]),
                    dtype=cat_cols_train_encoded.dtype,
                )
            num_cols_train = self.X_train[:, num_idx] if num_idx else np.array([])
            if num_idx:
                num_cols_val = (
                    self.X_val[:, num_idx]
                    if self.X_val.shape[0] > 0
                    else np.zeros((0, len(num_idx)), dtype=float)
                )
            else:
                num_cols_val = np.array([])
            if num_idx:
                num_cols_test = (
                    self.X_test[:, num_idx]
                    if self.X_test.shape[0] > 0
                    else np.zeros((0, len(num_idx)), dtype=float)
                )
            else:
                num_cols_test = np.array([])
            binary_cat_cols_train = (
                self.X_train[:, binary_cat_idx] if binary_cat_idx else np.array([])
            )
            if binary_cat_idx:
                binary_cat_cols_val = (
                    self.X_val[:, binary_cat_idx]
                    if self.X_val.shape[0] > 0
                    else np.zeros((0, len(binary_cat_idx)), dtype=float)
                )
            else:
                binary_cat_cols_val = np.array([])
            if binary_cat_idx:
                binary_cat_cols_test = (
                    self.X_test[:, binary_cat_idx]
                    if self.X_test.shape[0] > 0
                    else np.zeros((0, len(binary_cat_idx)), dtype=float)
                )
            else:
                binary_cat_cols_test = np.array([])
            if len(num_idx) > 0:
                try:
                    num_cols_train = num_cols_train.astype(float)
                    num_cols_val = num_cols_val.astype(float)
                    num_cols_test = num_cols_test.astype(float)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Could not convert numerical columns to float. This might indicate non-numeric values in columns {num_idx}. Error: {e}"
                    )
            if len(binary_cat_idx) > 0:
                binary_cat_cols_train = binary_cat_cols_train.astype(float)
                binary_cat_cols_val = binary_cat_cols_val.astype(float)
                binary_cat_cols_test = binary_cat_cols_test.astype(float)
            parts_train = []
            parts_val = []
            parts_test = []
            if len(num_idx) > 0:
                parts_train.append(num_cols_train)
                parts_val.append(num_cols_val)
                parts_test.append(num_cols_test)
            if len(binary_cat_idx) > 0:
                parts_train.append(binary_cat_cols_train)
                parts_val.append(binary_cat_cols_val)
                parts_test.append(binary_cat_cols_test)
            if len(non_binary_cat_idx) > 0:
                parts_train.append(cat_cols_train_encoded)
                parts_val.append(cat_cols_val_encoded)
                parts_test.append(cat_cols_test_encoded)
            if parts_train:
                self.X_train = np.hstack(parts_train).astype(float)
                self.X_val = np.hstack(parts_val).astype(float)
                self.X_test = np.hstack(parts_test).astype(float)
            else:
                self.X_train = self.X_train.astype(float)
                self.X_val = self.X_val.astype(float)
                self.X_test = self.X_test.astype(float)
            if self.split_indices is not None and "cv_folds" not in self.split_indices:
                X_encoded = np.zeros((self.X.shape[0], self.X_train.shape[1]))
                X_encoded[np.array(self.split_indices["train"], dtype=int)] = (
                    self.X_train
                )
                X_encoded[np.array(self.split_indices["val"], dtype=int)] = self.X_val
                if "test" in self.split_indices and len(self.split_indices["test"]) > 0:
                    X_encoded[np.array(self.split_indices["test"], dtype=int)] = (
                        self.X_test
                    )
                self.X = X_encoded
            else:
                cat_cols = self.X[:, non_binary_cat_idx]
                cat_cols_encoded = enc.transform(cat_cols)
                parts = []
                if len(num_idx) > 0:
                    parts.append(self.X[:, num_idx].astype(float))
                if len(binary_cat_idx) > 0:
                    parts.append(self.X[:, binary_cat_idx].astype(float))
                if len(non_binary_cat_idx) > 0:
                    parts.append(cat_cols_encoded.astype(float))
                if parts:
                    self.X = np.hstack(parts).astype(float)
                else:
                    self.X = self.X.astype(float)
            self.num_features = self.X.shape[1]
            self.cat_idx = list(range(len(num_idx), self.num_features))
            self.cat_dims = [2] * len(self.cat_idx)
            self._categorical_encoder = enc
        elif strategy == "label":
            cardinalities = {i: 2 for i in binary_cat_idx}
            label_encoders = {}
            for i in non_binary_cat_idx:
                le = LabelEncoder()
                le.fit(self.X_train[:, i])
                label_encoders[i] = le
                categories = pd.Index(le.classes_)
                for partition in (self.X_train, self.X_val, self.X_test):
                    partition[:, i] = categories.get_indexer(partition[:, i])
                cardinalities[i] = len(le.classes_)
            self.cat_dims = [cardinalities[i] for i in self.cat_idx]
            if self.split_indices is not None and "cv_folds" not in self.split_indices:
                X_encoded = self.X.copy()
                for i in non_binary_cat_idx:
                    X_encoded[:, i] = -1
                X_encoded[np.array(self.split_indices["train"], dtype=int)] = (
                    self.X_train
                )
                X_encoded[np.array(self.split_indices["val"], dtype=int)] = self.X_val
                if "test" in self.split_indices and len(self.split_indices["test"]) > 0:
                    X_encoded[np.array(self.split_indices["test"], dtype=int)] = (
                        self.X_test
                    )
                self.X = X_encoded
            else:
                for i in non_binary_cat_idx:
                    self.X[:, i] = label_encoders[i].transform(self.X[:, i])
            self._label_encoders = label_encoders
        else:
            raise ValueError(f"Unknown encoding strategy: {strategy}")

    def impute_missing_values(self, strategy="knn", **kwargs):
        from sklearn.impute import KNNImputer, SimpleImputer
        import numpy as np

        cat_indices = set(self.cat_idx) if self.cat_idx else set()
        num_indices = [i for i in range(self.X.shape[1]) if i not in cat_indices]
        if not num_indices:
            return
        num_cols_train = self.X_train[:, num_indices].astype(float)
        has_nan_train = np.any(np.isnan(num_cols_train))
        has_nan_val = self.X_val.shape[0] > 0 and np.any(
            np.isnan(self.X_val[:, num_indices].astype(float))
        )
        has_nan_test = self.X_test.shape[0] > 0 and np.any(
            np.isnan(self.X_test[:, num_indices].astype(float))
        )
        if not (has_nan_train or has_nan_val or has_nan_test):
            return
        if strategy == "knn":
            n_neighbors = kwargs.get("n_neighbors", 5)
            imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
            imputer.fit(num_cols_train)
        elif strategy in ["mean", "median"]:
            imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)
            imputer.fit(num_cols_train)
        elif strategy == "constant":
            if "fill_value" not in kwargs:
                raise ValueError("fill_value must be provided for constant imputation")
            imputer = SimpleImputer(
                strategy="constant",
                fill_value=kwargs["fill_value"],
                keep_empty_features=True,
            )
            imputer.fit(num_cols_train)
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
        self._imputer = imputer
        if self.X.dtype.kind in "SU":
            self.X = self.X.astype(object, copy=True)
            self.X_train = self.X_train.astype(object, copy=False)
            self.X_val = self.X_val.astype(object, copy=False)
            self.X_test = self.X_test.astype(object, copy=False)
        num_cols_train_imputed = imputer.transform(num_cols_train)
        self.X_train[:, num_indices] = num_cols_train_imputed
        if self.X_val.shape[0] > 0:
            num_cols_val = self.X_val[:, num_indices].astype(float)
            num_cols_val_imputed = imputer.transform(num_cols_val)
            self.X_val[:, num_indices] = num_cols_val_imputed
        if self.X_test.shape[0] > 0:
            num_cols_test = self.X_test[:, num_indices].astype(float)
            num_cols_test_imputed = imputer.transform(num_cols_test)
            self.X_test[:, num_indices] = num_cols_test_imputed
        if self.split_indices is not None and "cv_folds" not in self.split_indices:
            X_imputed = np.zeros_like(self.X)
            X_imputed[np.array(self.split_indices["train"], dtype=int)] = self.X_train
            X_imputed[np.array(self.split_indices["val"], dtype=int)] = self.X_val
            if "test" in self.split_indices and len(self.split_indices["test"]) > 0:
                X_imputed[np.array(self.split_indices["test"], dtype=int)] = self.X_test
            self.X = X_imputed
        else:
            self.X[:, num_indices] = imputer.transform(
                self.X[:, num_indices].astype(float)
            )

    def scale_features(self, scaler_type="standard", random_state=None):
        Xtr, Xva, Xte, scaler, plan = scale_splits_fit_on_train(
            self.X_train,
            self.X_val,
            self.X_test,
            cat_idx=getattr(self, "cat_idx", None),
            scaler_type=scaler_type,
            random_state=random_state,
        )
        self.X_train, self.X_val, self.X_test = (Xtr, Xva, Xte)
        self._scaler = scaler
        self._scale_indices = plan.scale_indices
        self._binary_col_indices = plan.binary_indices
        if self.split_indices is not None and "cv_folds" not in self.split_indices:
            X_scaled = self.X.astype(self.X_train.dtype, copy=True)
            X_scaled[np.array(self.split_indices["train"], dtype=int)] = self.X_train
            X_scaled[np.array(self.split_indices["val"], dtype=int)] = self.X_val
            if "test" in self.split_indices and len(self.split_indices["test"]) > 0:
                X_scaled[np.array(self.split_indices["test"], dtype=int)] = self.X_test
            self.X = X_scaled
        else:
            self.X = self.X.copy()

    def get_cv_splits(self):
        if self.split_indices is not None and "cv_folds" in self.split_indices:
            custom_folds = self.split_indices["cv_folds"]
            for fold_id, fold in enumerate(custom_folds):
                for part in ("train", "val"):
                    if len(fold[part]) == 0:
                        raise ValueError(f"CV fold {fold_id} has empty {part} indices")
            logger.info(
                f"Using {len(custom_folds)} predefined CV folds from split_indices"
            )
            for fold in custom_folds:
                train_idx = np.array(fold["train"], dtype=int)
                val_idx = np.array(fold["val"], dtype=int)
                test_idx = np.array(fold.get("test", []), dtype=int)
                fold_split_indices = {
                    "train": train_idx.tolist(),
                    "val": val_idx.tolist(),
                    "test": test_idx.tolist(),
                }
                fold_dataset = TabularDataset(
                    name=self.name,
                    X=self.X,
                    y=self.y,
                    cat_idx=list(self.cat_idx),
                    task=self.task,
                    num_classes=self.num_classes,
                    num_features=self.num_features,
                    num_samples=self.num_samples,
                    cat_dims=list(self.cat_dims) if self.cat_dims is not None else None,
                    split_indices=fold_split_indices,
                    source=self.source,
                    description=self.description,
                    label=self.biomed_label,
                    force_scale=self.force_scale,
                    groups=self.groups,
                )
                fold_dataset.is_cv_fold = True
                yield (fold_dataset, train_idx, val_idx)
            return
        raise ValueError(
            "CV split strategy is misconfigured. Expected predefined 'cv_folds' in split_indices. Run preprocessing to generate and save cv_folds into split_indices.npy."
        )

    def prepare_dataset(
        self,
        inpute_mv,
        encode_cat,
        scale_features=False,
        mv_strategy="knn",
        cat_encoding_strategy="onehot",
        scaler_type="standard",
        random_state=None,
        **kwargs,
    ):
        if (
            inpute_mv
            and self.split_indices is not None
            and ("cv_folds" not in self.split_indices)
        ):
            self.impute_missing_values(strategy=mv_strategy, **kwargs)
        elif inpute_mv and (
            self.split_indices is not None and "cv_folds" in self.split_indices
        ):
            logger.info(
                "CV mode detected: skipping imputation in prepare_dataset() - will impute per-fold to avoid data leakage"
            )
        if (
            encode_cat
            and self.split_indices is not None
            and ("cv_folds" not in self.split_indices)
        ):
            self.cat_feature_encode(strategy=cat_encoding_strategy)
        elif encode_cat and (
            self.split_indices is not None and "cv_folds" in self.split_indices
        ):
            logger.info(
                "CV mode detected: skipping categorical encoding in prepare_dataset() - will encode per-fold to avoid data leakage"
            )
        if (
            encode_cat
            and self.split_indices is not None
            and ("cv_folds" not in self.split_indices)
            and (self.X.dtype.kind == "O")
        ):
            try:
                self.X = self.X.astype(np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    "Feature matrix has object dtype and could not be converted to float. Ensure all features are numeric or properly encoded (e.g. categorical columns encoded)."
                ) from e
        _cv_fold_no_transform = (
            getattr(self, "is_cv_fold", False)
            and (not inpute_mv)
            and (not encode_cat)
            and (not scale_features)
        )
        if (
            self.split_indices is not None
            and "cv_folds" not in self.split_indices
            and (not _cv_fold_no_transform)
        ):
            self.X_train = self.X[np.array(self.split_indices["train"], dtype=int)]
            self.X_val = self.X[np.array(self.split_indices["val"], dtype=int)]
            if "test" in self.split_indices and len(self.split_indices["test"]) > 0:
                self.X_test = self.X[np.array(self.split_indices["test"], dtype=int)]
            else:
                self.X_test = np.array([]).reshape(0, self.X.shape[1])
        if scale_features:
            if self.split_indices is not None and "cv_folds" in self.split_indices:
                logger.info(
                    "CV mode detected: skipping feature scaling in prepare_dataset() - will scale per-fold to avoid data leakage"
                )
            else:
                self.scale_features(scaler_type=scaler_type, random_state=random_state)
                if self.split_indices is not None:
                    self.X_train = self.X[
                        np.array(self.split_indices["train"], dtype=int)
                    ]
                    self.X_val = self.X[np.array(self.split_indices["val"], dtype=int)]
                    if (
                        "test" in self.split_indices
                        and len(self.split_indices["test"]) > 0
                    ):
                        self.X_test = self.X[
                            np.array(self.split_indices["test"], dtype=int)
                        ]
                    else:
                        self.X_test = np.array([]).reshape(0, self.X.shape[1])

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
            "target_type": self.task,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "num_instances": self.num_samples,
            "source": self.source,
            "description": self.description,
        }

    @classmethod
    def read(cls, dataset_name: str, datasets_path="data/processed"):
        from tabbiomed.data import ensure_dataset
        from tabbiomed.data.storage import load_arrays, load_metadata, load_splits

        directory = ensure_dataset(dataset_name, datasets_path)
        root = directory.parent
        meta = dict(load_metadata(dataset_name, root))
        X, y = load_arrays(dataset_name, root)
        split_indices = load_splits(dataset_name, root)
        meta.update(X=X, y=y, split_indices=split_indices)
        if "groups" in split_indices:
            from .cv_splits import load_sample_groups

            meta["groups"] = load_sample_groups(directory, split_indices)
        accepted = {
            "name",
            "X",
            "y",
            "cat_idx",
            "task",
            "num_classes",
            "num_features",
            "num_samples",
            "cat_dims",
            "split_indices",
            "source",
            "description",
            "label",
            "force_scale",
            "groups",
        }
        return cls(**{key: value for key, value in meta.items() if key in accepted})
