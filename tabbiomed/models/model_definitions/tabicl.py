"""TabICLv2 classification and regression through in-context learning.

https://github.com/soda-inria/tabicl"""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union

from loguru import logger

from tabbiomed.models.base_model import BaseModel, ModelArgs

try:
    from tabicl import TabICLClassifier, TabICLRegressor
    from tabbiomed.models.tabicl_memory import bounded_inference
except ImportError as e:
    raise ImportError(
        "tabicl is required for TabICL. Install with: uv pip install tabicl"
    ) from e

try:
    from torch.cuda import is_available as cuda_is_available
except ImportError:
    cuda_is_available = lambda: False


class TabICL(BaseModel):
    """TabICLv2 estimators with native preprocessing and bounded inference memory."""

    @classmethod
    def get_base_params(cls, args):
        return {
            "apply_pca": False,
            "pca_max_features": 500,
            "pca_fit": "lowrank",
            "pca_impute_missing": True,
            "pca_imputation_strategy": "mean",
            "pca_knn_imputer_neighbors": 5,
            "pca_cat_encoding": "label",
            "device": None,
            "batch_size": None,
            "n_estimators": 8,
            "norm_methods": None,
            "feat_shuffle_method": "latin",
            "model_path": None,
            "allow_auto_download": True,
            "checkpoint_version": None,
            "verbose": False,
            "use_amp": "auto",
            "use_fa3": "auto",
            "offload_mode": "auto",
            "disk_offload_dir": None,
            "n_jobs": None,
            "inference_config": None,
            "outlier_threshold": 4.0,
            "class_shuffle_method": "shift",
            "softmax_temperature": 0.9,
            "average_logits": True,
            "support_many_classes": True,
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)

        self.apply_pca = params["apply_pca"]
        self.pca_max_features = params["pca_max_features"]
        self.pca_fit = params["pca_fit"]
        self.pca_impute_missing = params["pca_impute_missing"]
        self.pca_imputation_strategy = params["pca_imputation_strategy"]
        self.pca_knn_imputer_neighbors = params["pca_knn_imputer_neighbors"]
        self.pca_cat_encoding = params["pca_cat_encoding"]
        self.pca = None
        self.pca_imputer_ = None
        self.pca_cat_maps_ = {}
        self.pca_preprocess_n_features_in_ = None

        device = params["device"]
        if device is None and getattr(args, "use_gpu", False) and cuda_is_available():
            device = (
                f"cuda:{args.gpu_ids[0]}"
                if getattr(args, "gpu_ids", None) and len(args.gpu_ids) > 0
                else "cuda"
            )
        if getattr(args, "use_gpu", False) and not cuda_is_available():
            logger.warning("GPU requested but CUDA not available. Using CPU.")
        self.device = device or "cpu"
        self.batch_size = params["batch_size"]

        common_kw = {
            "n_estimators": params["n_estimators"],
            "batch_size": self.batch_size if self.batch_size is not None else 8,
            "norm_methods": params["norm_methods"],
            "feat_shuffle_method": params["feat_shuffle_method"],
            "random_state": getattr(args, "seed", 42),
            "model_path": params["model_path"],
            "allow_auto_download": params["allow_auto_download"],
            "checkpoint_version": params["checkpoint_version"],
            "device": device,
            "verbose": params["verbose"],
            "use_amp": params["use_amp"],
            "use_fa3": params["use_fa3"],
            "offload_mode": params["offload_mode"],
            "disk_offload_dir": params["disk_offload_dir"],
            "n_jobs": params["n_jobs"],
            "inference_config": params["inference_config"],
        }
        for key in (
            "embed_dim",
            "col_num_blocks",
            "row_num_blocks",
            "icl_num_blocks",
            "max_classes",
            "recompute",
        ):
            if key in params:
                common_kw[key] = params[key]
        common_kw = {k: v for k, v in common_kw.items() if v is not None}

        if args.objective in ["regression", "multiregression"]:
            reg_common_kw = dict(common_kw)

            reg_common_kw["max_classes"] = 0
            reg_kw = self._filter_supported_kwargs(
                TabICLRegressor,
                {
                    "outlier_threshold": params["outlier_threshold"],
                    **reg_common_kw,
                },
            )
            self.model = TabICLRegressor(
                **reg_kw,
            )
        elif args.objective in ["classification", "binary"]:
            clf_kw = self._filter_supported_kwargs(
                TabICLClassifier,
                {
                    "class_shuffle_method": params["class_shuffle_method"],
                    "outlier_threshold": params["outlier_threshold"],
                    "softmax_temperature": params["softmax_temperature"],
                    "average_logits": params["average_logits"],
                    "support_many_classes": params["support_many_classes"],
                    **common_kw,
                },
            )
            self.model = TabICLClassifier(**clf_kw)
        else:
            raise ValueError(
                f"TabICL does not support objective={args.objective!r}. "
                "Use 'classification', 'binary', 'regression', or 'multiregression'."
            )

    @staticmethod
    def _filter_supported_kwargs(model_ctor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Pass only kwargs supported by the installed tabicl package version."""
        try:
            sig = inspect.signature(model_ctor.__init__)
        except (TypeError, ValueError):
            return {k: v for k, v in kwargs.items() if v is not None}

        supports_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if supports_var_kw:
            return {k: v for k, v in kwargs.items() if v is not None}

        allowed = {
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        unsupported = sorted(
            k for k in kwargs if k not in allowed and kwargs[k] is not None
        )
        if unsupported:
            logger.debug(
                "Skipping unsupported TabICL constructor kwargs for installed package: {}",
                unsupported,
            )
        return {k: v for k, v in kwargs.items() if k in allowed and v is not None}

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return False

    @classmethod
    def needs_scaling(cls) -> bool:
        return False

    @classmethod
    def get_cat_encoding_strategy(cls) -> str:
        return "label"

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        """TabICL regressor is single-target, multi-target uses BaseModel's per-output training."""
        return False

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "n_estimators": 8,
            "batch_size": None,
            "outlier_threshold": 4.0,
            "norm_methods": None,
            "feat_shuffle_method": "latin",
            "class_shuffle_method": "shift",
            "softmax_temperature": 0.9,
            "average_logits": True,
            "support_many_classes": True,
            "model_path": None,
            "allow_auto_download": True,
            "checkpoint_version": None,
            "device": None,
            "verbose": False,
            "use_amp": "auto",
            "use_fa3": "auto",
            "offload_mode": "auto",
            "disk_offload_dir": None,
            "n_jobs": None,
            "inference_config": None,
            "apply_pca": False,
            "pca_max_features": 500,
            "pca_fit": "lowrank",
            "pca_impute_missing": True,
            "pca_imputation_strategy": "mean",
            "pca_knn_imputer_neighbors": 5,
            "pca_cat_encoding": "label",
        }

    # Batch-size heuristic adapted from https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/models/tabicl/model.py
    @staticmethod
    def _get_auto_batch_size(n_cells: int) -> int:
        """Keep TabArena's small-table batching, serialize large-table ensembles."""
        if n_cells <= 4_000_000:
            return 8
        if n_cells <= 6_000_000:
            return 4
        return 1

    def _pca_categorical_indices(self, n_features: int) -> list[int]:
        cat_idx = getattr(self.args, "cat_idx", None) or []
        valid_idx = []
        for idx in cat_idx:
            idx = int(idx)
            if 0 <= idx < n_features:
                valid_idx.append(idx)
        return sorted(set(valid_idx))

    @staticmethod
    def _imputer_kwargs(imputer_cls, **kwargs) -> Dict[str, Any]:
        try:
            sig = inspect.signature(imputer_cls.__init__)
        except (TypeError, ValueError):
            return kwargs
        if "keep_empty_features" in sig.parameters:
            kwargs["keep_empty_features"] = True
        return kwargs

    def _make_pca_imputer(self):
        strategy = str(self.pca_imputation_strategy).lower()
        if strategy == "knn":
            from sklearn.impute import KNNImputer

            kwargs = self._imputer_kwargs(
                KNNImputer,
                n_neighbors=self.pca_knn_imputer_neighbors,
            )
            return KNNImputer(**kwargs)

        from sklearn.impute import SimpleImputer

        if strategy == "constant":
            kwargs = {"strategy": "constant", "fill_value": 0.0}
        else:
            kwargs = {"strategy": strategy}
        return SimpleImputer(**self._imputer_kwargs(SimpleImputer, **kwargs))

    def _coerce_pca_numeric_matrix(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"PCA expects a 2D feature matrix, got shape {X_arr.shape}"
            )

        if np.issubdtype(X_arr.dtype, np.number):
            return X_arr.astype(float, copy=True)

        X_obj = X_arr.astype(object, copy=False)
        columns = [
            pd.to_numeric(pd.Series(X_obj[:, col]), errors="coerce").to_numpy(
                dtype=float
            )
            for col in range(X_obj.shape[1])
        ]
        return np.column_stack(columns)

    def _encode_pca_categoricals(
        self,
        X: np.ndarray,
        X_num: np.ndarray,
        fit: bool,
    ) -> np.ndarray:
        if str(self.pca_cat_encoding).lower() != "label":
            raise ValueError(
                "TabICL PCA currently supports pca_cat_encoding='label' only, "
                f"got {self.pca_cat_encoding!r}."
            )

        X_obj = np.asarray(X, dtype=object)
        cat_indices = self._pca_categorical_indices(X_obj.shape[1])
        if fit:
            self.pca_cat_maps_ = {}

        for col in cat_indices:
            values = X_obj[:, col]
            missing = pd.isna(values)
            encoded = np.full(values.shape[0], np.nan, dtype=float)

            if fit:
                observed = values[~missing].astype(str)
                classes = np.unique(observed)
                mapping = {str(value): idx for idx, value in enumerate(classes)}
                self.pca_cat_maps_[col] = mapping
            else:
                mapping = self.pca_cat_maps_.get(col, {})

            if mapping and np.any(~missing):
                encoded[~missing] = [
                    mapping.get(str(value), np.nan) for value in values[~missing]
                ]
            X_num[:, col] = encoded
        return X_num

    def _fit_transform_pca_preprocessor(self, X: np.ndarray) -> np.ndarray:
        X_num = self._coerce_pca_numeric_matrix(X)
        self.pca_preprocess_n_features_in_ = X_num.shape[1]
        X_num = self._encode_pca_categoricals(X, X_num, fit=True)
        X_num[~np.isfinite(X_num)] = np.nan

        if self.pca_impute_missing:
            self.pca_imputer_ = self._make_pca_imputer()
            X_num = self.pca_imputer_.fit_transform(X_num)
        else:
            self.pca_imputer_ = None
        return np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

    def _transform_pca_preprocessor(self, X: np.ndarray) -> np.ndarray:
        X_num = self._coerce_pca_numeric_matrix(X)
        if (
            self.pca_preprocess_n_features_in_ is not None
            and X_num.shape[1] != self.pca_preprocess_n_features_in_
        ):
            raise ValueError(
                f"PCA preprocessor was fitted with {self.pca_preprocess_n_features_in_} "
                f"features, but transform got {X_num.shape[1]}."
            )

        X_num = self._encode_pca_categoricals(X, X_num, fit=False)
        X_num[~np.isfinite(X_num)] = np.nan
        if self.pca_imputer_ is not None:
            X_num = self.pca_imputer_.transform(X_num)
        return np.nan_to_num(X_num, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_transform_pca(
        self,
        X: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
    ) -> tuple[np.ndarray, Union[None, np.ndarray]]:
        if not (
            self.apply_pca
            and self.pca_max_features is not None
            and X.shape[1] > self.pca_max_features
        ):
            self.pca = None
            return X, X_val

        from .torch_pca import TorchPCA

        logger.info(
            f"Applying PCA (reducing from {X.shape[1]} to at most {self.pca_max_features} features)"
        )
        self.pca = TorchPCA(n_components=self.pca_max_features, fit=self.pca_fit)

        X = self._fit_transform_pca_preprocessor(X)
        if X_val is not None:
            X_val = self._transform_pca_preprocessor(X_val)

        device = self.device
        X = self.pca.fit_transform_array(X, device)
        logger.info(
            f"PCA fitted with {self.pca.n_components_} output features on {self.pca.fit_device_}"
        )

        if X_val is not None:
            X_val = self.pca.transform_array(X_val)
        return X, X_val

    def _transform_pca(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "pca", None) is None:
            return X

        X = self._transform_pca_preprocessor(X)
        return self.pca.transform_array(X)

    def _to_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """Preserve feature types for TabICL's internal preprocessing."""
        if self.pca is not None:
            return pd.DataFrame(X)
        cat_indices = set(self._pca_categorical_indices(X.shape[1]))
        return pd.DataFrame(
            {
                col: pd.Categorical(X[:, col])
                if col in cat_indices
                else pd.to_numeric(X[:, col]).astype(float)
                for col in range(X.shape[1])
            }
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit TabICL (in-context learning, no weight updates)."""
        X, X_val = self._fit_transform_pca(X, X_val)
        self.model.batch_size = (
            self._get_auto_batch_size(X.shape[0] * X.shape[1])
            if self.batch_size is None
            else self.batch_size
        )

        X_df = self._to_dataframe(X)
        y_ser = pd.Series(y) if y.ndim == 1 else pd.DataFrame(y)
        if y_ser.ndim == 2 and y_ser.shape[1] == 1:
            y_ser = y_ser.iloc[:, 0]

        if self.args.objective in ("binary", "classification"):
            self._classes_ = np.unique(np.asarray(y).ravel())
        self.model = self.model.fit(X=X_df, y=y_ser)
        if self.args.objective in ("regression", "multiregression") and hasattr(
            self.model, "model_"
        ):
            if hasattr(self.model.model_, "max_classes"):
                self.model.model_.max_classes = 0
            if hasattr(self.model, "model_config_") and isinstance(
                self.model.model_config_, dict
            ):
                self.model.model_config_["max_classes"] = 0
        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels (classification) or continuous targets (regression)."""

        if hasattr(self, "_output_models_") and self._output_models_:
            return super().predict(X)

        assert self.model is not None, "Model must be fitted before predict."

        if self.args.objective in ("regression", "multiregression"):
            X = self._transform_pca(X)
            X_df = self._to_dataframe(X)
            with bounded_inference(self.model):
                pred = self.model.predict(X_df)
            pred = np.asarray(pred)
            if pred.ndim == 1 or (
                self.args.objective == "regression"
                and pred.ndim == 2
                and pred.shape[1] == 1
            ):
                pred = pred.ravel()
            self.predictions = pred
            self.prediction_probabilities = None
            return self.predictions

        proba = self.predict_proba(X)
        self.prediction_probabilities = proba
        if self.args.objective == "binary":
            self.predictions = (proba[:, 1] >= self.threshold).astype(int)
        else:
            self.predictions = np.argmax(proba, axis=1)
        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        assert self.model is not None, "Model must be fitted before predict_proba."
        assert self.args.objective in ("binary", "classification")

        X = self._transform_pca(X)

        X_df = self._to_dataframe(X)
        with bounded_inference(self.model):
            proba = self.model.predict_proba(X_df)
        return self._normalize_and_align_probabilities(proba)
