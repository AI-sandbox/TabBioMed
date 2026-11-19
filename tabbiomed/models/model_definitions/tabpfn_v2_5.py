"""TabPFNv2.5 with native missing-value, categorical and scaling transforms."""

import os

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
from loguru import logger

# The pinned TabPFN imports scikit-learn's Array API test helpers during startup.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
except ImportError:
    raise ImportError(
        "tabpfn library is required. Install with: uv pip install tabpfn==6.4.1"
    )

try:
    from tabpfn_extensions.many_class import ManyClassClassifier
except ImportError:
    ManyClassClassifier = None

try:
    from torch.cuda import is_available as cuda_is_available
except ImportError:
    raise ImportError("torch library is required. Install with: uv pip install torch")

from tabbiomed.models.base_model import BaseModel, ModelArgs


def prediction_batch_size_for_context(
    n_context_rows: int, requested_batch_size: int
) -> int:
    """Bound prediction memory for large contexts without reducing training rows."""
    return (
        min(requested_batch_size, 32)
        if n_context_rows > 10_000
        else requested_batch_size
    )


class TabPFN(BaseModel):
    """TabPFNv2.5 in-context inference from explicitly selected checkpoints.

    Above ten classes, use ManyClassClassifier with alphabet size ten unless
    overridden. The extension controls its estimator count by default."""

    allow_metadata_scaling = False

    @classmethod
    def get_base_params(cls, args):
        return {
            "softmax_temperature": 1.0,
            "balance_probabilities": False,
            "ignore_pretraining_limits": True,
            "n_jobs": "auto",
            "zip_model_path": [
                "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
                "tabpfn-v2.5-regressor-v2.5_default.ckpt",
            ],
            "inference_config/OUTLIER_REMOVAL_STD": None,
            "inference_config/POLYNOMIAL_FEATURES": None,
            "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS": None,
            "preprocessing/scaling": None,
            "preprocessing/categoricals": "numeric",
            "preprocessing/append_original": True,
            "preprocessing/global": None,
            "custom_model_dir": None,
            "inference_batch_size": getattr(args, "val_batch_size", 256),
            "many_class_n_estimators": None,
            "many_class_alphabet_size": 10,
            "many_class_random_state": getattr(args, "seed", 42),
            "apply_pca": False,
            "pca_max_features": 500,
            "pca_fit": "lowrank",
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)

        self.device = (
            (f"cuda:{args.gpu_ids[0]}" if args.gpu_ids else "cuda")
            if args.use_gpu and cuda_is_available()
            else "cpu"
        )

        self.softmax_temperature = params["softmax_temperature"]
        self.balance_probabilities = params["balance_probabilities"]
        self.ignore_pretraining_limits = params["ignore_pretraining_limits"]
        self.n_jobs = params["n_jobs"]
        if self.n_jobs == "auto":
            self.n_jobs = args.num_cpus if hasattr(args, "num_cpus") else 1

        self.zip_model_path = params["zip_model_path"]

        self.outlier_removal_std = params["inference_config/OUTLIER_REMOVAL_STD"]
        self.polynomial_features = params["inference_config/POLYNOMIAL_FEATURES"]
        self.regression_y_preprocess_transforms = params[
            "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS"
        ]

        self.preprocessing_scaling = params["preprocessing/scaling"]
        self.preprocessing_categoricals = params["preprocessing/categoricals"]
        self.preprocessing_append_original = params["preprocessing/append_original"]
        self.preprocessing_global = params["preprocessing/global"]

        self.custom_model_dir = params["custom_model_dir"]

        self.inference_batch_size = params["inference_batch_size"]
        if self.inference_batch_size is None or self.inference_batch_size <= 0:
            self.inference_batch_size = 256
        self._inference_batch_size = self.inference_batch_size

        self._cat_indices = None

        self._many_class_n_estimators = params["many_class_n_estimators"]
        self._many_class_alphabet_size = params["many_class_alphabet_size"]
        self._many_class_random_state = params["many_class_random_state"]

        self.apply_pca = params["apply_pca"]
        self.pca_max_features = params["pca_max_features"]
        self.pca_fit = params["pca_fit"]
        self.pca = None

        self.model = None

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        """Whether the model needs external missing-value imputation."""
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        """Whether the model needs external categorical encoding."""
        return False

    @classmethod
    def needs_scaling(cls) -> bool:
        return False

    @classmethod
    def get_cat_encoding_strategy(cls) -> str:
        """Get the categorical encoding strategy for TabPFN."""
        return "label"

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        """TabPFN does not natively support multi-output predictions"""
        return False

    @classmethod
    def get_default_params(cls):
        """Default v2.5 inference parameters."""

        return {
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "inference_config/OUTLIER_REMOVAL_STD": None,
            "inference_config/POLYNOMIAL_FEATURES": "no",
            "inference_config/REGRESSION_Y_PREPROCESS_TRANSFORMS": None,
            "many_class_alphabet_size": 10,
            "apply_pca": False,
            "pca_max_features": 500,
            "pca_fit": "lowrank",
            "zip_model_path": [
                "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
                "tabpfn-v2.5-regressor-v2.5_default.ckpt",
            ],
        }

    def _as_pca_numeric_array(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        X_arr = np.asarray(X)
        if np.issubdtype(X_arr.dtype, np.number):
            return X_arr.astype(float, copy=False)
        try:
            if fit:
                self._pca_cat_maps_ = {}
            if fit or not getattr(self, "_pca_cat_maps_", None):
                converted = X_arr.astype(float)
                if fit:
                    self._pca_cat_maps_ = {}
                return converted
        except (ValueError, TypeError):
            pass

        X_obj = X_arr.astype(object, copy=False)
        if fit or not hasattr(self, "_pca_cat_maps_"):
            self._pca_cat_maps_ = {}
        cat_set = set(getattr(self.args, "cat_idx", None) or [])
        columns = []
        for col in range(X_obj.shape[1]):
            values = X_obj[:, col]
            numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
                dtype=float
            )
            missing = pd.isna(values)
            has_non_numeric = bool(np.any(~missing & np.isnan(numeric)))
            if col in cat_set or has_non_numeric or col in self._pca_cat_maps_:
                encoded = np.full(values.shape[0], np.nan, dtype=float)
                if fit:
                    observed = values[~missing].astype(str)
                    classes = np.unique(observed)
                    mapping = {str(val): idx for idx, val in enumerate(classes)}
                    self._pca_cat_maps_[col] = mapping
                else:
                    mapping = self._pca_cat_maps_.get(col, {})
                if mapping and np.any(~missing):
                    encoded[~missing] = [
                        mapping.get(str(val), np.nan) for val in values[~missing]
                    ]
                columns.append(encoded)
            else:
                columns.append(numeric)
        return np.column_stack(columns) if columns else np.empty((len(X_obj), 0), dtype=float)

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

        X = self._as_pca_numeric_array(X, fit=True)
        if X_val is not None:
            X_val = self._as_pca_numeric_array(X_val, fit=False)

        device = self.device if isinstance(self.device, str) else "cpu"
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

        X = self._as_pca_numeric_array(X, fit=False)
        return self.pca.transform_array(X)

    def _identify_categorical_indices(self, X: np.ndarray):
        """Preserve raw column indices, PCA components are all numerical."""
        indices = getattr(self.args, "cat_idx", None)
        self._cat_indices = (
            list(indices) if indices is not None and self.pca is None else []
        )

    def _build_inference_config(self) -> Dict[str, Any]:
        """Build the inference_config dictionary from hyperparameters."""
        inference_config = {}

        if self.preprocessing_scaling:
            inference_config["PREPROCESS_TRANSFORMS"] = [
                {
                    "name": scaler,
                    "global_transformer_name": self.preprocessing_global,
                    "categorical_name": self.preprocessing_categoricals,
                    "append_original": self.preprocessing_append_original,
                }
                for scaler in self.preprocessing_scaling
            ]

        if self.outlier_removal_std is not None:
            inference_config["OUTLIER_REMOVAL_STD"] = self.outlier_removal_std

        if self.polynomial_features is not None:
            inference_config["POLYNOMIAL_FEATURES"] = self.polynomial_features

        if (
            self.args.objective == "regression"
            and self.regression_y_preprocess_transforms is not None
        ):
            inference_config["REGRESSION_Y_PREPROCESS_TRANSFORMS"] = (
                self.regression_y_preprocess_transforms
            )

        return inference_config

    def _resolve_model_path(self) -> Optional[Path]:
        from tabbiomed.models.checkpoints import resolve_checkpoint

        classifier, regressor = self.zip_model_path
        filename = (
            classifier
            if self.args.objective in ("binary", "classification")
            else regressor
        )
        if filename is None:
            return None
        return resolve_checkpoint("tabpfn_2_5", filename, self.custom_model_dir)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Load the selected checkpoint and prepare its training context for inference."""
        X, X_val = self._fit_transform_pca(X, X_val)

        X_df = pd.DataFrame(X)
        self._all_missing_columns_ = X_df.isna().all(axis=0).to_numpy()
        self._inference_batch_size = prediction_batch_size_for_context(
            len(X_df),
            self.inference_batch_size,
        )
        if self._inference_batch_size != self.inference_batch_size:
            logger.info(
                f"{self.__class__.__name__}: {len(X_df)} context rows, "
                f"using prediction batches of {self._inference_batch_size}"
            )

        self._identify_categorical_indices(X)

        is_classification = self.args.objective in ["binary", "classification"]
        model_base = TabPFNClassifier if is_classification else TabPFNRegressor

        hps = {
            "device": self.device,
            "n_jobs": self.n_jobs,
            "categorical_features_indices": self._cat_indices
            if self._cat_indices
            else None,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "softmax_temperature": self.softmax_temperature,
        }

        inference_config = self._build_inference_config()
        if inference_config:
            hps["inference_config"] = inference_config

        if is_classification:
            hps["balance_probabilities"] = self.balance_probabilities

        model_path = self._resolve_model_path()
        if model_path is not None:
            hps["model_path"] = model_path

        n_classes = getattr(self.args, "num_classes", None)
        if n_classes is None:
            n_classes = int(len(np.unique(y)))
        use_many_class = (
            is_classification and n_classes > 10 and ManyClassClassifier is not None
        )

        if use_many_class:
            base_clf = TabPFNClassifier(**hps)
            alphabet_size = min(self._many_class_alphabet_size, n_classes)
            many_class_kw = {
                "estimator": base_clf,
                "alphabet_size": alphabet_size,
                "random_state": self._many_class_random_state,
            }
            if self._many_class_n_estimators is not None:
                many_class_kw["n_estimators"] = self._many_class_n_estimators
            self.model = ManyClassClassifier(**many_class_kw)
            n_est_msg = (
                f", n_estimators={self._many_class_n_estimators}"
                if self._many_class_n_estimators is not None
                else ""
            )
            logger.info(
                f"TabPFN: using ManyClassClassifier for {n_classes} classes "
                f"(alphabet_size={alphabet_size}{n_est_msg})"
            )
            self.model.fit(X_df, y)
        else:
            if is_classification and n_classes > 10 and ManyClassClassifier is None:
                logger.warning(
                    f"TabPFN: {n_classes} classes exceeds TabPFN limit (10). "
                    "Install tabpfn_extensions for many-class: uv pip install tabpfn-extensions"
                )

            self.model = model_base(**hps)
            self.model = self.model.fit(X=X_df, y=y)

        return [], []

    def _predict_in_batches(self, X_df: pd.DataFrame, method: str):
        """Run model.predict or model.predict_proba in row batches to avoid CUDA OOM."""
        if self._all_missing_columns_.any():
            X_df = X_df.copy()
            X_df.iloc[:, self._all_missing_columns_] = np.nan
        n = len(X_df)
        batch_size = self._inference_batch_size
        if n <= batch_size:
            out = getattr(self.model, method)(X_df)
            return np.asarray(out) if not isinstance(out, np.ndarray) else out
        chunks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_df.iloc[start:end]
            out = getattr(self.model, method)(X_batch)
            arr = np.asarray(out) if not isinstance(out, np.ndarray) else out
            chunks.append(arr)
        return np.concatenate(chunks, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert (self.model is not None) or (
            hasattr(self, "_output_models_") and len(self._output_models_) > 0
        ), "Model must be fitted before prediction!"

        X_orig = X
        X = self._transform_pca(X)

        X_df = pd.DataFrame(X)

        if self.args.objective == "regression":
            y_pred = self._predict_in_batches(X_df, "predict")
            self.predictions = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
            self.prediction_probabilities = None
        elif self.args.objective == "multiregression":
            self.prediction_probabilities = None
            _predictions = []
            for _model in self._output_models_:
                _X_df = X_df
                if getattr(_model, "pca", None) is not None:
                    _X_df = pd.DataFrame(_model._transform_pca(X_orig))
                _y_pred = _model._predict_in_batches(_X_df, "predict")
                _predictions.append(
                    _y_pred.flatten() if len(_y_pred.shape) > 1 else _y_pred
                )
            self.predictions = np.column_stack(_predictions)
        else:
            proba = self.predict_proba(X_orig)
            self.prediction_probabilities = proba

            if self.args.objective == "binary":
                self.predictions = (
                    self.prediction_probabilities[:, 1] >= self.threshold
                ).astype(int)
            else:
                self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert (self.model is not None) or (
            hasattr(self, "_output_models_") and len(self._output_models_) > 0
        ), "Model must be fitted before prediction!"
        assert self.args.objective in [
            "binary",
            "classification",
        ], "predict_proba only available for classification tasks"

        X = self._transform_pca(X)

        X_df = pd.DataFrame(X)

        proba = self._predict_in_batches(X_df, "predict_proba")

        return self._align_class_probabilities(proba, self.model.classes_)
