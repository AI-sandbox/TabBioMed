"""Benchmark iLTM interface with native preprocessing and constructor API checks."""

from inspect import signature
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_string_dtype
from iltm import iLTMClassifier, iLTMRegressor

from tabbiomed.models.base_model import BaseModel, ModelArgs


class iLTM(BaseModel):
    """iLTM with native preprocessing and its recommended configurations."""

    @classmethod
    def get_base_params(cls, args):
        return cls.get_default_params()

    @classmethod
    def get_supported_parameters(cls, args):
        estimator = (
            iLTMClassifier
            if args.objective in ("binary", "classification")
            else iLTMRegressor
        )
        return super().get_supported_parameters(args) | {
            name
            for name, parameter in signature(estimator).parameters.items()
            if parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }

    @staticmethod
    def _check_api() -> None:
        required = {
            "checkpoint_mix",
            "retrieval_alpha_adaptive",
            "tree_use_external_eval_set",
            "val_subsample_before_preprocessing",
        }
        for model_cls in (iLTMClassifier, iLTMRegressor):
            parameters = signature(model_cls.__init__).parameters
            missing = required.difference(parameters)
            val_checks = parameters.get("val_checks_per_epoch_target")
            if missing or val_checks is None or val_checks.default != "auto":
                raise RuntimeError(
                    "Install the pinned tabbiomed[iltm] dependency. "
                    f"Incompatible {model_cls.__name__} constructor: missing {sorted(missing)}, "
                    "val_checks_per_epoch_target must default to 'auto'."
                )

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        cls._check_api()
        return {
            "checkpoint": "xgbrconcat",
            "n_ensemble": 12,
            "batch_size": 4096,
            "finetuning_dropout": 0.15,
            "finetuning_max_steps": 2048,
            "finetuning_batch_size": 2048,
            "finetuning_lr": 0.0012,
            "gradient_clip_norm": 1.18,
            "finetuning_optimizer": "lion",
            "tree_data_split": "all",
            "tree_n_estimators": 200,
            "tree_lr": 0.55,
            "tree_max_depth": 5,
            "tree_min_samples_leaf": 12,
            "tree_subsample": 0.63,
            "tree_feature_fraction": 0.99,
            "tree_gamma": 0.0,
            "tree_l2_leaf_reg": 1.25,
            "tree_bagging_temperature": None,
            "do_retrieval": True,
            "retrieval_alpha": 0.75,
            "retrieval_temperature": 1.05,
            "retrieval_distance": "cosine",
            "scheduler_min_lr": 6e-07,
            "clip_predictions": True,
            "corr_select_k": 0,
            "finetuning": True,
            "finetuning_data": "entire_dataset",
            "finetuning_val_frac": 0.1,
            "finetuning_classification_val_metric": "auto",
            "initial_transformations_finetuning": True,
            "early_stopping_mode": "auto",
            "patience_epochs": 50,
            "patience_checks": None,
            "val_checks_per_epoch_target": "auto",
            "val_check_interval_batches": None,
            "max_train_batches_per_epoch": 128,
            "finetuning_subset_frac": None,
            "finetuning_subset_max_samples": 100_000,
            "val_max_samples": 25_000,
            "val_subsample_before_preprocessing": True,
            "min_epochs": 0,
            "cooldown_checks": 2,
            "checkpoint_mix": None,
            "tree_for_each_predictor": True,
            "tree_use_external_eval_set": False,
            "retrieval_alpha_finetuning": False,
            "retrieval_temperature_finetuning": False,
            "retrieval_alpha_adaptive": True,
            "clip_data_value": 1_000_000,
            "rf_size": 32_768,
            "pca_sampling": "zeropad",
            "adaptive_memory": True,
            "gpu_mem_very_low_mb": 1024,
            "gpu_oom_retries": 2,
            "auto_disable_retrieval_on_low_mem": False,
            "auto_amp_on_low_mem": False,
            "auto_stop_on_low_cpu_memory": True,
            "cpu_memory_limit_gb": None,
            "cpu_memory_safety_margin_gb": 2.0,
            "inference_storage_dtype": "float16",
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)
        self.model_cls = self._get_model_cls()
        self.device = (
            "cuda:" + str(args.gpu_ids[0])
            if getattr(args, "use_gpu", False) and torch.cuda.is_available()
            else "cpu"
        )
        cat_idx = getattr(args, "cat_idx", None)
        self.cat_features = [] if cat_idx is None else [int(idx) for idx in cat_idx]
        self._constructor_params()

    def _get_model_cls(self):
        """Get the appropriate iLTM model class based on problem type."""
        if self.args.objective in ["binary", "classification"]:
            return iLTMClassifier
        elif self.args.objective in ["regression", "multiregression"]:
            return iLTMRegressor
        else:
            raise ValueError(
                f"Unsupported objective type: {self.args.objective}. Supported types are 'binary', 'classification', 'regression', and 'multiregression'."
            )

    def _constructor_params(self) -> Dict[str, Any]:
        params = dict(self.params)

        if self.args.objective in ("regression", "multiregression"):
            params.pop("finetuning_classification_val_metric", None)
        params["device"] = self.device
        params["cat_features"] = self.cat_features
        if hasattr(self.args, "seed"):
            params["seed"] = self.args.seed

        signature(self.model_cls).bind(**params)
        return params

    @staticmethod
    def _pandas_categorical_indices(X: Any) -> list[int]:
        if not isinstance(X, pd.DataFrame):
            return []

        indices = []
        for idx, col_name in enumerate(X.columns):
            dtype = X[col_name].dtype
            if isinstance(dtype, pd.CategoricalDtype) or (
                dtype != object and is_string_dtype(dtype)
            ):
                indices.append(idx)
        return indices

    @classmethod
    def _prepare_iltm_features(cls, X: Any) -> Any:
        """Make pandas string/category columns visible to iLTM's categorical path."""
        if not isinstance(X, pd.DataFrame):
            return X

        columns_to_object = []
        for col_name in X.columns:
            dtype = X[col_name].dtype
            if isinstance(dtype, pd.CategoricalDtype) or (
                dtype != object and is_string_dtype(dtype)
            ):
                columns_to_object.append(col_name)

        if not columns_to_object:
            return X

        X_prepared = X.copy()
        for col_name in columns_to_object:
            X_prepared[col_name] = X_prepared[col_name].astype(object)
        return X_prepared

    def _cat_features_for_fit(self, X: Any) -> list[int]:
        cat_features = set(self.cat_features)
        cat_features.update(self._pandas_categorical_indices(X))
        return sorted(cat_features)

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        """iLTM handles missing values internally."""
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        """iLTM handles categorical features internally."""
        return False

    @classmethod
    def needs_scaling(cls) -> bool:
        """iLTM handles feature scaling internally."""
        return False

    @classmethod
    def uses_validation_during_fit(cls) -> bool:
        return True

    @classmethod
    def supports_native_multioutput(cls):
        return False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        if self.args.objective in ("binary", "classification"):
            self._classes_ = np.unique(np.asarray(y).ravel())
        params = self._constructor_params()
        params["cat_features"] = self._cat_features_for_fit(X)
        X = self._prepare_iltm_features(X)
        X_val = self._prepare_iltm_features(X_val)
        self.model = self.model_cls(**params)
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model.fit(X, y, eval_set=eval_set)
        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_output_models_", None):
            return super().predict(X)
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if self.args.objective == "regression":
            X_prep = self._prepare_iltm_features(X)
            self.predictions = self.model.predict(X_prep)
        elif self.args.objective == "multiregression":
            self.predictions = super().predict(X)
        elif self.args.objective == "binary":
            self.prediction_probabilities = self.predict_proba(X)
            self.predictions = (
                self.prediction_probabilities[:, 1] >= self.threshold
            ).astype(int)
        elif self.args.objective == "classification":
            self.prediction_probabilities = self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if self.args.objective not in ["binary", "classification"]:
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )

        X = self._prepare_iltm_features(X)

        return self._normalize_and_align_probabilities(self.model.predict_proba(X))
