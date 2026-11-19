"""xRFM with mean imputation, missingness indicators, one-hot encoding and scaling."""

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Union
from loguru import logger

try:
    import nvidia.cuda_runtime as cuda_runtime
except ImportError:
    cuda_runtime = None
if cuda_runtime is not None:
    os.environ["CUDA_HOME"] = str(Path(next(iter(cuda_runtime.__path__))).resolve())

try:
    import xrfm
except ImportError:
    raise ImportError("xrfm library is required. Install with: uv pip install xrfm")

from tabbiomed.models.base_model import BaseModel, ModelArgs


class xRFM(BaseModel):
    """
    xRFM: Recursive Feature Machines with tree-based data splitting.
    """

    optional_parameters = ("split_M_batch_size",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "bandwidth": 10.0,
            "kernel": "l2",
            "exponent": 1.0,
            "p_interp": 1.0,
            "reg": 0.001,
            "diag": True,
            "bandwidth_mode": "constant",
            "standardize_cats": False,
            "early_stop_multiplier": 1.1,
            "max_cardinality_for_one_hot": 100,
            "M_batch_size": 5000,
            "fast_categorical": True,
            "iters": 5,
            "early_stop_rfm": True,
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)

        self.bandwidth = params["bandwidth"]
        self.kernel = params["kernel"]
        self.exponent = params["exponent"]
        self.p_interp = params["p_interp"]
        self.reg = params["reg"]
        self.diag = params["diag"]
        self.bandwidth_mode = params["bandwidth_mode"]
        self.standardize_cats = params["standardize_cats"]
        self.early_stop_multiplier = params["early_stop_multiplier"]
        self.max_cardinality_for_one_hot = params["max_cardinality_for_one_hot"]

        self.classification_mode = (
            "prevalence" if args.objective in ["binary", "classification"] else None
        )

        metric_mapping = {
            "log_loss": "logloss",
            "roc_auc": "auc",
            "f1": "f1",
            "accuracy": "accuracy",
            "rmse": "mse",
            "mse": "mse",
            "mae": "mae",
            "r2": "r2",
        }

        if hasattr(args, "optimize_optuna_with") and args.optimize_optuna_with:
            if isinstance(args.optimize_optuna_with, dict):
                optuna_metric = args.optimize_optuna_with.get(args.objective, None)

                self.tuning_metric = metric_mapping.get(optuna_metric, optuna_metric)
            else:
                self.tuning_metric = None
        else:
            self.tuning_metric = None

        if self.tuning_metric:
            logger.info(f"xRFM: Using tuning metric '{self.tuning_metric}' from config")
        else:
            logger.info(f"xRFM: No tuning metric specified, using xRFM default")

        self.device = torch.device(
            "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
        )

        batch_cap = (
            256
            if (
                self.device.type == "cuda"
                and getattr(args, "dataset", None) == "subpop_50k"
            )
            else 5000
        )
        batch_size = min(params["M_batch_size"], batch_cap)

        self.rfm_params = {
            "model": {
                "kernel": self.kernel,
                "bandwidth": self.bandwidth,
                "exponent": self.exponent,
                "norm_p": self.exponent + (2 - self.exponent) * self.p_interp,
                "diag": self.diag,
                "bandwidth_mode": self.bandwidth_mode,
                "fast_categorical": params["fast_categorical"],
            },
            "fit": {
                "reg": self.reg,
                "iters": params["iters"],
                "M_batch_size": batch_size,
                "verbose": False,
                "early_stop_rfm": params["early_stop_rfm"],
                "early_stop_multiplier": self.early_stop_multiplier,
                "solver": "solve",
            },
        }
        self.ohe_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.scaler_combined_ = None
        self.categorical_info_ = None
        self.cat_cols_ = None
        self.num_cols_ = None

        self.mean_ = None
        self.std_ = None

        self.model = None

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
    def uses_validation_during_fit(cls) -> bool:
        return True

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        """xRFM does not natively support multi-output prediction."""
        return False

    @classmethod
    def get_default_params(cls):
        return {
            "bandwidth": 10.0,
            "kernel": "l2",
            "exponent": 1.0,
            "p_interp": 1.0,
            "reg": 1e-3,
            "diag": True,
            "bandwidth_mode": "constant",
            "standardize_cats": False,
            "early_stop_multiplier": 1.1,
        }

    # Preprocessing adapted from https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/models/xrfm/model.py
    def _preprocess_data(
        self, X: np.ndarray, y: np.ndarray = None, is_train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess data for xRFM:
        1. Separate categorical and numerical features
        2. Mean-impute numerical values and one-hot encode categoricals/indicators
        3. Scale numericals (and optionally categoricals)
        4. Build categorical info structure
        5. Convert to PyTorch tensors
        """

        if is_train:
            if self.args.cat_idx is not None and len(self.args.cat_idx) > 0:
                self.cat_cols_ = list(self.args.cat_idx)
                self.num_cols_ = [
                    i for i in range(X.shape[1]) if i not in self.args.cat_idx
                ]
            else:
                self.cat_cols_ = []
                self.num_cols_ = list(range(X.shape[1]))

        if self.cat_cols_:
            X_cat = X[:, self.cat_cols_]
            X_num = (
                X[:, self.num_cols_] if self.num_cols_ else np.empty((X.shape[0], 0))
            )
        else:
            X_cat = np.empty((X.shape[0], 0))
            X_num = X

        X_num = np.asarray(X_num, dtype=float)
        if is_train:
            self.imputer_ = None
            self.categorical_info_ = None
            if X_num.shape[1]:
                self.imputer_ = SimpleImputer(strategy="mean", add_indicator=True)
                self.imputer_.fit(X_num)
        if self.imputer_ is not None:
            X_imputed = self.imputer_.transform(X_num)
            n_indicators = len(self.imputer_.indicator_.features_)
            n_numeric = X_imputed.shape[1] - n_indicators
            X_num = X_imputed[:, :n_numeric]
            if n_indicators:
                X_cat = np.concatenate([X_cat, X_imputed[:, n_numeric:]], axis=1)

        if is_train and X_cat.shape[1] > 0:
            try:
                self.ohe_ = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop=None
                )
            except TypeError:
                self.ohe_ = OneHotEncoder(
                    handle_unknown="ignore", sparse=False, drop=None
                )
            X_cat_enc = self.ohe_.fit_transform(X_cat)
        elif not is_train and self.ohe_ is not None:
            X_cat_enc = self.ohe_.transform(X_cat)
        else:
            if is_train:
                self.ohe_ = None
            X_cat_enc = np.empty((X.shape[0], 0))

        if X_num.shape[1] > 0:
            if is_train:
                self.scaler_ = StandardScaler()
                X_num_scaled = self.scaler_.fit_transform(X_num)
            else:
                X_num_scaled = self.scaler_.transform(X_num)
        else:
            X_num_scaled = X_num
            if is_train:
                self.scaler_ = None

        if self.standardize_cats:
            X_combined = np.concatenate([X_cat_enc, X_num_scaled], axis=1)
            if is_train:
                self.scaler_combined_ = StandardScaler()
                X_final = self.scaler_combined_.fit_transform(X_combined)
            else:
                if (
                    hasattr(self, "scaler_combined_")
                    and self.scaler_combined_ is not None
                ):
                    X_final = self.scaler_combined_.transform(X_combined)
                else:
                    X_final = X_combined
        else:
            X_final = np.concatenate([X_cat_enc, X_num_scaled], axis=1).astype(float)

        if is_train and self.ohe_ is not None:
            categorical_indices = []
            categorical_vectors = []
            numerical_indices_parts = []

            idx = 0
            dev = self.device
            for cat_categories in self.ohe_.categories_:
                cat_len = len(cat_categories)
                cat_idxs = torch.arange(
                    idx, idx + cat_len, dtype=torch.long, device=dev
                )

                if cat_len > self.max_cardinality_for_one_hot:
                    categorical_indices.append(cat_idxs)
                    if (
                        self.standardize_cats
                        and hasattr(self, "scaler_combined_")
                        and self.scaler_combined_ is not None
                    ):
                        mean = self.scaler_combined_.mean_[idx : idx + cat_len]
                        scale = self.scaler_combined_.scale_[idx : idx + cat_len]
                        cat_vectors = (np.eye(cat_len) - mean) / scale
                        categorical_vectors.append(
                            torch.tensor(cat_vectors, dtype=torch.float32, device=dev)
                        )
                    else:
                        categorical_vectors.append(
                            torch.eye(cat_len, dtype=torch.float32, device=dev)
                        )
                else:
                    numerical_indices_parts.append(cat_idxs)

                idx += cat_len

            numerical_block_start = idx
            if X_num_scaled.shape[1] > 0:
                numerical_block = torch.arange(
                    numerical_block_start,
                    numerical_block_start + X_num_scaled.shape[1],
                    dtype=torch.long,
                    device=dev,
                )
            else:
                numerical_block = torch.tensor([], dtype=torch.long, device=dev)

            if len(numerical_indices_parts) > 0:
                numerical_indices = torch.cat(
                    [*numerical_indices_parts, numerical_block]
                )
            else:
                numerical_indices = numerical_block

            if len(categorical_indices) > 0:
                self.categorical_info_ = {
                    "categorical_indices": categorical_indices,
                    "categorical_vectors": categorical_vectors,
                    "numerical_indices": numerical_indices,
                }
            else:
                self.categorical_info_ = None
        elif not is_train:
            pass

        X_tensor = torch.from_numpy(X_final).float().to(self.device)

        if y is not None:
            y_processed = y.copy()
            if self.args.objective in ["regression", "multiregression"]:
                if is_train:
                    self.mean_ = np.mean(y_processed, axis=0)
                    self.std_ = np.std(y_processed, axis=0) + 1e-30
                y_processed = (y_processed - self.mean_) / self.std_

                if len(y_processed.shape) == 1:
                    y_processed = y_processed.reshape(-1, 1)
                y_tensor = torch.from_numpy(y_processed).float().to(self.device)
            else:
                y_processed = y_processed.astype(int)

                if len(y_processed.shape) == 1:
                    y_processed = y_processed.reshape(-1, 1)
                y_tensor = torch.from_numpy(y_processed).long().to(self.device)
        else:
            y_tensor = None

        return X_tensor, y_tensor, self.categorical_info_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit xRFM with validation-based early stopping."""

        X_train_tensor, y_train_tensor, categorical_info = self._preprocess_data(
            X, y, is_train=True
        )

        if X_val is not None and y_val is not None:
            X_val_tensor, y_val_tensor, _ = self._preprocess_data(
                X_val, y_val, is_train=False
            )
        else:
            X_val_tensor = None
            y_val_tensor = None

        labels = None
        if self.args.objective in ["binary", "classification"]:
            unique_labels = np.unique(y)
            self._classes_ = [int(label) for label in sorted(unique_labels.tolist())]
            if self.args.objective == "binary":
                labels = [0, 1]
            else:
                labels = list(self._classes_)
                logger.info(
                    f"xRFM: Extracted labels={labels} from y (before preprocessing)"
                )

        model_kwargs = {
            "categorical_info": categorical_info,
            "device": str(self.device),
            "classification_mode": self.classification_mode,
            "rfm_params": self.rfm_params,
        }
        if self.device.type == "cuda":
            model_kwargs["n_threads"] = 1

        if self.tuning_metric is not None:
            model_kwargs["tuning_metric"] = self.tuning_metric

        if labels is not None:
            model_kwargs["labels"] = labels
            logger.info(
                f"xRFM: Passing labels={labels} for classification_mode={self.classification_mode}"
            )

        if hasattr(self.args, "time_limit") and self.args.time_limit is not None:
            model_kwargs["time_limit_s"] = self.args.time_limit

        self.model = xrfm.xRFM(**model_kwargs)
        if "split_M_batch_size" in self.params:
            self.model.default_rfm_params["fit"]["M_batch_size"] = self.params[
                "split_M_batch_size"
            ]

        fit_kwargs = {
            "X": X_train_tensor,
            "y": y_train_tensor,
        }

        if X_val_tensor is not None and y_val_tensor is not None:
            fit_kwargs["X_val"] = X_val_tensor
            fit_kwargs["y_val"] = y_val_tensor

        if self.classification_mode == "prevalence" and labels is not None:
            fit_kwargs["labels"] = labels
            logger.info(
                f"xRFM: Also passing labels={labels} to fit() for prevalence mode"
            )

        self.model.fit(**fit_kwargs)

        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "_output_models_", None):
            return super().predict(X)

        assert self.model is not None, "Model must be fitted before prediction!"

        if self.args.objective in ("regression", "multiregression"):
            X_tensor, _, _ = self._preprocess_data(X, y=None, is_train=False)
            y_pred = self.model.predict(X_tensor)

            if isinstance(y_pred, torch.Tensor):
                y_pred = np.squeeze(y_pred.cpu().numpy(), axis=1)
            else:
                y_pred = np.squeeze(np.asarray(y_pred), axis=1)

            if self.mean_ is not None and self.std_ is not None:
                y_pred = y_pred * self.std_ + self.mean_

            self.predictions = y_pred
            self.prediction_probabilities = None
        else:
            self.prediction_probabilities = self.predict_proba(X)

            if self.args.objective == "binary":
                self.predictions = (
                    self.prediction_probabilities[:, 1] >= self.threshold
                ).astype(int)
            else:
                self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model must be fitted before prediction!"
        assert self.args.objective in [
            "binary",
            "classification",
        ], "predict_proba only available for classification tasks"

        X_tensor, _, _ = self._preprocess_data(X, y=None, is_train=False)

        proba = self.model.predict_proba(X_tensor)

        if isinstance(proba, torch.Tensor):
            proba = proba.cpu().numpy()
        else:
            proba = np.asarray(proba)

        return self._normalize_and_align_probabilities(proba)
