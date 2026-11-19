# Preprocessing and model setup adapted from https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/models/realmlp/model.py
"""RealMLP with numerical imputation, native scaling and categorical encoding."""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Union, Optional
from loguru import logger

try:
    from pytabkit import (
        RealMLP_TD_Classifier,
        RealMLP_TD_Regressor,
        RealMLP_TD_S_Classifier,
        RealMLP_TD_S_Regressor,
    )
except ImportError:
    raise ImportError(
        "pytabkit library is required. Install with: uv pip install 'pytabkit>=1.4.0,<2.0'"
    )

from tabbiomed.models.base_model import BaseModel, ModelArgs


class RealMLP(BaseModel):
    """Adapt the official pytabkit RealMLP estimators to the benchmark interface."""

    allow_metadata_scaling = False

    optional_parameters = ("config_type",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "default_hyperparameters": "td",
            "use_native_defaults": False,
            "n_hidden_layers": 3,
            "hidden_sizes": "rectangular",
            "hidden_width": 256,
            "p_drop": 0.1,
            "act": "mish",
            "plr_sigma": 1.0,
            "plr_lr_factor": 0.1,
            "scale_lr_factor": 5.0,
            "first_layer_lr_factor": 0.8,
            "sq_mom": 0.99,
            "plr_hidden_1": 16,
            "plr_hidden_2": 4,
            "lr": 0.1,
            "wd": 0.001,
            "n_epochs": 256,
            "use_early_stopping": False,
            "early_stopping_additive_patience": 40,
            "early_stopping_multiplicative_patience": 3,
            "ls_eps": 0.01,
            "ls_eps_sched": "coslog4",
            "use_ls": "auto",
            "p_drop_sched": "flat_cos",
            "max_one_hot_cat_size": 10,
            "embedding_size": 8,
            "n_ens": 8,
            "ens_av_before_softmax": False,
            "bool_to_cat": True,
            "impute_bool": False,
            "name_categories": True,
            "predict_batch_size": "auto",
        }

    @classmethod
    def configure_parameters(cls, parameters, args, supplied):
        if parameters["use_native_defaults"]:
            model_class = cls._model_class_for_task(
                args, parameters["default_hyperparameters"]
            )
            native = model_class().get_config()
            parameters.update(
                {
                    key: native[key]
                    for key in cls.get_base_params(args)
                    if key in native
                }
            )
            parameters.update(
                first_layer_lr_factor=None,
                ls_eps=native.get("ls_eps"),
                ls_eps_sched=None,
            )
            parameters.update(supplied)
            if (getattr(args, "num_features", 0) or 0) > 8192:
                width = supplied["plr_hidden_2"] if "plr_hidden_2" in supplied else 16
                parameters["plr_hidden_2"] = min(width, 16)
        return parameters

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)
        self.default_hyperparameters = params["default_hyperparameters"]
        self.model_cls = self._get_model_cls()
        self.n_hidden_layers = params["n_hidden_layers"]
        self.hidden_sizes = params["hidden_sizes"]
        self.hidden_width = params["hidden_width"]
        self.p_drop = params["p_drop"]
        self.act = params["act"]

        self.plr_sigma = params["plr_sigma"]
        self.plr_lr_factor = params["plr_lr_factor"]
        self.scale_lr_factor = params["scale_lr_factor"]
        self.first_layer_lr_factor = params["first_layer_lr_factor"]
        self.sq_mom = params["sq_mom"]
        self.plr_hidden_1 = params["plr_hidden_1"]
        self.plr_hidden_2 = params["plr_hidden_2"]

        self.lr = params["lr"]
        self.wd = params["wd"]
        self.n_epochs = params["n_epochs"]
        self.use_early_stopping = params["use_early_stopping"]
        self.early_stopping_additive_patience = params[
            "early_stopping_additive_patience"
        ]
        self.early_stopping_multiplicative_patience = params[
            "early_stopping_multiplicative_patience"
        ]

        self.ls_eps = params["ls_eps"]
        self.ls_eps_sched = params["ls_eps_sched"]
        self.use_ls = params["use_ls"]

        self.p_drop_sched = params["p_drop_sched"]

        self.max_one_hot_cat_size = params["max_one_hot_cat_size"]
        self.embedding_size = params["embedding_size"]

        self.n_ens = params["n_ens"]
        self.ens_av_before_softmax = params["ens_av_before_softmax"]

        self.bool_to_cat = params["bool_to_cat"]
        self.impute_bool = params["impute_bool"]
        self.name_categories = params["name_categories"]

        predict_batch_size = params["predict_batch_size"]
        if predict_batch_size == "auto":
            n_features = args.num_features if hasattr(args, "num_features") else 100
            self.predict_batch_size = max(min(int(8192 * 200 / n_features), 8192), 64)
        else:
            self.predict_batch_size = predict_batch_size

        metric_map = {
            "roc_auc": "1-auc_ovr_alt",
            "accuracy": "class_error",
            "balanced_accuracy": "1-balanced_accuracy",
            "log_loss": "cross_entropy",
            "rmse": "rmse",
            "root_mean_squared_error": "rmse",
            "r2": "rmse",
            "mae": "mae",
            "mean_average_error": "mae",
        }

        objective_to_metric = {
            "regression": "rmse",
            "multiregression": "rmse",
            "binary": "roc_auc",
            "classification": "log_loss",
        }
        default_metric = objective_to_metric.get(args.objective, "rmse")
        self.val_metric_name = metric_map.get(default_metric, None)

        if (
            self.use_ls is not None
            and isinstance(self.use_ls, str)
            and self.use_ls == "auto"
        ):
            if self.val_metric_name is None or self.val_metric_name in [
                "cross_entropy",
                "1-auc_ovr_alt",
            ]:
                self.use_ls = False
            else:
                self.use_ls = None

        self.device = "cpu"
        self.trainer_kwargs = {}
        if getattr(args, "use_gpu", False) and torch.cuda.is_available():
            self.device = (
                f"cuda:{args.gpu_ids[0]}"
                if hasattr(args, "gpu_ids") and args.gpu_ids
                else "cuda:0"
            )

            self.trainer_kwargs = {"accelerator": "gpu", "devices": 1}

        logger.debug(
            f"Using device: {self.device}, trainer_kwargs={self.trainer_kwargs}"
        )

        self._imputer = None
        self._features_to_impute = None
        self._features_to_keep = None
        self._indicator_columns = None
        self._features_bool = None
        self._bool_columns = None
        self._binary_col_indices = None
        self._imputed_col_indices = None
        self._cat_col_names_list = None
        self._should_impute = False
        self._X_num_train_reference = None
        self._num_features_before_impute = 0

        self.model = None

    @classmethod
    def _model_class_for_task(cls, args, preset):
        if preset not in ("td", "td_s"):
            raise ValueError(f"Unknown RealMLP preset: {preset}")
        if args.objective in ("binary", "classification"):
            return RealMLP_TD_Classifier if preset == "td" else RealMLP_TD_S_Classifier
        return RealMLP_TD_Regressor if preset == "td" else RealMLP_TD_S_Regressor

    def _get_model_cls(self):
        return self._model_class_for_task(self.args, self.default_hyperparameters)

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
        return False

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Use the native task-specific TD preset with one network."""
        return {
            "use_native_defaults": True,
            "n_ens": 1,
            "use_early_stopping": False,
            "early_stopping_additive_patience": 40,
            "early_stopping_multiplicative_patience": 3,
            "use_ls": "auto",
            "default_hyperparameters": "td",
            "bool_to_cat": True,
            "impute_bool": False,
            "name_categories": True,
            "predict_batch_size": "auto",
        }

    def _has_nan_in_impute_features(self, X: np.ndarray) -> bool:
        if X.shape[0] == 0 or not self._features_to_impute:
            return False
        return bool(np.isnan(X[:, self._features_to_impute].astype(float)).any())

    def _should_run_imputation(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> bool:
        """Match TabularDataset.impute_missing_values gate: any split with NaNs triggers imputation."""
        if self._has_nan_in_impute_features(X_train):
            return True
        if X_val is not None and self._has_nan_in_impute_features(X_val):
            return True
        if X_test is not None and self._has_nan_in_impute_features(X_test):
            return True
        return False

    def _fit_imputer_on_train_reference(self) -> None:
        """Fit imputer on training numerical columns only (never on val/test)."""
        X_num = self._X_num_train_reference
        self._imputer = SimpleImputer(
            strategy="mean", add_indicator=True, keep_empty_features=True
        )
        X_imputed = self._imputer.fit_transform(X_num)
        self._num_features_before_impute = X_num.shape[1]
        num_indicators = X_imputed.shape[1] - self._num_features_before_impute
        self._indicator_columns = [
            f"missing_indicator_{i}" for i in range(num_indicators)
        ]

    def _transform_imputed_numerical_block(
        self, X_num: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        assert self._imputer is not None
        X_imputed = self._imputer.transform(X_num)
        n = self._num_features_before_impute
        X_num_imputed = X_imputed[:, :n]
        X_indicators = X_imputed[:, n:] if X_imputed.shape[1] > n else None
        return X_num_imputed, X_indicators

    def _ensure_imputer_for_inference(self, X: np.ndarray) -> None:
        """Fit on stored train reference when test (or val) has NaNs but train did not at fit time."""
        if self._imputer is not None or not self._features_to_impute:
            return
        if not self._has_nan_in_impute_features(X):
            return
        if self._X_num_train_reference is None:
            return
        self._should_impute = True
        self._fit_imputer_on_train_reference()

    def _preprocess_data(
        self,
        X: np.ndarray,
        is_train: bool = True,
        X_val_for_gate: Optional[np.ndarray] = None,
        X_test_for_gate: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, Optional[list]]:
        """
        Preprocess data for RealMLP:
        1. Work with numpy array first (like other models)
        2. Identify categorical columns from cat_idx
        3. Identify binary columns (0/1 values)
        4. Impute numerical columns (excluding categorical and binary unless impute_bool):
           gate on NaNs in train/val/test, always fit imputer on train only
        5. Convert to DataFrame
        6. Mark categorical columns (by name) as category dtype
        7. Mark binary columns as category dtype if bool_to_cat=True
        8. Ensure all numerical columns are float with no NaNs
        """

        cat_col_indices = (
            set(self.args.cat_idx)
            if hasattr(self.args, "cat_idx") and self.args.cat_idx
            else set()
        )
        cat_col_names_list = (
            [f"feature_{i}" for i in sorted(cat_col_indices)] if cat_col_indices else []
        )

        if is_train:
            self._cat_col_names_list = cat_col_names_list

            binary_col_indices = []
            for col_idx in range(X.shape[1]):
                if col_idx in cat_col_indices:
                    continue
                try:
                    col_data = X[:, col_idx]

                    try:
                        col_data_float = col_data.astype(float)
                    except (ValueError, TypeError):
                        continue

                    unique_vals = np.unique(col_data_float[~np.isnan(col_data_float)])

                    if len(unique_vals) <= 2:
                        is_binary = all(
                            np.isclose(val, 0, atol=1e-6)
                            or np.isclose(val, 1, atol=1e-6)
                            for val in unique_vals
                        )
                        if is_binary:
                            binary_col_indices.append(col_idx)
                except (TypeError, ValueError):
                    continue

            self._binary_col_indices = set(binary_col_indices)
            self._bool_columns = (
                [f"feature_{i}" for i in binary_col_indices] if self.bool_to_cat else []
            )

            cols_to_exclude = cat_col_indices.copy()
            if not self.impute_bool:
                cols_to_exclude.update(binary_col_indices)

            self._features_to_impute = [
                i for i in range(X.shape[1]) if i not in cols_to_exclude
            ]
            self._features_to_keep = list(cols_to_exclude)
            self._imputed_col_indices = self._features_to_impute.copy()

            if len(self._features_to_impute) > 0:
                self._X_num_train_reference = (
                    X[:, self._features_to_impute].astype(float).copy()
                )
                self._should_impute = self._should_run_imputation(
                    X, X_val_for_gate, X_test_for_gate
                )
                if self._should_impute:
                    self._fit_imputer_on_train_reference()
                    (
                        self._X_num_imputed,
                        self._X_indicators,
                    ) = self._transform_imputed_numerical_block(
                        self._X_num_train_reference
                    )
                else:
                    self._imputer = None
                    self._indicator_columns = []
                    self._num_features_before_impute = (
                        self._X_num_train_reference.shape[1]
                    )
                    self._X_num_imputed = self._X_num_train_reference
                    self._X_indicators = None
            else:
                self._X_num_train_reference = None
                self._should_impute = False
                self._imputer = None
                self._X_num_imputed = None
                self._X_indicators = None
                self._indicator_columns = []
                self._imputed_col_indices = []

            final_arrays = []
            final_col_names = []

            if cat_col_indices:
                X_cat = X[:, sorted(cat_col_indices)]
                final_arrays.append(X_cat)
                final_col_names.extend(
                    [f"feature_{i}" for i in sorted(cat_col_indices)]
                )

            if not self.impute_bool and binary_col_indices:
                X_binary = X[:, sorted(binary_col_indices)].astype(float)
                final_arrays.append(X_binary)
                final_col_names.extend(
                    [f"feature_{i}" for i in sorted(binary_col_indices)]
                )

            if self._X_num_imputed is not None:
                final_arrays.append(self._X_num_imputed)
                final_col_names.extend(
                    [f"feature_{i}" for i in self._imputed_col_indices]
                )

            if self._X_indicators is not None:
                final_arrays.append(self._X_indicators)
                final_col_names.extend(self._indicator_columns)

            if final_arrays:
                X_final = np.hstack(final_arrays)
            else:
                X_final = X

            X_df = pd.DataFrame(X_final, columns=final_col_names)

            for col_name in cat_col_names_list:
                if col_name in X_df.columns:
                    X_df[col_name] = X_df[col_name].astype("category")

            if self.bool_to_cat and self._bool_columns:
                for col_name in self._bool_columns:
                    if col_name in X_df.columns:
                        X_df[col_name] = X_df[col_name].astype("category")

            if self._indicator_columns:
                for col_name in self._indicator_columns:
                    if col_name in X_df.columns:
                        X_df[col_name] = X_df[col_name].astype("category")

            for col_name in X_df.columns:
                if (
                    col_name in cat_col_names_list
                    or col_name in (self._bool_columns or [])
                    or col_name in (self._indicator_columns or [])
                ):
                    continue
                else:
                    if (
                        X_df[col_name].dtype != "float64"
                        and X_df[col_name].dtype != "float32"
                    ):
                        X_df[col_name] = pd.to_numeric(
                            X_df[col_name], errors="coerce"
                        ).astype("float64")

            if self.name_categories:
                cat_col_names = X_df.select_dtypes(include="category").columns.tolist()
            else:
                cat_col_names = None
        else:
            if not hasattr(self, "_binary_col_indices"):
                self._binary_col_indices = set()
            if not hasattr(self, "_cat_col_names_list"):
                self._cat_col_names_list = []

            cols_to_exclude = cat_col_indices.copy()
            if not self.impute_bool:
                cols_to_exclude.update(self._binary_col_indices)

            self._features_to_impute = [
                i for i in range(X.shape[1]) if i not in cols_to_exclude
            ]
            self._features_to_keep = list(cols_to_exclude)

            if len(self._features_to_impute) > 0:
                X_num = X[:, self._features_to_impute].astype(float)
                self._ensure_imputer_for_inference(X)
                if self._imputer is not None:
                    (
                        self._X_num_imputed,
                        self._X_indicators,
                    ) = self._transform_imputed_numerical_block(X_num)
                else:
                    self._X_num_imputed = X_num
                    self._X_indicators = None
            else:
                self._X_num_imputed = None
                self._X_indicators = None

            final_arrays = []
            final_col_names = []

            if cat_col_indices:
                X_cat = X[:, sorted(cat_col_indices)]
                final_arrays.append(X_cat)
                final_col_names.extend(
                    [f"feature_{i}" for i in sorted(cat_col_indices)]
                )

            if not self.impute_bool and self._binary_col_indices:
                X_binary = X[:, sorted(self._binary_col_indices)].astype(float)
                final_arrays.append(X_binary)
                final_col_names.extend(
                    [f"feature_{i}" for i in sorted(self._binary_col_indices)]
                )

            if self._X_num_imputed is not None:
                final_arrays.append(self._X_num_imputed)
                final_col_names.extend(
                    [f"feature_{i}" for i in self._imputed_col_indices]
                )

            if self._X_indicators is not None:
                final_arrays.append(self._X_indicators)
                final_col_names.extend(self._indicator_columns)

            if final_arrays:
                X_final = np.hstack(final_arrays)
            else:
                X_final = X

            X_df = pd.DataFrame(X_final, columns=final_col_names)

            for col_name in self._cat_col_names_list:
                if col_name in X_df.columns:
                    X_df[col_name] = X_df[col_name].astype("category")

            if self.bool_to_cat and self._bool_columns:
                for col_name in self._bool_columns:
                    if col_name in X_df.columns:
                        X_df[col_name] = X_df[col_name].astype("category")

            if self._indicator_columns:
                for col_name in self._indicator_columns:
                    if col_name in X_df.columns:
                        X_df[col_name] = X_df[col_name].astype("category")

            for col_name in X_df.columns:
                if (
                    col_name in self._cat_col_names_list
                    or col_name in (self._bool_columns or [])
                    or col_name in (self._indicator_columns or [])
                ):
                    continue
                else:
                    if (
                        X_df[col_name].dtype != "float64"
                        and X_df[col_name].dtype != "float32"
                    ):
                        X_df[col_name] = pd.to_numeric(
                            X_df[col_name], errors="coerce"
                        ).astype("float64")

            cat_col_names = None

        return X_df, cat_col_names

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
        X_test_for_gate: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit RealMLP with validation-based early stopping.

        X_test_for_gate only identifies whether missing-value imputation is needed,
        imputation statistics are fitted on training rows."""
        X_train_df, cat_col_names = self._preprocess_data(
            X,
            is_train=True,
            X_val_for_gate=X_val,
            X_test_for_gate=X_test_for_gate,
        )

        if X_val is not None:
            X_val_df, _ = self._preprocess_data(X_val, is_train=False)

        else:
            X_val_df = None

            self.use_early_stopping = False

        if self.args.objective in ["classification", "binary"]:
            y_train = np.asarray(y).flatten().astype(np.int64)

        else:
            y_train = np.asarray(y).flatten()

        if X_val_df is not None and y_val is not None:
            if self.args.objective in ["classification", "binary"]:
                y_val_series = np.asarray(y_val).flatten().astype(np.int64)

            else:
                y_val_series = np.asarray(y_val).flatten()
        else:
            y_val_series = None

        hyperparameters = {
            "n_hidden_layers": self.n_hidden_layers,
            "hidden_sizes": self.hidden_sizes,
            "hidden_width": self.hidden_width,
            "p_drop": self.p_drop,
            "act": self.act,
            "plr_sigma": self.plr_sigma,
            "plr_lr_factor": self.plr_lr_factor,
            "scale_lr_factor": self.scale_lr_factor,
            "first_layer_lr_factor": self.first_layer_lr_factor,
            "sq_mom": self.sq_mom,
            "plr_hidden_1": self.plr_hidden_1,
            "plr_hidden_2": self.plr_hidden_2,
            "lr": self.lr,
            "wd": self.wd,
            "n_epochs": self.n_epochs,
            "use_early_stopping": self.use_early_stopping,
            "early_stopping_additive_patience": self.early_stopping_additive_patience,
            "early_stopping_multiplicative_patience": self.early_stopping_multiplicative_patience,
            "ls_eps": self.ls_eps,
            "ls_eps_sched": self.ls_eps_sched,
            "use_ls": self.use_ls,
            "p_drop_sched": self.p_drop_sched,
            "max_one_hot_cat_size": int(self.max_one_hot_cat_size),
            "embedding_size": self.embedding_size,
            "n_ens": self.n_ens,
            "ens_av_before_softmax": self.ens_av_before_softmax,
            "predict_batch_size": self.predict_batch_size,
            "random_state": self.args.seed if hasattr(self.args, "seed") else 0,
        }

        init_kwargs = {}
        if self.val_metric_name is not None:
            init_kwargs["val_metric_name"] = self.val_metric_name

        n_threads = getattr(self.args, "num_cpus", None)
        if n_threads is None:
            allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if allocated_cpus:
                n_threads = int(allocated_cpus)
            else:
                n_threads = (os.cpu_count() or 1) if self.device == "cpu" else 1
        logger.info(f"RealMLP: using {n_threads} CPU threads with device {self.device}")

        self.model = self.model_cls(
            n_threads=n_threads,
            device=self.device,
            **init_kwargs,
            **hyperparameters,
        )

        fit_kwargs = {}
        if self.name_categories and cat_col_names:
            fit_kwargs["cat_col_names"] = cat_col_names

        if hasattr(self.args, "time_limit") and self.args.time_limit is not None:
            fit_kwargs["time_to_fit_in_seconds"] = self.args.time_limit

        if self.args.objective in ["binary", "classification"]:
            self._classes_ = np.unique(np.asarray(y).ravel())

        self.model = self.model.fit(
            X=X_train_df,
            y=y_train,
            X_val=X_val_df,
            y_val=y_val_series,
            **fit_kwargs,
        )

        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "_output_models_") and self._output_models_:
            return super().predict(X)

        assert self.model is not None, "Model must be fitted before prediction!"

        if self.args.objective in ["regression", "multiregression"]:
            X_test_df, _ = self._preprocess_data(X, is_train=False)
            y_pred = self.model.predict(X_test_df)
            if isinstance(y_pred, pd.Series):
                y_pred = y_pred.values
            self.predictions = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
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

        X_test_df, _ = self._preprocess_data(X, is_train=False)

        proba = self.model.predict_proba(X_test_df)

        if isinstance(proba, pd.DataFrame):
            proba = proba.values
        elif not isinstance(proba, np.ndarray):
            proba = np.array(proba)

        return self._normalize_and_align_probabilities(proba)
