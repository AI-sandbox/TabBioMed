# code adapted from https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/basemodel.py
import copy
import os
import numpy as np
import torch

from loguru import logger
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, Any
from joblib import Parallel, delayed


def _fit_single_output(model_class, params, args, X, y_i, X_val, y_val_i):
    """Instantiate a fresh model and fit it on a single output column."""
    model = model_class(params, args)
    model.fit(X, y_i, X_val, y_val_i)
    return model


class ModelArgs:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.num_cpus = int(
            settings.get("num_cpus", os.environ.get("SLURM_CPUS_PER_TASK", 10))
        )
        self.objective = settings["objective"]
        self.dataset = settings["dataset"]
        self.is_cv = settings.get("is_cv", False)
        self.model = settings["model"]
        self.use_gpu = settings["use_gpu"]
        self.epochs = settings["epochs"]
        self.early_stopping_rounds = settings["early_stopping_rounds"]
        self.logging_period = settings["logging_period"]
        self.gpu_ids = settings["gpu_ids"]
        self.cat_idx = settings["cat_idx"]
        self.num_classes = settings["num_classes"]
        self.num_features = settings["num_features"]
        self.num_outputs = settings.get("num_outputs", 1)
        self.data_parallel = settings["data_parallel"]
        self.batch_size = settings["batch_size"]

        self.val_batch_size = settings.get("val_batch_size", settings["batch_size"])
        self.seed = settings.get("seed", 42)
        self.optimize_optuna_with = settings.get("optimize_optuna_with", None)


class BaseModel(ABC):
    """Shared estimator interface for preprocessing, fitting and reporting."""

    required_parameters = ()
    optional_parameters = ()
    allow_metadata_scaling = False

    @classmethod
    def get_base_params(cls, args):
        """Base wrapper settings overridden by Default or sampled configurations."""
        return {}

    @classmethod
    def get_supported_parameters(cls, args):
        return (
            set(cls.required_parameters)
            | set(cls.optional_parameters)
            | set(cls.get_base_params(args))
            | set(cls.get_default_params())
            | {"threshold", "scaler"}
        )

    @classmethod
    def validate_parameter_names(cls, parameters, args):
        unknown = parameters.keys() - cls.get_supported_parameters(args)
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unrecognized parameters {sorted(unknown)}"
            )

    @classmethod
    def configure_parameters(cls, parameters, args, supplied):
        return parameters

    @classmethod
    def resolve_parameters(cls, parameters, args):
        """Build the complete wrapper configuration before construction."""
        cls.validate_parameter_names(parameters, args)
        supplied = copy.deepcopy(parameters)
        resolved = {
            "threshold": (
                0.5 if args.objective in ("binary", "classification") else None
            ),
            "scaler": "standard",
            **cls.get_base_params(args),
            **supplied,
        }
        resolved = cls.configure_parameters(resolved, args, supplied)
        cls.validate_parameters(resolved, args)
        return resolved

    @classmethod
    def validate_parameters(cls, parameters, args):
        cls.validate_parameter_names(parameters, args)
        required = (
            set(cls.required_parameters)
            | set(cls.get_base_params(args))
            | {"threshold", "scaler"}
        )
        missing = required - parameters.keys()
        if missing:
            raise ValueError(
                f"{cls.__name__}: missing resolved parameters {sorted(missing)}"
            )

    def __init__(self, params: Dict, args):
        """Accept only a resolved configuration."""
        self.validate_parameters(params, args)
        self.args = args
        self.input_parameters = copy.deepcopy(params)
        self.params = {
            key: value
            for key, value in params.items()
            if key not in ("threshold", "scaler")
        }

        self.model = None

        self.predictions = None
        self.prediction_probabilities = None

        if args.objective in ["classification", "binary"]:
            self.threshold = params["threshold"]
        else:
            self.threshold = None

    @classmethod
    @abstractmethod
    def needs_missing_value_imputation(cls) -> bool:
        """Whether the model needs missing value imputation."""
        pass

    @classmethod
    @abstractmethod
    def needs_cat_encoding(cls) -> bool:
        """Whether the model needs categorical encoding."""
        pass

    @classmethod
    @abstractmethod
    def needs_scaling(cls) -> bool:
        """Whether the model needs feature scaling."""
        pass

    @classmethod
    @abstractmethod
    def supports_native_multioutput(cls) -> bool:
        """Whether one estimator can fit a two-dimensional regression target."""
        pass

    @classmethod
    def get_cat_encoding_strategy(cls) -> str:
        """Return the categorical encoding requested from the dataset pipeline."""
        return "onehot"

    @classmethod
    def uses_validation_during_fit(cls) -> bool:
        """True if ``fit`` uses validation data (early stopping, eval sets, etc.)."""
        return False

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Return the model's Default parameter configuration."""
        raise NotImplementedError(
            f"Default hyperparameters have not been implemented for class {cls.__name__}."
        )

    @staticmethod
    def _is_multioutput(y: np.ndarray) -> bool:
        """Return True if y has multiple output columns (shape N x k, k > 1)."""
        return y.ndim == 2 and y.shape[1] > 1

    def _fit_multioutput(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit independent output models in fresh workers, sharing allocated CPUs.

        GPU workers receive device assignments without passing CUDA tensors between processes.
        """
        n_outputs = y.shape[1]

        if self.args.use_gpu and torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            n_jobs = min(n_gpus, n_outputs)
            logger.info(
                f"Multi-output regression: {n_outputs} outputs, {n_gpus} GPU(s) available "
                f"-> {n_jobs} parallel job(s) (round-robin GPU assignment)"
            )
        else:
            n_gpus = 0
            n_jobs = min(n_outputs, max(1, self.args.num_cpus))
            logger.info(
                f"Multi-output regression: {n_outputs} outputs, no GPU "
                f"-> {n_jobs} parallel CPU jobs"
            )

        per_output_args = []
        for i in range(n_outputs):
            args_i = copy.copy(self.args)
            args_i.num_cpus = max(1, self.args.num_cpus // n_jobs)
            if n_gpus > 0:
                args_i.gpu_ids = [i % n_gpus]
                logger.debug(f"  Output {i + 1}/{n_outputs} -> cuda:{i % n_gpus}")

            args_i.objective = "regression"
            args_i.num_outputs = 1
            per_output_args.append(args_i)

        self._output_models_ = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_fit_single_output)(
                self.__class__,
                self.input_parameters,
                per_output_args[i],
                X,
                y[:, i],
                X_val if X_val is not None else None,
                y_val[:, i] if y_val is not None else None,
            )
            for i in range(n_outputs)
        )

        logger.info(f"Multi-output regression training complete ({n_outputs} models)")
        return [], []

    def fit_multioutput(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit multiple targets natively or through independent output models."""
        if self._is_multioutput(y) and not self.supports_native_multioutput():
            logger.info(
                f"{self.__class__.__name__} does not support native multi-output regression. "
                f"Training {y.shape[1]} separate models."
            )
            return self._fit_multioutput(X, y, X_val, y_val)

        return self.fit(X, y, X_val, y_val)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit the estimator on X and y.

        Model-specific overrides may use the validation partition for early stopping.
        Return training and validation loss histories when available."""

        self.model.fit(X, y)

        return [], []

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict targets, using the configured threshold for binary classification.

        Independent output models are combined into an (n_samples, n_outputs) array."""

        if hasattr(self, "_output_models_") and self._output_models_:
            per_output = [model.predict(X) for model in self._output_models_]
            self.predictions = np.column_stack(per_output)
            return self.predictions

        if self.args.objective in ("regression", "multiregression"):
            self.predictions = self.model.predict(X)
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
        """Return probabilities in the dataset's zero-based class order."""
        assert self.model is not None, "self.model is not defined !!!"
        return self._align_class_probabilities(
            self.model.predict_proba(X), self.model.classes_
        )

    def _align_class_probabilities(self, probabilities, classes):
        probabilities = np.asarray(probabilities)
        classes = np.asarray(classes)
        if (
            probabilities.ndim != 2
            or classes.ndim != 1
            or classes.size == 0
            or probabilities.shape[1] != classes.size
        ):
            raise ValueError("Probability columns must match the fitted class labels")
        try:
            indices = classes.astype(np.intp)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "Classifier classes must be zero-based integer labels"
            ) from error
        num_classes = getattr(self.args, "num_classes", None)
        if num_classes is None:
            num_classes = (
                2
                if getattr(self.args, "objective", None) == "binary"
                else int(max(probabilities.shape[1], int(np.max(indices)) + 1))
            )
        if (
            not np.array_equal(classes, indices)
            or np.any(indices < 0)
            or np.any(indices >= num_classes)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError(
                "Classifier classes must be distinct labels in the dataset class range"
            )
        if np.array_equal(indices, np.arange(num_classes)):
            self.prediction_probabilities = probabilities
        else:
            aligned = np.zeros(
                (probabilities.shape[0], num_classes), dtype=probabilities.dtype
            )
            aligned[:, indices] = probabilities
            self.prediction_probabilities = aligned
        return self.prediction_probabilities

    def _normalize_and_align_probabilities(self, probabilities, classes=None):
        """Normalize 1D/single-column backend outputs and align to dataset classes."""
        proba = np.asarray(probabilities)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        num_classes = getattr(self.args, "num_classes", None)
        if num_classes is None and getattr(self.args, "objective", None) == "binary":
            num_classes = 2
        if classes is None:
            classes = getattr(self.model, "classes_", getattr(self, "_classes_", None))
        if classes is not None:
            classes = np.asarray(classes)
            if proba.shape[1] == 1 and classes.size == 2:
                proba = np.column_stack([1.0 - proba.ravel(), proba.ravel()])
            elif (
                num_classes is not None
                and proba.shape[1] == num_classes
                and classes.size != proba.shape[1]
            ):
                classes = np.arange(num_classes)
        else:
            if proba.shape[1] == 1 and num_classes == 2:
                proba = np.column_stack([1.0 - proba.ravel(), proba.ravel()])
            classes = np.arange(proba.shape[1])
        return self._align_class_probabilities(proba, classes)

    def clone(self):
        """Create an unfitted estimator with the same parameters and runtime settings."""
        return self.__class__(
            copy.deepcopy(self.input_parameters), copy.deepcopy(self.args)
        )
