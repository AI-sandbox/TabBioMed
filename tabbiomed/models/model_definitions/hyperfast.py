"""HyperFast with its native imputation, categorical encoding and scaling."""

import numpy as np
from loguru import logger
from typing import Any, Dict, Optional, Tuple
from torch.cuda import is_available as cuda_is_available

try:
    from hyperfast import HyperFastClassifier
except ImportError:
    raise ImportError(
        "hyperfast library is required. Install with: uv pip install hyperfast"
    )

from tabbiomed.models.base_model import BaseModel, ModelArgs


class HyperFast(BaseModel):
    """Pass raw feature values and categorical column indices to HyperFast."""

    @classmethod
    def get_base_params(cls, args):
        return {
            "n_ensemble": 16,
            "batch_size": 1024,
            "nn_bias": False,
            "nn_bias_mini_batches": True,
            "optimization": "ensemble_optimize",
            "optimize_steps": None,
            "stratify_sampling": False,
            "feature_bagging_size": 3000,
            "checkpoint_path": None,
            "feature_bagging": args.num_features > 3000,
            "nn_bias_scale": 1.0,
            "finetuning_max_epochs": 2048,
            "finetuning_lr": 0.0001,
            "finetuning_dropout": 0.15,
            "finetuning_weight_decay": 0.01,
            "finetuning_parameter_scope": "all",
            "finetuning_batch_size": None,
            "label_smoothing": 0.05,
            "gradient_clip_norm": 1.18,
            "scheduler_min_lr": 6e-07,
            "patience_epochs": 50,
            "cooldown_checks": 2,
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)

        self.device = "cuda" if args.use_gpu and cuda_is_available() else "cpu"
        if self.device == "cuda" and not cuda_is_available():
            raise AssertionError(
                "Fit specified to use GPU, but CUDA is not available on this machine. "
                "Please switch to CPU usage instead."
            )

        if args.objective in ["regression", "multiregression"]:
            raise NotImplementedError("HyperFast does not support regression tasks.")

        assert args.objective in (
            "binary",
            "classification",
        ), "Unrecognized objective type. \
                                    Supported types for HyperFast are 'binary' and 'classification'."

        self.n_ensemble = params["n_ensemble"]
        self.batch_size = params["batch_size"]
        self.nn_bias = params["nn_bias"]
        self.nn_bias_mini_batches = params["nn_bias_mini_batches"]
        self.optimization = params["optimization"]
        self.optimize_steps = params["optimize_steps"]
        self.stratify_sampling = params["stratify_sampling"]
        self.cat_features = args.cat_idx

        self.feature_bagging_size = params["feature_bagging_size"]
        self.feature_bagging = params["feature_bagging"]

        from tabbiomed.models.checkpoints import resolve_checkpoint

        checkpoint_path = params["checkpoint_path"]
        if checkpoint_path is None:
            checkpoint_path = resolve_checkpoint("hyperfast", "hyperfast.ckpt")
        self.model = HyperFastClassifier(
            seed=args.seed,
            n_ensemble=self.n_ensemble,
            batch_size=self.batch_size,
            nn_bias=self.nn_bias,
            nn_bias_scale=params["nn_bias_scale"],
            nn_bias_mini_batches=self.nn_bias_mini_batches,
            optimization=self.optimization,
            optimize_steps=self.optimize_steps,
            finetuning_max_epochs=params["finetuning_max_epochs"],
            finetuning_lr=params["finetuning_lr"],
            finetuning_dropout=params["finetuning_dropout"],
            finetuning_weight_decay=params["finetuning_weight_decay"],
            finetuning_parameter_scope=params["finetuning_parameter_scope"],
            finetuning_batch_size=params["finetuning_batch_size"],
            label_smoothing=params["label_smoothing"],
            gradient_clip_norm=params["gradient_clip_norm"],
            scheduler_min_lr=params["scheduler_min_lr"],
            patience_epochs=params["patience_epochs"],
            cooldown_checks=params["cooldown_checks"],
            stratify_sampling=self.stratify_sampling,
            cat_features=self.cat_features,
            feature_bagging=self.feature_bagging,
            feature_bagging_size=self.feature_bagging_size,
            device=self.device,
            custom_path=str(checkpoint_path),
        )

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
    def supports_native_multioutput(cls) -> bool:
        """HyperFast is a classification only model."""
        return False

    @classmethod
    def uses_validation_during_fit(cls) -> bool:
        return True

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Baseline defaults, the runner applies fixed search-space overrides.

        feature_bagging is omitted here, __init__ enables it when num_features > 3000.
        """
        return {
            "n_ensemble": 16,
            "batch_size": 1024,
            "nn_bias": False,
            "nn_bias_scale": 1.0,
            "nn_bias_mini_batches": True,
            "optimization": "ensemble_optimize",
            "optimize_steps": None,
            "finetuning_max_epochs": 2048,
            "finetuning_lr": 1e-4,
            "finetuning_dropout": 0.15,
            "finetuning_weight_decay": 0.01,
            "finetuning_parameter_scope": "all",
            "finetuning_batch_size": None,
            "label_smoothing": 0.05,
            "gradient_clip_norm": 1.18,
            "scheduler_min_lr": 6e-7,
            "patience_epochs": 50,
            "cooldown_checks": 2,
            "stratify_sampling": False,
            "feature_bagging_size": 3000,
        }

    def _log_label_sanity(self, y: np.ndarray) -> None:
        """Validate labels against the declared number of classes."""
        y_flat = np.asarray(y).ravel()
        uniq = np.unique(y_flat)
        n_unique = int(uniq.size)
        num_classes = int(self.args.num_classes)
        logger.info(
            f"HyperFast label check (dataset={self.args.dataset}): "
            f"dtype={y_flat.dtype}, n_samples={y_flat.size}, "
            f"n_unique={n_unique}, min={uniq.min()}, max={uniq.max()}, "
            f"num_classes(metadata)={num_classes}"
        )
        if n_unique <= 20:
            logger.info(f"HyperFast unique y values: {uniq.tolist()}")
        if int(uniq.max()) >= num_classes or int(uniq.min()) < 0:
            logger.warning(
                "HyperFast: y values outside [0, num_classes-1] , likely label/metadata mismatch."
            )
        elif n_unique < num_classes:
            logger.warning(
                f"HyperFast: only {n_unique} classes present in y but num_classes={num_classes} "
                "(e.g. some cell types only in held-out plates), can confuse GPU classifiers."
            )
        elif set(uniq.astype(int).tolist()) != set(range(num_classes)):
            logger.warning(
                "HyperFast: labels are not contiguous 0..num_classes-1 , consider LabelEncoder."
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[list, list]:
        self._log_label_sanity(y)
        self._classes_ = np.unique(np.asarray(y).ravel())
        if (X_val is None) != (y_val is None):
            raise ValueError("Both validation features and labels are required")
        eval_set = None if X_val is None else (X_val, y_val)
        self.model.fit(X, y, eval_set=eval_set)
        return [], []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._normalize_and_align_probabilities(self.model.predict_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        self.prediction_probabilities = proba
        if self.args.objective == "binary":
            self.predictions = (proba[:, 1] >= self.threshold).astype(int)
        else:
            self.predictions = np.argmax(proba, axis=1)
        return self.predictions
