"""TabDPT with pipeline label encoding and native missing-value and scaling choices.

Multioutput regression fits one regressor per target."""

from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
from typing import Dict, Any, Tuple, Union

from loguru import logger

from tabbiomed.models.base_model import BaseModel, ModelArgs

try:
    from tabdpt import TabDPTClassifier, TabDPTRegressor
except ImportError as e:
    raise ImportError(
        "tabdpt is required for TabDPT. Install with: uv pip install 'tabdpt>=1.1.7'"
    ) from e

try:
    from torch.cuda import is_available as cuda_is_available
except ImportError:
    cuda_is_available = lambda: False


class TabDPT(BaseModel):
    """
    TabDPT - Tabular Discriminative Pre-trained Transformer.

    Label encoding is done by the pipeline, missing values and scaling are
    handled inside TabDPT via missing_indicators and normalizer hyperparameters.
    """

    @classmethod
    def get_base_params(cls, args):
        return {
            "device": None,
            "normalizer": "standard",
            "missing_indicators": False,
            "clip_sigma": 4,
            "feature_reduction": "pca",
            "faiss_metric": "l2",
            "use_flash": True,
            "temperature": 0.8,
            "context_size": 2048,
            "permute_classes": True,
        }

    def __init__(self, params: Dict[str, Any], args: ModelArgs):
        super().__init__(params, args)

        device = params["device"]
        if device is None and getattr(args, "use_gpu", False) and cuda_is_available():
            device = (
                f"cuda:{args.gpu_ids[0]}"
                if getattr(args, "gpu_ids", None) and len(args.gpu_ids) > 0
                else "cuda"
            )
        if device is None:
            device = "cpu"
        if getattr(args, "use_gpu", False) and not cuda_is_available():
            logger.warning("TabDPT: GPU requested but CUDA not available. Using CPU.")

        self.device = str(torch.device(device))
        if self.device == "cuda":
            self.device = f"cuda:{torch.cuda.current_device()}"
        self.seed = getattr(args, "seed", 42)

        self.normalizer = params["normalizer"]
        self.missing_indicators = params["missing_indicators"]
        self.clip_sigma = params["clip_sigma"]
        self.feature_reduction = params["feature_reduction"]
        self.faiss_metric = params["faiss_metric"]
        self.use_flash = params["use_flash"]

        self.temperature = params["temperature"]
        self.context_size = params["context_size"]
        self.permute_classes = params["permute_classes"]

        constructor_kw = dict(
            device="cuda" if torch.device(self.device).type == "cuda" else self.device
        )
        for key in (
            "normalizer",
            "missing_indicators",
            "clip_sigma",
            "feature_reduction",
            "faiss_metric",
        ):
            val = getattr(self, key, None)

            if key == "normalizer" or val is not None:
                constructor_kw[key] = val
        if getattr(self, "use_flash", True) is not None:
            constructor_kw["use_flash"] = self.use_flash

        is_clf = args.objective in ["binary", "classification"]

        if args.objective in ["regression", "multiregression"]:
            model_class = TabDPTRegressor
        elif is_clf:
            model_class = TabDPTClassifier
        else:
            raise ValueError(
                f"TabDPT does not support objective={args.objective!r}. "
                "Use 'classification', 'binary', 'regression', or 'multiregression'."
            )
        with self._device_context():
            self.model = model_class(**constructor_kw)
        self._default_inf_batch_size = self.model.inf_batch_size

    def _device_context(self):
        if torch.device(self.device).type == "cuda":
            return torch.cuda.device(self.device)
        return nullcontext()

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64 if self.normalizer == "power" else None)

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        """TabDPT handles missing values via internal mean imputation, additional binary missing indocator is added based on HPO."""
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        """Pipeline does label encoding, TabDPT receives numeric data."""
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        """TabDPT applies normalizer internally."""
        return False

    @classmethod
    def get_cat_encoding_strategy(cls) -> str:
        return "label"

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return False

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "temperature": 0.8,
            "context_size": 2048,
            "permute_classes": True,
            "normalizer": "standard",
            "missing_indicators": False,
            "clip_sigma": 4,
            "feature_reduction": "pca",
            "faiss_metric": "l2",
            "use_flash": True,
        }

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Union[None, np.ndarray] = None,
        y_val: Union[None, np.ndarray] = None,
    ) -> Tuple[list, list]:
        """Fit TabDPT on training data (no validation set used by the library)."""
        X_arr = self._prepare_features(X)
        y_flat = np.asarray(y).ravel()
        batch_size = self._default_inf_batch_size
        if (
            torch.device(self.device).type == "cuda"
            and self.context_size is not None
            and self.context_size >= 2048
            and len(X_arr) > self.context_size
        ):
            batch_size = min(batch_size, 64)
        self.model.inf_batch_size = batch_size
        if batch_size != self._default_inf_batch_size:
            logger.info(
                f"TabDPT: {len(X_arr)} training rows, context_size={self.context_size}, "
                f"using prediction batches of {batch_size}"
            )
        if self.args.objective in ("binary", "classification"):
            self._classes_ = np.unique(y_flat)
        with self._device_context():
            self.model.fit(X_arr, y_flat)
        return [], []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or regression values."""
        if hasattr(self, "_output_models_") and self._output_models_:
            return super().predict(X)

        if self.args.objective in ("regression", "multiregression"):
            X_arr = self._prepare_features(X)
            with self._device_context():
                try:
                    out = self.model.predict(
                        X_arr,
                        n_ensembles=getattr(self.model, "n_ensembles", 8),
                        context_size=self.context_size,
                        seed=self.seed,
                    )
                except TypeError:
                    try:
                        out = self.model.predict(
                            X_arr,
                            context_size=self.context_size,
                            seed=self.seed,
                        )
                    except TypeError:
                        out = self.model.predict(X_arr)
            self.predictions = np.asarray(out).ravel()
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
        assert self.args.objective in [
            "binary",
            "classification",
        ], "predict_proba only for classification"
        X_arr = self._prepare_features(X)

        with self._device_context():
            if hasattr(self.model, "ensemble_predict_proba"):
                proba = self.model.ensemble_predict_proba(
                    X_arr,
                    temperature=self.temperature,
                    context_size=self.context_size,
                    permute_classes=self.permute_classes,
                    seed=self.seed,
                )
            else:
                proba = self.model.predict_proba(
                    X_arr,
                    temperature=self.temperature,
                    context_size=self.context_size,
                )
        return self._normalize_and_align_probabilities(proba)
