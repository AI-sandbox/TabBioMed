"""Default-only TabPFN-3.5 inference."""

# Adapted from https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/models/tabpfn_3_5/model.py
import numpy as np
import pandas as pd
import torch

from .base_model import BaseModel
from .checkpoints import resolve_checkpoint


class TabPFN35(BaseModel):
    allow_metadata_scaling = False
    optional_parameters = ("custom_model_dir",)

    @classmethod
    def needs_missing_value_imputation(cls):
        return False

    @classmethod
    def needs_cat_encoding(cls):
        return False

    @classmethod
    def needs_scaling(cls):
        return False

    @classmethod
    def supports_native_multioutput(cls):
        return False

    @classmethod
    def get_default_params(cls):
        return {
            "checkpoint": "tabpfn-v3.5-20260909.safetensors",
            "ignore_pretraining_limits": True,
            "n_preprocessing_jobs": 1,
            "inference_batch_size": 150_000,
        }

    def __init__(self, params, args):
        super().__init__(params, args)
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        self._estimator_class = (
            TabPFNClassifier
            if args.objective in ("binary", "classification")
            else TabPFNRegressor
        )

    def fit(self, X, y, X_val=None, y_val=None):
        device = (
            (f"cuda:{self.args.gpu_ids[0]}" if self.args.gpu_ids else "cuda")
            if self.args.use_gpu and torch.cuda.is_available()
            else "cpu"
        )
        params = dict(self.params)
        checkpoint = params.pop("checkpoint")
        directory = params.pop("custom_model_dir", None)
        self._batch_size = params.pop("inference_batch_size")
        if not isinstance(self._batch_size, int) or self._batch_size < 1:
            raise ValueError("inference_batch_size must be a positive integer")
        self.model = self._estimator_class(
            **params,
            model_path=resolve_checkpoint("tabpfn_3_5", checkpoint, directory),
            device=device,
            random_state=self.args.seed,
            categorical_features_indices=(
                [] if self.args.cat_idx is None else list(self.args.cat_idx)
            ),
        )
        self.model.fit(pd.DataFrame(X), y)
        return [], []

    def _predict_batches(self, X, method):
        frame = pd.DataFrame(X)
        return np.concatenate(
            [
                np.asarray(
                    getattr(self.model, method)(
                        frame.iloc[start : start + self._batch_size]
                    )
                )
                for start in range(0, len(frame), self._batch_size)
            ],
            axis=0,
        )

    def predict(self, X):
        if hasattr(self, "_output_models_"):
            return super().predict(X)
        if self.args.objective in ("regression", "multiregression"):
            self.predictions = self._predict_batches(X, "predict")
            return self.predictions
        return super().predict(X)

    def predict_proba(self, X):
        return self._align_class_probabilities(
            self._predict_batches(X, "predict_proba"), self.model.classes_
        )
