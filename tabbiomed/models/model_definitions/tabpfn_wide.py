"""TabPFN-Wide with tabpfnwide==0.3.0 and tabpfn==8.0.3.

Use upstream checkpoint loading/encoder padding and public fit/predict APIs.
Raw inputs use internal preprocessing, >10 classes use ManyClassClassifier ECOC.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin

from .tabpfn_v2_5 import TabPFN, prediction_batch_size_for_context

CHECKPOINTS_PATH = None
DEFAULT_CHECKPOINT = "tabpfn-wide-v2-5k.pt"
BASE_CHECKPOINT = "tabpfn-v2-classifier-v2_default.ckpt"

V2_FEATURE_BUDGET = 1_000_000


def _make_wide_classifier(*, base_tabpfn_path, custom_model_dir=None, **kwargs):
    import torch
    from tabpfn.base import ClassifierModelSpecs
    from tabpfn.model_loading import load_model_criterion_config
    from tabpfnwide.classifier import TabPFNWideClassifier
    from tabpfnwide.patches import patch_encoder_for_narrow_feature_groups

    from tabbiomed.models.checkpoints import resolve_checkpoint

    supplied = Path(base_tabpfn_path).expanduser()
    base_path = (
        resolve_checkpoint("tabpfn_v2", supplied.name, custom_model_dir)
        if len(supplied.parts) == 1
        else supplied.resolve()
    )
    if not base_path.is_file():
        raise FileNotFoundError(f"Local TabPFN v2 checkpoint not found: {base_path}")

    class LocalWideClassifier(TabPFNWideClassifier):
        @staticmethod
        def _build_model_specs(model_name, model_path, features_per_group, device):
            models, _, configs, inference_config = load_model_criterion_config(
                model_path=base_path,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which="classifier",
                version="v2",
                download_if_not_exists=False,
            )
            model, config = models[0], configs[0]
            model.features_per_group = config.features_per_group = features_per_group
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)
            patch_encoder_for_narrow_feature_groups(model, features_per_group)
            return ClassifierModelSpecs(
                model=model,
                architecture_config=config,
                inference_config=inference_config,
            )

    return LocalWideClassifier(**kwargs)


class _TabPFNWideClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable adapter: upstream's **kwargs constructor loses HPO params on clone."""

    def __init__(
        self,
        *,
        device="cpu",
        n_estimators=1,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        categorical_features_indices=None,
        base_tabpfn_path=BASE_CHECKPOINT,
        custom_model_dir=None,
        inference_config=None,
        wide_checkpoint_path="",
        random_state=42,
        inference_batch_size=256,
    ):
        self.device = device
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.categorical_features_indices = categorical_features_indices
        self.base_tabpfn_path = base_tabpfn_path
        self.custom_model_dir = custom_model_dir
        self.inference_config = inference_config
        self.wide_checkpoint_path = wide_checkpoint_path
        self.random_state = random_state
        self.inference_batch_size = inference_batch_size
        self._inference_batch_size = inference_batch_size

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self._all_missing_columns_ = X.isna().all(axis=0).to_numpy()
        self._model = _make_wide_classifier(
            model_path=self.wide_checkpoint_path,
            base_tabpfn_path=self.base_tabpfn_path,
            custom_model_dir=self.custom_model_dir,
            device=self.device,
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            categorical_features_indices=self.categorical_features_indices,
            inference_config=self.inference_config,
            random_state=self.random_state,
            n_preprocessing_jobs=1,
        )
        self._model.fit(X, np.asarray(y).ravel())
        self._inference_batch_size = prediction_batch_size_for_context(
            len(X),
            self.inference_batch_size,
        )
        if self._inference_batch_size != self.inference_batch_size:
            logger.info(
                f"TabPFN-Wide: {len(X)} context rows, "
                f"using prediction batches of {self._inference_batch_size}"
            )
        self.classes_ = self._model.classes_
        self.n_features_in_ = self._model.n_features_in_
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X)
        if self._all_missing_columns_.any():
            X = X.copy()
            X.iloc[:, self._all_missing_columns_] = np.nan
        batch_size = self._inference_batch_size
        if len(X) <= batch_size:
            return self._model.predict_proba(X)
        return np.concatenate(
            [
                self._model.predict_proba(X.iloc[start : start + batch_size])
                for start in range(0, len(X), batch_size)
            ],
            axis=0,
        )

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TabPFNWide(TabPFN):
    @classmethod
    def get_base_params(cls, args):
        return {
            **super().get_base_params(args),
            **cls.get_default_params(),
            "zip_model_path": DEFAULT_CHECKPOINT,
            "custom_model_dir": CHECKPOINTS_PATH,
            "average_before_softmax": False,
            "n_estimators": 1,
            "base_tabpfn_path": BASE_CHECKPOINT,
        }

    def __init__(self, params, args):
        super().__init__(params, args)
        self.zip_model_path = params["zip_model_path"]
        self.custom_model_dir = params["custom_model_dir"]
        self.average_before_softmax = params["average_before_softmax"]
        self.n_estimators = params["n_estimators"]
        self.base_tabpfn_path = params["base_tabpfn_path"]

    @classmethod
    def get_default_params(cls):
        return {
            "softmax_temperature": 0.9,
            "balance_probabilities": False,
            "average_before_softmax": False,
            "n_estimators": 1,
            "zip_model_path": DEFAULT_CHECKPOINT,
            "base_tabpfn_path": BASE_CHECKPOINT,
            "inference_config/OUTLIER_REMOVAL_STD": None,
            "inference_config/POLYNOMIAL_FEATURES": "no",
        }

    def _build_inference_config(self):
        config = super()._build_inference_config()
        for transform in config.get("PREPROCESS_TRANSFORMS", []):
            transform["max_features_per_estimator"] = V2_FEATURE_BUDGET
        return config

    def _resolve_model_path(self):
        from tabbiomed.models.checkpoints import resolve_checkpoint

        return resolve_checkpoint(
            "tabpfn_wide", self.zip_model_path, self.custom_model_dir
        )

    def fit(self, X, y, X_val=None, y_val=None):
        if self.args.objective not in ("binary", "classification"):
            raise ValueError("TabPFN-Wide supports classification only.")
        self._identify_categorical_indices(X)
        estimator = _TabPFNWideClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            categorical_features_indices=self._cat_indices or None,
            base_tabpfn_path=self.base_tabpfn_path,
            custom_model_dir=self.custom_model_dir,
            inference_config=self._build_inference_config() or None,
            wide_checkpoint_path=str(self._resolve_model_path()),
            random_state=getattr(self.args, "seed", 42),
            inference_batch_size=self.inference_batch_size,
        )
        n_classes = len(np.unique(y))
        if n_classes > 10:
            from tabpfn_extensions.many_class import ManyClassClassifier

            self.model = ManyClassClassifier(
                estimator,
                alphabet_size=10,
                random_state=getattr(self.args, "seed", 42),
            )
            logger.info(
                f"TabPFN-Wide: ManyClassClassifier ECOC for {n_classes} classes"
            )
        else:
            self.model = estimator
        self.model.fit(pd.DataFrame(X), y)
        return [], []

    def _predict_in_batches(self, X_df, method):
        return np.asarray(getattr(self.model, method)(X_df))

    def predict(self, X):
        proba = self.predict_proba(X)
        self.prediction_probabilities = proba
        if self.args.objective == "binary":
            self.predictions = (proba[:, 1] >= self.threshold).astype(int)
        else:
            self.predictions = np.argmax(proba, axis=1).astype(int)
        return self.predictions
