"""Frozen TabOmics v2.5 classifiers with one or eight inference members."""

import numpy as np

from tabbiomed.models.base_model import BaseModel
from tabbiomed.models.checkpoints import resolve_checkpoint


CHECKPOINT_FAMILY = "tabomics_2_5"
RELEASE_FILES = ("model.safetensors", "config.json")
MAX_NATIVE_CLASSES = 10


class TabOmicsV25X1(BaseModel):
    required_parameters = ("n_estimators", "custom_model_dir", "inference_batch_size")
    n_estimators = 1

    @classmethod
    def get_base_params(cls, args):
        return {
            **cls.get_default_params(),
            "custom_model_dir": None,
            "inference_batch_size": getattr(args, "val_batch_size", 256),
        }

    @classmethod
    def get_default_params(cls):
        return {"n_estimators": cls.n_estimators}

    @classmethod
    def validate_parameters(cls, parameters, args):
        super().validate_parameters(parameters, args)
        if parameters["n_estimators"] != cls.n_estimators:
            raise ValueError(f"{cls.__name__} uses {cls.n_estimators} ensemble members")
        batch_size = parameters["inference_batch_size"]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("inference_batch_size must be a positive integer")

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

    def fit(self, X, y, X_val=None, y_val=None):
        if self.args.objective not in ("binary", "classification"):
            raise ValueError("TabOmics v2.5 supports classification tasks only")

        from tabomics import Classifier
        from tabomics.hub import PUBLISHED_MODELS, sha256_of

        files = [
            resolve_checkpoint(CHECKPOINT_FAMILY, name, self.params["custom_model_dir"])
            for name in RELEASE_FILES
        ]
        weights = files[0]
        release = weights.parent
        if sha256_of(weights) != PUBLISHED_MODELS["v2.5"].sha256:
            raise ValueError(f"TabOmics v2.5 checkpoint SHA-256 differs: {weights}")

        device = "cpu"
        if self.args.use_gpu:
            device = f"cuda:{self.args.gpu_ids[0]}" if self.args.gpu_ids else "cuda"
        y = np.asarray(y).ravel()
        classifier = Classifier(
            model=release,
            device=device,
            categorical_features=list(self.args.cat_idx or ()),
            infer_categorical=False,
            n_estimators=self.params["n_estimators"],
            random_state=self.args.seed,
        )
        if np.unique(y).size > MAX_NATIVE_CLASSES:
            from tabpfn_extensions.many_class import ManyClassClassifier

            classifier = ManyClassClassifier(
                estimator=classifier,
                alphabet_size=MAX_NATIVE_CLASSES,
                random_state=self.args.seed,
            )
        self.model = classifier.fit(X, y)
        return [], []

    def predict_proba(self, X):
        batch_size = self.params["inference_batch_size"]
        if len(X) <= batch_size:
            return super().predict_proba(X)

        # Query rows can be batched while each pass retains the full training context.
        probabilities = np.concatenate(
            [
                self.model.predict_proba(X[start : start + batch_size])
                for start in range(0, len(X), batch_size)
            ],
            axis=0,
        )
        return self._align_class_probabilities(probabilities, self.model.classes_)


class TabOmicsV25X8(TabOmicsV25X1):
    n_estimators = 8
