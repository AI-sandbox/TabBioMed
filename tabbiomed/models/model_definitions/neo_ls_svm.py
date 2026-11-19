import numpy as np
from neo_ls_svm import NeoLSSVM

from tabbiomed.models.base_model import BaseModel, ModelArgs


class NeoLsSVM(BaseModel):
    """Use one-vs-one classifiers with vote-based probabilities for multiclass tasks."""

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        n_classes = getattr(args, "num_classes", 2)
        self._use_ovo = args.objective == "classification" and n_classes > 2
        self._ovo_models = None
        self._classes = None

        if args.objective in ["regression", "multiregression"]:
            self.model = NeoLSSVM(estimator_type="regressor")
        elif args.objective == "classification" or args.objective == "binary":
            if self._use_ovo:
                self.model = None
            else:
                self.model = NeoLSSVM(estimator_type="classifier")

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        if self.args.objective in ["regression", "multiregression"]:
            self.model.fit(X, y)
            return [], []
        if not self._use_ovo:
            self.model.fit(X, y)
            return [], []

        classes = np.unique(y)
        n_classes = len(classes)
        self._classes = classes
        self._ovo_models = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                c_i, c_j = classes[i], classes[j]
                mask = (y == c_i) | (y == c_j)
                X_pair = X[mask]
                y_pair = y[mask]
                y_binary = (y_pair == c_j).astype(np.int64)
                clf = NeoLSSVM(estimator_type="classifier")
                clf.fit(X_pair, y_binary)
                self._ovo_models.append((i, j, clf))
        return [], []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self._use_ovo or self._ovo_models is None:
            return super().predict_proba(X)

        n_samples = X.shape[0]
        n_classes = len(self._classes)
        votes = np.zeros((n_samples, n_classes))
        for idx_i, idx_j, clf in self._ovo_models:
            proba = clf.predict_proba(X)
            if proba.shape[1] == 1:
                proba = np.concatenate((1 - proba, proba), axis=1)
            which = (proba[:, 1] >= 0.5).astype(int)
            votes[:, idx_i] += 1 - which
            votes[:, idx_j] += which
        n_pairs = len(self._ovo_models)
        if n_pairs:
            votes /= n_pairs
        elif n_classes == 1:
            votes[:, 0] = 1.0
        return self._align_class_probabilities(votes, self._classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return super().predict(X)

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return True

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        return True

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return False

    @classmethod
    def get_default_params(cls):
        return {
            "scaler": "standard",
        }
