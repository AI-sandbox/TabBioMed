import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.dummy import DummyClassifier

from tabbiomed.models import str2model


class WrapperProbabilityAlignment(unittest.TestCase):
    def model(self, name, library, *, task="classification", num_classes=3, **parameters):
        if importlib.util.find_spec(library) is None:
            self.skipTest(f"Optional library {library} is not installed")
        cls = str2model(name)
        args = SimpleNamespace(
            objective=task,
            num_classes=num_classes,
            num_cpus=1,
            use_gpu=False,
            seed=42,
            epochs=5,
            early_stopping_rounds=2,
            logging_period=False,
            dataset="example",
            cat_idx=[],
            gpu_ids=[],
        )
        values = {**cls.get_default_params(), **parameters}
        if name == "CatBoost":
            values["iterations"] = 5
        return cls(cls.resolve_parameters(values, args), args)

    def tabpfn_with_backend(self, name, X, y, **settings):
        model = self.model(name, "tabpfn", **settings)
        model.model = DummyClassifier(strategy="prior").fit(X, y)
        model._all_missing_columns_ = np.zeros(X.shape[1], dtype=bool)
        model._inference_batch_size = 2
        return model

    def test_tabpfn_probabilities_preserve_missing_and_complete_class_labels(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        for name in ("TabPFNv2.5", "RealTabPFNv2.5"):
            for labels in ((0, 2), (1, 2), (0, 1), (0, 1, 2)):
                with self.subTest(model=name, classes=labels):
                    y = np.resize(labels, len(X))
                    y[-3:] = labels[-1]
                    model = self.tabpfn_with_backend(name, X, y)
                    raw = model.model.predict_proba(X)
                    expected = np.zeros((len(X), 3))
                    expected[:, model.model.classes_.astype(int)] = raw
                    np.testing.assert_allclose(model.predict_proba(X), expected)
                    np.testing.assert_array_equal(
                        model.predict(X), model.model.predict(X)
                    )
                    np.testing.assert_allclose(model.prediction_probabilities, expected)

    def test_tabpfn_single_probability_column_keeps_its_native_class(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        for name in ("TabPFNv2.5", "RealTabPFNv2.5"):
            for label in range(3):
                with self.subTest(model=name, label=label):
                    model = self.tabpfn_with_backend(name, X, np.full(len(X), label))
                    expected = np.zeros((len(X), 3))
                    expected[:, label] = 1.0
                    np.testing.assert_array_equal(model.predict_proba(X), expected)
                    np.testing.assert_array_equal(model.predict(X), label)

    def test_tabpfn_binary_prediction_retains_the_configured_threshold(self):
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        for name in ("TabPFNv2.5", "RealTabPFNv2.5"):
            for threshold, label in ((0.5, 1), (0.7, 0)):
                with self.subTest(model=name, threshold=threshold):
                    model = self.tabpfn_with_backend(
                        name, X, y, task="binary", num_classes=2, threshold=threshold
                    )
                    np.testing.assert_array_equal(model.predict(X), label)
                    np.testing.assert_allclose(
                        model.predict_proba(X), model.model.predict_proba(X)
                    )

    def test_catboost_preserves_native_labels_and_probability_column_order(self):
        X = np.linspace(-2, 2, 16).reshape(-1, 1)
        for labels in ((0, 2), (1, 2), (0, 1), (0, 1, 2)):
            with self.subTest(classes=labels):
                model = self.model("CatBoost", "catboost")
                y = np.asarray(labels)[np.arange(len(X)) % len(labels)]
                model.fit(X, y, X, y)
                raw = model.model.predict_proba(X)
                expected = np.zeros((len(X), 3))
                expected[:, model.model.classes_.astype(int)] = raw
                np.testing.assert_allclose(model.predict_proba(X), expected)
                np.testing.assert_array_equal(
                    model.predict(X), model.model.predict(X).ravel()
                )

    def test_neolssvm_votes_use_the_full_dataset_class_space(self):
        X = np.linspace(-2, 2, 16).reshape(-1, 1)
        for labels in ((0, 2), (1, 2), (0, 1), (0, 1, 2)):
            with self.subTest(classes=labels):
                model = self.model("NeoLsSVM", "neo_ls_svm")
                y = np.asarray(labels)[np.arange(len(X)) % len(labels)]
                model.fit(X, y)
                expected = np.zeros((len(X), 3))
                for i, j, native in model._ovo_models:
                    vote = native.predict_proba(X)[:, 1] >= 0.5
                    expected[:, model._classes[i]] += ~vote
                    expected[:, model._classes[j]] += vote
                expected /= len(model._ovo_models)
                np.testing.assert_allclose(model.predict_proba(X), expected)
                np.testing.assert_array_equal(model.predict(X), expected.argmax(axis=1))
                np.testing.assert_allclose(expected.sum(axis=1), 1.0)

    def test_neolssvm_single_multiclass_training_label_is_constant(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        for label in range(3):
            with self.subTest(label=label):
                model = self.model("NeoLsSVM", "neo_ls_svm")
                model.fit(X, np.full(len(X), label))
                expected = np.zeros((len(X), 3))
                expected[:, label] = 1.0
                np.testing.assert_array_equal(model.predict_proba(X), expected)
                np.testing.assert_array_equal(model.predict(X), label)


if __name__ == "__main__":
    unittest.main()
