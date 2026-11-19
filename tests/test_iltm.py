import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabbiomed.models import get_recommended_model_configs


class Estimator:
    def __init__(
        self,
        *,
        checkpoint_mix=None,
        retrieval_alpha_adaptive=True,
        tree_use_external_eval_set=False,
        val_subsample_before_preprocessing=True,
        val_checks_per_epoch_target="auto",
        **parameters,
    ):
        self.parameters = parameters

    def fit(self, X, y, eval_set=None):
        self.X = X
        self.y = np.asarray(y)
        self.eval_set = eval_set
        return self

    def predict(self, X):
        return np.full(len(X), self.y.mean())


class Classifier(Estimator):
    def predict_proba(self, X):
        weights = np.arange(1, len(np.unique(self.y)) + 1, dtype=float)
        return np.tile(weights / weights.sum(), (len(X), 1))


class ILTMWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__import__("tabbiomed").__file__).parent
            / "models/model_definitions/iltm.py"
        )
        spec = importlib.util.spec_from_file_location("test_iltm_wrapper", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            "sys.modules",
            {
                "iltm": SimpleNamespace(
                    iLTMClassifier=Classifier, iLTMRegressor=Estimator
                )
            },
        ):
            spec.loader.exec_module(module)
        cls.model_class = module.iLTM

    def make_model(self, task, **parameters):
        args = SimpleNamespace(
            objective=task,
            num_cpus=1,
            num_outputs=2 if task == "multiregression" else 1,
            use_gpu=False,
            gpu_ids=[],
            cat_idx=[],
            seed=42,
        )
        cls = self.model_class
        values = cls.resolve_parameters(
            {**cls.get_default_params(), **parameters}, args
        )
        return cls(values, args)

    def test_native_constructor_arguments_for_each_task(self):
        for task in ("binary", "classification", "regression", "multiregression"):
            with self.subTest(task=task):
                model = self.make_model(task)
                params = model._constructor_params()
                classification = task in ("binary", "classification")
                self.assertIs(
                    model.model_cls, Classifier if classification else Estimator
                )
                self.assertEqual(params["n_ensemble"], 12)
                self.assertEqual(params["checkpoint"], "xgbrconcat")
                self.assertEqual(params["seed"], 42)
                self.assertEqual(params["cat_features"], [])
                self.assertEqual(params["device"], "cpu")
                self.assertEqual(
                    "finetuning_classification_val_metric" in params, classification
                )
                self.assertNotIn("scaler", params)
                self.assertNotIn("threshold", params)

    def test_removed_flags_are_rejected(self):
        for task in ("binary", "classification", "regression", "multiregression"):
            for name in (
                "tree_embedding_enabled",
                "xgboost_tree_enabled",
                "catboost_tree_enabled",
            ):
                with self.subTest(task=task, parameter=name):
                    with self.assertRaisesRegex(ValueError, "unrecognized.*" + name):
                        self.make_model(task, **{name: False})

    def test_recommended_configurations_come_from_the_native_package(self):
        configurations = [dict(checkpoint="xgbrconcat", n_ensemble=i + 1) for i in range(25)]
        provider = SimpleNamespace(get_hyperparameter_configs=lambda: configurations)
        with patch.dict(
            "sys.modules", {"iltm.hyperparameter_search_space": provider}
        ):
            selected = get_recommended_model_configs("iLTM")
        self.assertEqual(selected, configurations)
        selected[0]["n_ensemble"] = 99
        self.assertEqual(configurations[0]["n_ensemble"], 1)

    def test_fit_preserves_validation_and_categorical_inputs(self):
        X = pd.DataFrame(
            {
                "numeric": [1.0, np.nan, 3.0],
                "category": pd.Categorical(["a", "b", "a"]),
                "string": pd.Series(["x", "y", "x"], dtype="string"),
            }
        )
        original = X.copy(deep=True)
        for task, y, expected in (
            ("binary", [0, 1, 0], [0, 0, 0]),
            ("classification", [0, 1, 2], [2, 2, 2]),
            ("regression", [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
        ):
            with self.subTest(task=task):
                model = self.make_model(task, threshold=0.8)
                model.fit(X, np.asarray(y), X.iloc[:2], np.asarray(y[:2]))
                self.assertEqual(model.model.parameters["cat_features"], [1, 2])
                self.assertEqual(model.model.X["category"].dtype, object)
                self.assertEqual(model.model.X["string"].dtype, object)
                validation, labels = model.model.eval_set
                self.assertEqual(validation["category"].dtype, object)
                np.testing.assert_array_equal(labels, y[:2])
                np.testing.assert_allclose(model.predict(X), expected)
                pd.testing.assert_frame_equal(X, original)
                if task != "regression":
                    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)

    def test_independent_regression_outputs(self):
        model = self.make_model("multiregression")
        X = np.arange(6).reshape(3, 2)
        y = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        model.fit_multioutput(X, y)
        self.assertEqual(len(model._output_models_), 2)
        np.testing.assert_allclose(model.predict(X), np.tile([2.0, 5.0], (3, 1)))


if __name__ == "__main__":
    unittest.main()
