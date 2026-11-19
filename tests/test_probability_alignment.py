from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.metrics import log_loss

from tabbiomed.models.model_definitions.decision_tree import DecisionTree
from tabbiomed.models.model_definitions.elasticnet import ElasticNet
from tabbiomed.models.model_definitions.knn import KNN
from tabbiomed.models.model_definitions.lasso import LassoModel
from tabbiomed.models.model_definitions.linear import LinearModel
from tabbiomed.models.model_definitions.random_forest import RandomForest
from tabbiomed.models.model_definitions.ridge import RidgeModel
from tabbiomed.models.model_definitions.svm import SVM
from tabbiomed.trainer.utils.eval_utils import evaluate_split


class ProbabilityAlignment(unittest.TestCase):
    def model(self, cls=DecisionTree, *, task="binary", num_classes=2, **parameters):
        args = SimpleNamespace(
            objective=task,
            num_classes=num_classes,
            num_features=1,
            num_cpus=1,
            seed=42,
            is_cv=False,
        )
        values = cls.get_default_params()
        if cls is RandomForest:
            values["n_estimators"] = 5
        if cls is KNN:
            values["n_neighbors"] = 1
        values.update(parameters)
        return cls(cls.resolve_parameters(values, args), args)

    def test_single_training_class_keeps_its_label(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        cases = (
            ("binary", 2, 0),
            ("binary", 2, 1),
            ("classification", 3, 0),
            ("classification", 3, 1),
            ("classification", 3, 2),
        )
        for cls in (DecisionTree, RandomForest, KNN):
            for task, num_classes, label in cases:
                with self.subTest(model=cls.__name__, task=task, label=label):
                    model = self.model(cls, task=task, num_classes=num_classes)
                    y = np.full(len(X), label)
                    model.fit(X, y)
                    np.testing.assert_array_equal(model.predict(X), y)
                    expected = np.zeros((len(X), num_classes))
                    expected[:, label] = 1
                    np.testing.assert_array_equal(model.predict_proba(X), expected)

    def test_missing_classes_are_padded_in_all_shared_classifier_wrappers(self):
        X = np.linspace(-1, 1, 12).reshape(-1, 1)
        for cls in (
            DecisionTree,
            RandomForest,
            KNN,
            LinearModel,
            LassoModel,
            RidgeModel,
            ElasticNet,
            SVM,
        ):
            for labels in ((0, 2), (1, 2), (0, 1)):
                with self.subTest(model=cls.__name__, classes=labels):
                    model = self.model(cls, task="classification", num_classes=3)
                    model.fit(X, np.repeat(labels, 6))
                    raw = model.model.predict_proba(X)
                    expected = np.zeros((len(X), 3))
                    expected[:, list(labels)] = raw
                    np.testing.assert_allclose(model.predict_proba(X), expected)
                    expected_labels = np.asarray(labels)[raw.argmax(axis=1)]
                    np.testing.assert_array_equal(model.predict(X), expected_labels)
                    np.testing.assert_allclose(model.prediction_probabilities, expected)

    def test_complete_class_probabilities_are_unchanged(self):
        X = np.linspace(-1, 1, 12).reshape(-1, 1)
        for cls in (
            DecisionTree,
            RandomForest,
            KNN,
            LinearModel,
            LassoModel,
            RidgeModel,
            ElasticNet,
            SVM,
        ):
            for task, count in (("binary", 2), ("classification", 3)):
                with self.subTest(model=cls.__name__, task=task):
                    model = self.model(cls, task=task, num_classes=count)
                    model.fit(X, (np.arange(len(X)) % count).astype(float))
                    raw = model.model.predict_proba(X)
                    np.testing.assert_array_equal(model.predict_proba(X), raw)

    def test_explicit_class_order_and_probability_dtype_are_respected(self):
        X = np.zeros((2, 1))
        for dtype in (np.float32, np.float64):
            for classes, values, expected in (
                ([2, 0, 1], [0.6, 0.3, 0.1], [0.3, 0.1, 0.6]),
                ([2, 0], [0.7, 0.3], [0.3, 0.0, 0.7]),
            ):
                with self.subTest(dtype=dtype, classes=classes):
                    model = self.model(task="classification", num_classes=3)
                    raw = np.tile(np.asarray(values, dtype=dtype), (len(X), 1))
                    model.model = SimpleNamespace(
                        classes_=np.asarray(classes), predict_proba=lambda _: raw
                    )
                    probabilities = model.predict_proba(X)
                    self.assertEqual(probabilities.dtype, dtype)
                    np.testing.assert_allclose(
                        probabilities, np.tile(expected, (len(X), 1))
                    )
                    np.testing.assert_array_equal(model.predict(X), [2, 2])

    def test_invalid_class_metadata_and_probability_shapes_fail_loudly(self):
        X = np.zeros((2, 1))
        for classes, probabilities in (
            ([0, 0], np.ones((2, 2))),
            ([-1, 1], np.ones((2, 2))),
            ([0, 3], np.ones((2, 2))),
            ([0.5, 2], np.ones((2, 2))),
            (["0", "2"], np.ones((2, 2))),
            ([[0, 2]], np.ones((2, 2))),
            ([0, 2], np.ones((2, 3))),
            ([0, 2], np.ones(2)),
            ([], np.ones((2, 0))),
        ):
            with self.subTest(classes=classes, shape=probabilities.shape):
                model = self.model(task="classification", num_classes=3)
                model.model = SimpleNamespace(
                    classes_=np.asarray(classes), predict_proba=lambda _: probabilities
                )
                with self.assertRaises(ValueError):
                    model.predict_proba(X)

    def test_binary_threshold_is_applied_after_alignment(self):
        X = np.zeros((2, 1))
        model = self.model(threshold=0.8)
        raw = np.array([[0.7, 0.3], [0.9, 0.1]])
        model.model = SimpleNamespace(
            classes_=np.array([1, 0]), predict_proba=lambda _: raw
        )
        np.testing.assert_array_equal(model.predict(X), [0, 1])
        np.testing.assert_array_equal(model.prediction_probabilities, raw[:, ::-1])

    def test_split_evaluation_receives_the_full_class_space(self):
        X = np.arange(6, dtype=float).reshape(-1, 1)
        model = self.model(task="classification", num_classes=3)
        model.fit(X, np.array([0, 0, 0, 2, 2, 2]))
        y = np.array([0, 1, 2, 0, 1, 2])
        result = evaluate_split(
            model, X, y, objective="classification", compute_proba=True
        )
        expected = np.array([[1, 0, 0]] * 3 + [[0, 0, 1]] * 3, dtype=float)
        np.testing.assert_array_equal(result.y_proba, expected)
        self.assertAlmostEqual(
            result.metrics["log_loss"], log_loss(y, expected, labels=[0, 1, 2])
        )


if __name__ == "__main__":
    unittest.main()
