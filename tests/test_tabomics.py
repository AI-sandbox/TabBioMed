import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from tabbiomed.models import get_model_spec, get_search_space, str2model
from tabbiomed.models.model_definitions import tabomics as module


class FakeClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        self.calls.append(len(X))
        return np.tile([0.25, 0.75], (len(X), 1))


def fake_backend(weights_sha="published"):
    backend = ModuleType("tabomics")
    backend.Classifier = FakeClassifier
    hub = ModuleType("tabomics.hub")
    hub.PUBLISHED_MODELS = {"v2.5": SimpleNamespace(sha256="published")}
    hub.sha256_of = Mock(return_value=weights_sha)
    return {"tabomics": backend, "tabomics.hub": hub}


def make_args(**overrides):
    settings = dict(
        objective="classification", num_classes=3, use_gpu=False, gpu_ids=[],
        cat_idx=[1], seed=42, val_batch_size=2,
    )
    return SimpleNamespace(**{**settings, **overrides})


class TabOmicsTests(unittest.TestCase):
    def test_registration_and_defaults(self):
        for name, members in (("TabOmicsv2.5x1", 1), ("TabOmicsv2.5x8", 8)):
            with self.subTest(model=name):
                spec = get_model_spec(name)
                self.assertTrue(spec.gpu)
                self.assertFalse(spec.hpo_enabled)
                self.assertFalse(spec.requires_separate_env)
                self.assertEqual(spec.tasks, ("binary", "classification"))
                self.assertEqual(get_search_space(name), {})
                cls = str2model(name)
                self.assertEqual(cls.get_default_params(), {"n_estimators": members})
                params = cls.resolve_parameters(cls.get_default_params(), make_args())
                self.assertIsNone(params["custom_model_dir"])

    def test_pinned_release_batched_prediction_and_class_alignment(self):
        cache = Path("/cache/tabomics_2_5")
        resolve = Mock(side_effect=lambda family, name, directory: cache / name)
        args = make_args()
        cls = str2model("TabOmicsv2.5x8")
        model = cls(cls.resolve_parameters(cls.get_default_params(), args), args)
        with patch.dict(sys.modules, fake_backend()), patch.object(module, "resolve_checkpoint", resolve):
            model.fit(np.ones((6, 2)), np.tile([0, 2], 3))
            probabilities = model.predict_proba(np.ones((5, 2)))
        self.assertEqual(
            resolve.call_args_list,
            [call("tabomics_2_5", "model.safetensors", None), call("tabomics_2_5", "config.json", None)],
        )
        self.assertEqual(
            model.model.kwargs,
            dict(model=cache, device="cpu", categorical_features=[1],
                 infer_categorical=False, n_estimators=8, random_state=42),
        )
        self.assertEqual(model.model.calls, [2, 2, 1])
        np.testing.assert_allclose(probabilities, np.tile([0.25, 0.0, 0.75], (5, 1)))

    def test_rejects_unpublished_weights_and_regression(self):
        resolve = Mock(side_effect=lambda family, name, directory: Path("/local") / name)
        args = make_args()
        cls = str2model("TabOmicsv2.5x1")
        model = cls(cls.resolve_parameters(cls.get_default_params(), args), args)
        with patch.dict(sys.modules, fake_backend("other")), patch.object(module, "resolve_checkpoint", resolve):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                model.fit(np.ones((4, 2)), np.array([0, 1, 0, 1]))
            model.args = make_args(objective="regression")
            with self.assertRaisesRegex(ValueError, "classification tasks only"):
                model.fit(np.ones((4, 2)), np.arange(4.0))


if __name__ == "__main__":
    unittest.main()
