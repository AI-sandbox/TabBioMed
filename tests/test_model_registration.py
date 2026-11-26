from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.dummy import DummyRegressor

import tabbiomed.models as models
from tabbiomed.models.base_model import BaseModel


class ExampleRegressor(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = DummyRegressor(strategy="mean")

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
        return True

    @classmethod
    def get_default_params(cls):
        return {}


class ModelRegistration(unittest.TestCase):
    def setUp(self):
        state = (
            dict(models._registry),
            dict(models._aliases),
            list(models.all_models),
            set(models.NO_HPO_MODELS),
        )

        def restore():
            models._registry.clear()
            models._registry.update(state[0])
            models._aliases.clear()
            models._aliases.update(state[1])
            models.all_models[:] = state[2]
            models.NO_HPO_MODELS.clear()
            models.NO_HPO_MODELS.update(state[3])

        self.addCleanup(restore)

    def test_builtin_roster_and_search_spaces(self):
        self.assertEqual(
            set(models.NO_HPO_MODELS),
            {"LinearModel", "NeoLsSVM", "Causilo-1.5", "Causilo-1.5-Light"},
        )
        for name in models.BENCHMARK_MODELS:
            self.assertIsInstance(models.get_search_space(name), dict)
        for name in ("HyperFast", "TabPFN-Wide"):
            self.assertEqual(
                models.get_model_spec(name).tasks, ("binary", "classification")
            )
        self.assertEqual(models.model_name_map("xrfm"), "xRFM")
        self.assertEqual(
            models.get_model_spec("xRFM").implementation,
            "tabbiomed.models.model_definitions.xrfm:xRFM",
        )
        self.assertFalse(models.get_model_spec("RandomForest").gpu)
        self.assertTrue(models.get_model_spec("TabM").gpu)

    def test_causilo_uses_published_defaults_and_original_feature_arrays(self):
        backend = Mock()
        backend.CausiloClassifier.return_value.classes_ = np.array([0, 2])
        backend.CausiloClassifier.return_value.predict_proba.return_value = [[0.2, 0.8]]
        args = SimpleNamespace(
            objective="classification", num_classes=3, cat_idx=[1],
            seed=42, use_gpu=True, gpu_ids=[0],
        )
        train = np.array([[0.5, 11], [1.0, 35]])
        test = np.array([[0.6, 35]])
        for name, ensemble_size in [("causilo", None), ("causilo-light", 1)]:
            with self.subTest(model=name), patch.dict("sys.modules", {"causilo": backend}):
                cls = models.str2model(name)
                model = cls(cls.resolve_parameters(cls.get_default_params(), args), args)
                backend.CausiloClassifier.assert_called_with(
                    device="cuda:0", random_state=42, model_version="1.5",
                    n_estimators=ensemble_size, max_context=None, use_kv_cache=False,
                )
                model.fit(train, np.array([0, 2]))
                np.testing.assert_allclose(model.predict_proba(test), [[0.2, 0.0, 0.8]])
                self.assertIs(model.model.fit.call_args.args[0], train)
                self.assertIs(model.model.predict_proba.call_args.args[0], test)

    def test_new_model_fit_and_metrics(self):
        models.register_model(
            "ExampleRegressor",
            ExampleRegressor,
            tasks=("regression",),
            hpo_enabled=False,
            aliases=("example",),
        )
        self.assertEqual(models.model_name_map("example"), "ExampleRegressor")
        self.assertIn("ExampleRegressor", models.NO_HPO_MODELS)
        cls = models.str2model("example")
        args = SimpleNamespace(objective="regression", use_gpu=False, num_cpus=1)
        model = cls(cls.resolve_parameters({}, args), args)
        x, y = np.arange(12).reshape(6, 2), np.arange(6)
        model.fit_multioutput(x, y)
        np.testing.assert_allclose(model.predict(x), np.full(6, 2.5))
        from tabbiomed.evaluator.metrics import compute_metrics

        self.assertAlmostEqual(
            compute_metrics(y, model.predict(x), objective="regression")["mse"],
            np.var(y),
        )

    def test_custom_search_space_is_not_shared_with_caller(self):
        space = {"offset": {"type": "categorical", "choices": [0, 1]}}
        models.register_model(
            "SearchExample", ExampleRegressor, search_space=space, gpu=True
        )
        self.assertTrue(models.get_model_spec("SearchExample").gpu)
        space["offset"]["choices"].append(2)
        loaded = models.get_search_space("SearchExample")
        self.assertEqual(loaded["offset"]["choices"], [0, 1])
        loaded["offset"]["choices"].append(3)
        self.assertEqual(
            models.get_search_space("SearchExample")["offset"]["choices"], [0, 1]
        )

    def test_registration_does_not_import_optional_model_packages(self):
        with patch.object(
            models, "import_module", side_effect=AssertionError("eager import")
        ):
            models.register_model(
                "LazyExample", "example_package:Example", hpo_enabled=False
            )
            self.assertEqual(
                models.get_model_spec("LazyExample").implementation,
                "example_package:Example",
            )

    def test_existing_names_and_aliases_cannot_be_replaced(self):
        with self.assertRaises(ValueError):
            models.register_model(
                "Other", ExampleRegressor, hpo_enabled=False, aliases=("RF",)
            )
        self.assertNotIn("Other", models.all_models)
        with self.assertRaises(ValueError):
            models.register_model("../Other", ExampleRegressor, hpo_enabled=False)


if __name__ == "__main__":
    unittest.main()
