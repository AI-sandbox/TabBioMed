import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from tabbiomed.models import str2model


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("catboost", "lightgbm", "xgboost")),
    "Optional tree libraries are not installed",
)
class GBDTSeeds(unittest.TestCase):
    def args(self, seed):
        return SimpleNamespace(
            objective="binary",
            num_classes=2,
            num_cpus=1,
            use_gpu=False,
            seed=seed,
            epochs=5,
            early_stopping_rounds=2,
            logging_period=False,
            dataset="example",
            cat_idx=[],
            gpu_ids=[],
        )

    def test_default_and_tuned_parameters_include_the_requested_native_seed(self):
        for name, key in (
            ("CatBoost", "random_seed"),
            ("LightGBM", "seed"),
            ("XGBoost", "seed"),
        ):
            cls = str2model(name)
            for seed in (7, 42):
                for defaults in (False, True):
                    with self.subTest(model=name, seed=seed, defaults=defaults):
                        args = self.args(seed)
                        values = cls.resolve_parameters(
                            cls.get_default_params() if defaults else {}, args
                        )
                        self.assertEqual(values[key], seed)
                        model = cls(values, args)
                        self.assertEqual(model.params[key], seed)
                        self.assertEqual(model.input_parameters[key], seed)
                        self.assertEqual(cls.resolve_parameters(values, args), values)

    def test_explicit_native_seed_aliases_are_resolved_without_duplicates(self):
        for name, key, aliases in (
            ("CatBoost", "random_seed", ("random_state",)),
            ("LightGBM", "seed", ("random_seed", "random_state")),
            ("XGBoost", "seed", ("random_state",)),
        ):
            cls = str2model(name)
            args = self.args(42)
            for alias in aliases:
                with self.subTest(model=name, alias=alias):
                    values = cls.resolve_parameters({alias: 7}, args)
                    self.assertEqual(values[key], 7)
                    self.assertNotIn(alias, values)
                    self.assertEqual(cls(values, args).params[key], 7)
                    with self.assertRaisesRegex(ValueError, "one .* seed"):
                        cls.resolve_parameters({key: 42, alias: 7}, args)

    def test_catboost_backend_uses_the_requested_seed(self):
        cls = str2model("CatBoost")
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.arange(20) % 2
        for seed in (7, 42):
            with self.subTest(seed=seed):
                args = self.args(seed)
                values = cls.resolve_parameters(
                    {**cls.get_default_params(), "iterations": 5}, args
                )
                model = cls(values, args)
                model.fit(X[:16], y[:16], X[16:], y[16:])
                self.assertEqual(model.model.get_all_params()["random_seed"], seed)


if __name__ == "__main__":
    unittest.main()
