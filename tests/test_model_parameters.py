from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import optuna
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tabbiomed.models import get_search_space, str2model
from tabbiomed.models.base_model import BaseModel
from tabbiomed.models.model_definitions.elasticnet import ElasticNet
from tabbiomed.models.model_definitions.lasso import LassoModel
from tabbiomed.models.model_definitions.ridge import RidgeModel
from tabbiomed.models.model_definitions.svm import SVM
from tabbiomed.trainer.train_eval import train
from tabbiomed.trainer.utils.parameter_config import define_trial_parameters_from_config


class ModelParameters(unittest.TestCase):
    def args(self, **changes):
        return SimpleNamespace(
            **{
                "objective": "regression",
                "num_cpus": 1,
                "use_gpu": False,
                "seed": 42,
                "num_features": 10,
                "epochs": 100,
                "gpu_ids": [],
                "num_outputs": 1,
                "cat_idx": [],
                "val_batch_size": 256,
                **changes,
            }
        )

    def test_default_and_search_solver_limits_are_preserved(self):
        for model, sampled, limit in (
            (SVM, {"C": 0.5}, -1),
            (LassoModel, {"alpha": 0.5}, 10000),
            (RidgeModel, {"alpha": 0.5}, 10000),
            (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5}, 10000),
        ):
            for task in ("binary", "regression"):
                with self.subTest(model=model.__name__, task=task):
                    args = self.args(objective=task)
                    original = deepcopy(sampled)
                    default = model.resolve_parameters(model.get_default_params(), args)
                    tuned = model.resolve_parameters(sampled, args)
                    self.assertEqual(
                        model(default, args).model.get_params()["max_iter"], 100
                    )
                    self.assertEqual(
                        model(tuned, args).model.get_params()["max_iter"], limit
                    )
                    self.assertEqual(sampled, original)
                    self.assertEqual(model.resolve_parameters(tuned, args), tuned)

    def test_missing_values_fail_before_model_construction_or_data_access(self):
        args = self.args()
        complete = SVM.resolve_parameters({"C": 1.0}, args)
        for key in ("C", "max_iter", "scaler", "threshold"):
            with self.subTest(key=key):
                incomplete = {
                    name: value for name, value in complete.items() if name != key
                }
                with self.assertRaisesRegex(ValueError, key):
                    SVM(incomplete, args)
                with self.assertRaisesRegex(ValueError, key):
                    train(SVM, incomplete, args, None, None, [])

    def test_unknown_parameters_fail_before_construction_or_data_access(self):
        for task in ("binary", "regression"):
            with self.subTest(task=task):
                args = self.args(objective=task)
                supplied = {"C": 1.0, "max_iters": 7}
                with self.assertRaisesRegex(ValueError, "unrecognized.*max_iters"):
                    SVM.resolve_parameters(supplied, args)
                values = SVM.resolve_parameters({"C": 1.0}, args)
                values["max_iters"] = 7
                with self.assertRaisesRegex(ValueError, "unrecognized.*max_iters"):
                    SVM(values, args)
                with self.assertRaisesRegex(ValueError, "unrecognized.*max_iters"):
                    train(SVM, values, args, None, None, [])
                corrected = SVM.resolve_parameters({"C": 1.0, "max_iter": 7}, args)
                self.assertEqual(SVM(corrected, args).model.max_iter, 7)
                self.assertEqual(supplied, {"C": 1.0, "max_iters": 7})

    def test_search_space_does_not_make_unknown_names_valid(self):
        space = {
            **get_search_space("SVM"),
            "max_iters": {"type": "int", "low": 7, "high": 7},
        }
        with patch(
            "tabbiomed.trainer.utils.parameter_config.load_model_search_space",
            return_value=space,
        ):
            values = define_trial_parameters_from_config(
                optuna.trial.FixedTrial(
                    {"C": 1.0, "scaler": "standard", "max_iters": 7}
                ),
                "SVM",
            )
        with self.assertRaisesRegex(ValueError, "unrecognized.*max_iters"):
            SVM.resolve_parameters(values, self.args())

    def test_configuration_hooks_cannot_hide_or_add_unknown_names(self):
        class DroppingModel(SVM):
            @classmethod
            def configure_parameters(cls, parameters, args, supplied):
                parameters.pop("max_iters", None)
                return parameters

        class AddingModel(SVM):
            @classmethod
            def configure_parameters(cls, parameters, args, supplied):
                return {**parameters, "max_iters": 7}

        for model, values in (
            (DroppingModel, {"C": 1.0, "max_iters": 7}),
            (AddingModel, {"C": 1.0}),
        ):
            with self.subTest(model=model.__name__):
                with self.assertRaisesRegex(ValueError, "unrecognized.*max_iters"):
                    model.resolve_parameters(values, self.args())

    def test_declared_optional_parameters_are_accepted(self):
        from tabbiomed.models.model_definitions.random_forest import RandomForest

        args = self.args()
        values = RandomForest.resolve_parameters(
            {**RandomForest.get_default_params(), "min_samples_leaf": 3}, args
        )
        self.assertEqual(RandomForest(values, args).model.min_samples_leaf, 3)
        with self.assertRaisesRegex(ValueError, "unrecognized.*min_sample_leaf"):
            RandomForest.resolve_parameters(
                {**RandomForest.get_default_params(), "min_sample_leaf": 3}, args
            )

    def test_native_lightgbm_parameter_names_and_aliases(self):
        if importlib.util.find_spec("lightgbm") is None:
            self.skipTest("Requires the trees extra")
        from tabbiomed.models.model_definitions.lightgbm import LightGBM

        args = self.args()
        values = LightGBM.resolve_parameters(
            {"max_bin": 31, "min_child_samples": 3}, args
        )
        self.assertEqual(values["max_bin"], 31)
        self.assertEqual(values["min_child_samples"], 3)
        with self.assertRaisesRegex(ValueError, "unrecognized.*max_bni"):
            LightGBM.resolve_parameters({"max_bni": 31}, args)

    def test_native_catboost_parameter_names(self):
        if importlib.util.find_spec("catboost") is None:
            self.skipTest("Requires the trees extra")
        from tabbiomed.models.model_definitions.catboost import CatBoost

        args = self.args()
        values = CatBoost.resolve_parameters({"random_strength": 0.5}, args)
        self.assertEqual(values["random_strength"], 0.5)
        with self.assertRaisesRegex(ValueError, "unrecognized.*random_strenght"):
            CatBoost.resolve_parameters({"random_strenght": 0.5}, args)

    def test_native_xgboost_parameter_names_and_aliases(self):
        if importlib.util.find_spec("xgboost") is None:
            self.skipTest("Requires the trees extra")
        from tabbiomed.models.model_definitions.xgboost import XGBoost

        args = self.args()
        values = XGBoost.resolve_parameters({"max_bin": 31, "nthread": 1}, args)
        self.assertEqual(values["max_bin"], 31)
        self.assertEqual(values["nthread"], 1)
        with self.assertRaisesRegex(ValueError, "unrecognized.*max_bni"):
            XGBoost.resolve_parameters({"max_bni": 31}, args)

    def test_shared_settings_do_not_leak_to_backend_parameters(self):
        args = self.args(objective="binary")
        values = SVM.resolve_parameters(
            {"C": 2.0, "threshold": 0.8, "scaler": "quantile"}, args
        )
        model = SVM(values, args)
        self.assertEqual(model.threshold, 0.8)
        self.assertEqual(model.input_parameters, values)
        self.assertNotIn("threshold", model.params)
        self.assertNotIn("scaler", model.params)
        self.assertEqual(model.model.C, 2.0)

    def test_clones_preserve_complete_resolved_configurations(self):
        for name in (
            "DecisionTree",
            "RandomForest",
            "KNN",
            "LinearModel",
            "LassoModel",
            "RidgeModel",
            "ElasticNet",
            "SVM",
        ):
            model_class = str2model(name)
            for task in ("binary", "regression"):
                with self.subTest(model=name, task=task):
                    args = self.args(objective=task)
                    values = model_class.resolve_parameters(
                        {
                            **model_class.get_default_params(),
                            "threshold": 0.8 if task == "binary" else None,
                            "scaler": "quantile",
                        },
                        args,
                    )
                    model = model_class(values, args)
                    cloned = model.clone()
                    self.assertIs(type(cloned), model_class)
                    self.assertIsNot(cloned, model)
                    self.assertIsNot(cloned.model, model.model)
                    self.assertEqual(cloned.input_parameters, values)
                    self.assertEqual(cloned.threshold, model.threshold)
                    self.assertEqual(
                        cloned.model.get_params(), model.model.get_params()
                    )
                    self.assertEqual(vars(cloned.args), vars(args))

    def test_cloning_a_fitted_model_returns_an_unfitted_independent_estimator(self):
        model_class = str2model("DecisionTree")
        args = self.args(objective="binary", num_classes=2, num_features=1)
        model = model_class(
            model_class.resolve_parameters(model_class.get_default_params(), args), args
        )
        X = np.arange(6).reshape(-1, 1)
        y = np.array([0, 0, 0, 1, 1, 1])
        model.fit(X, y)
        original_predictions = model.predict(X).copy()
        cloned = model.clone()
        with self.assertRaises(NotFittedError):
            check_is_fitted(cloned.model)
        self.assertIsNone(cloned.predictions)
        self.assertIsNone(cloned.prediction_probabilities)
        cloned.fit(X, 1 - y)
        np.testing.assert_array_equal(cloned.predict(X), 1 - y)
        np.testing.assert_array_equal(model.predict(X), original_predictions)

    def test_clones_isolate_nested_parameters_and_runtime_arguments(self):
        class ConfiguredTree(str2model("DecisionTree")):
            optional_parameters = ("settings",)

            def __init__(self, params, args):
                super().__init__(params, args)
                self.params["backend_only"] = True

        args = self.args(
            gpu_ids=[0], cat_idx=[0], optimize_optuna_with={"binary": "roc_auc"}
        )
        values = ConfiguredTree.resolve_parameters(
            {
                **ConfiguredTree.get_default_params(),
                "settings": {"labels": ["original"]},
            },
            args,
        )
        model = ConfiguredTree(values, args)
        snapshot = deepcopy(model.input_parameters)
        cloned = model.clone()
        self.assertNotIn("backend_only", cloned.input_parameters)
        cloned.params["settings"]["labels"].append("changed")
        cloned.args.gpu_ids.append(1)
        cloned.args.cat_idx.append(1)
        cloned.args.optimize_optuna_with["binary"] = "accuracy"
        self.assertEqual(model.input_parameters, snapshot)
        self.assertEqual(cloned.input_parameters, snapshot)
        self.assertEqual(model.params["settings"], snapshot["settings"])
        self.assertEqual(model.args.gpu_ids, [0])
        self.assertEqual(model.args.cat_idx, [0])
        self.assertEqual(model.args.optimize_optuna_with, {"binary": "roc_auc"})

    def test_incomplete_search_specification_cannot_use_a_fixed_value(self):
        space = get_search_space("SVM")
        space["max_iter"] = {"type": "unsupported"}
        trial = optuna.trial.FixedTrial({"C": 1.0, "scaler": "standard"})
        with patch(
            "tabbiomed.trainer.utils.parameter_config.load_model_search_space",
            return_value=space,
        ):
            with self.assertRaisesRegex(ValueError, "max_iter"):
                define_trial_parameters_from_config(
                    trial, "SVM", dataset_metadata={"objective": "binary"}
                )

    def test_inactive_conditional_parameter_is_not_required(self):
        space = {
            "use_tree": {"type": "categorical", "choices": [False, True]},
            "tree_depth": {"type": "conditional", "depends_on": "use_tree", True: 4},
        }
        with patch(
            "tabbiomed.trainer.utils.parameter_config.load_model_search_space",
            return_value=space,
        ):
            parameters = define_trial_parameters_from_config(
                optuna.trial.FixedTrial({"use_tree": False}), "Example"
            )
        self.assertEqual(parameters, {"use_tree": False})

    def test_tabpfn_auto_threads_follow_per_output_allocations(self):
        library = SimpleNamespace(TabPFNClassifier=object, TabPFNRegressor=object)
        source = (
            Path(__import__("tabbiomed").__file__).parent
            / "models/model_definitions/tabpfn_v2_5.py"
        )
        spec = importlib.util.spec_from_file_location("test_tabpfn_parameters", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"tabpfn": library}):
            spec.loader.exec_module(module)
        model_class = module.TabPFN
        parent_args = self.args(num_cpus=10)
        child_args = self.args(num_cpus=2)
        for supplied, expected in (({}, 2), ({"n_jobs": 7}, 7)):
            with self.subTest(supplied=supplied):
                values = model_class.resolve_parameters(supplied, parent_args)
                self.assertEqual(model_class(values, child_args).n_jobs, expected)
                incomplete = {
                    key: value for key, value in values.items() if key != "n_jobs"
                }
                with self.assertRaisesRegex(ValueError, "n_jobs"):
                    model_class(incomplete, child_args)

    def test_native_realmlp_configuration_and_feature_cap_are_preserved(self):
        class NativePreset:
            def get_config(self):
                return {
                    "act": "native",
                    "n_hidden_layers": 7,
                    "plr_hidden_2": 4,
                    "ls_eps": 0.03,
                }

        library = SimpleNamespace(
            **{
                name: NativePreset
                for name in (
                    "RealMLP_TD_Classifier",
                    "RealMLP_TD_Regressor",
                    "RealMLP_TD_S_Classifier",
                    "RealMLP_TD_S_Regressor",
                )
            }
        )
        source = (
            Path(__import__("tabbiomed").__file__).parent
            / "models/model_definitions/realmlp.py"
        )
        spec = importlib.util.spec_from_file_location("test_realmlp_parameters", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"pytabkit": library}):
            spec.loader.exec_module(module)
        model_class = module.RealMLP
        for task in ("binary", "regression"):
            for width in (10, 9000):
                with self.subTest(task=task, width=width):
                    args = self.args(objective=task, num_features=width)
                    values = model_class.resolve_parameters(
                        model_class.get_default_params(), args
                    )
                    model = model_class(values, args)
                    self.assertEqual(model.n_ens, 1)
                    self.assertEqual(model.act, "native")
                    self.assertEqual(model.n_hidden_layers, 7)
                    self.assertEqual(model.plr_hidden_2, 16 if width > 8192 else 4)
                    self.assertEqual(
                        model_class.resolve_parameters(values, args), values
                    )
        tuned = model_class.resolve_parameters({"n_ens": 8}, self.args())
        self.assertEqual(model_class(tuned, self.args()).n_ens, 8)
        self.assertFalse(tuned["use_native_defaults"])


if __name__ == "__main__":
    unittest.main()
