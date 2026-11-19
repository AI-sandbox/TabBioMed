import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import optuna
import torch

from tabbiomed.trainer.utils.parameter_config import define_trial_parameters_from_config


class XRFMParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__import__("tabbiomed").__file__).parent
            / "models/model_definitions/xrfm.py"
        )
        spec = importlib.util.spec_from_file_location("test_xrfm_wrapper", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"xrfm": SimpleNamespace()}):
            spec.loader.exec_module(module)
        cls.model_class = module.xRFM

    def args(self, task):
        return SimpleNamespace(
            objective=task, use_gpu=False, num_cpus=1, num_features=2,
            num_outputs=1, cat_idx=[], seed=42, dataset="example",
        )

    def test_default_and_sampled_configs_preserve_internal_algorithm(self):
        for task in ("binary", "classification", "regression", "multiregression"):
            with self.subTest(task=task):
                args = self.args(task)
                study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
                sampled = define_trial_parameters_from_config(
                    study.ask(), "xRFM", dataset_metadata={"objective": task}
                )
                for supplied in (self.model_class.get_default_params(), sampled):
                    params = self.model_class.resolve_parameters(supplied, args)
                    model = self.model_class(params, args)
                    self.assertEqual(model.rfm_params["fit"]["solver"], "solve")
                    self.assertEqual(
                        model.classification_mode,
                        "prevalence" if task in ("binary", "classification") else None,
                    )
                    self.assertEqual(model.device, torch.device("cpu"))

    def test_internal_controls_cannot_be_passed_as_parameters(self):
        for task in ("binary", "classification", "regression", "multiregression"):
            for name, value in (("solver", "solve"), ("classification_mode", "prevalence")):
                with self.subTest(task=task, parameter=name):
                    with self.assertRaisesRegex(ValueError, "unrecognized.*" + name):
                        self.model_class.resolve_parameters({name: value}, self.args(task))


if __name__ == "__main__":
    unittest.main()
