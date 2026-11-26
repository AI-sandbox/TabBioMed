import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tabbiomed.data import dataset_names
from tabbiomed.models import get_model_spec, get_search_space, model_name_map
from tabbiomed.models import tabpfn35 as module
from tabbiomed.slurm import prepare


class NativeEstimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def fit(self, X, y):
        self.training_rows = len(X)
        self.classes_ = np.unique(y)
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        self.calls.append(len(X))
        return np.full(len(X), self.mean)

    def predict_proba(self, X):
        self.calls.append(len(X))
        return np.tile([0.25, 0.75], (len(X), 1))


class TabPFN35Workflow(unittest.TestCase):
    def test_registration_and_slurm_selection(self):
        name = model_name_map("tabpfn35")
        self.assertEqual(name, "TabPFN-3.5")
        spec = get_model_spec(name)
        self.assertFalse(spec.hpo_enabled)
        self.assertTrue(spec.gpu)
        self.assertTrue(spec.requires_separate_env)
        self.assertEqual(get_search_space(name), {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(root / "implicit", datasets=["pima_diabetes"])
            implicit = json.loads((root / "implicit" / "manifest.json").read_text())
            self.assertNotIn(name, {task["model"] for task in implicit["tasks"]})
            prepare(
                root / "explicit", models=[name], model_module="tabbiomed.models.tabpfn35"
            )
            explicit = json.loads((root / "explicit" / "manifest.json").read_text())
            self.assertEqual(
                {(task["model"], task["dataset"], task["gpu"]) for task in explicit["tasks"]},
                {(name, dataset, True) for dataset in dataset_names()},
            )

    def test_native_defaults_seed_categoricals_and_prediction_batches(self):
        args = SimpleNamespace(
            objective="binary",
            use_gpu=False,
            gpu_ids=[],
            seed=42,
            cat_idx=np.array([1]),
            num_cpus=1,
        )
        params = {**module.TabPFN35.get_default_params(), "inference_batch_size": 2}
        fake = SimpleNamespace(
            TabPFNClassifier=NativeEstimator, TabPFNRegressor=NativeEstimator
        )
        with (
            patch.dict(sys.modules, tabpfn=fake),
            patch.object(
                module,
                "resolve_checkpoint",
                return_value=Path("weights.safetensors"),
            ),
        ):
            model = module.TabPFN35(
                module.TabPFN35.resolve_parameters(params, args), args
            )
            model.fit(
                np.ones((8, 2)), np.arange(8) % 2, np.ones((2, 2)), np.zeros(2)
            )
            self.assertEqual(model.model.training_rows, 8)
            self.assertEqual(model.model.kwargs["random_state"], 42)
            self.assertEqual(
                model.model.kwargs["categorical_features_indices"], [1]
            )
            self.assertNotIn("n_estimators", model.model.kwargs)
            np.testing.assert_array_equal(
                model.predict(np.ones((5, 2))), np.ones(5)
            )
            self.assertEqual(model.model.calls, [2, 2, 1])
            args.objective = "classification"
            args.num_classes = 3
            model.fit(np.ones((8, 2)), np.tile([0, 2], 4))
            np.testing.assert_array_equal(
                model.predict_proba(np.ones((1, 2))), [[0.25, 0.0, 0.75]]
            )
            args.objective = "multiregression"
            args.num_outputs = 2
            multi = module.TabPFN35(
                module.TabPFN35.resolve_parameters(params, args), args
            )
            multi.fit_multioutput(np.ones((8, 2)), np.tile([1.0, 3.0], (8, 1)))
            np.testing.assert_allclose(
                multi.predict(np.ones((5, 2))), np.tile([1.0, 3.0], (5, 1))
            )


if __name__ == "__main__":
    unittest.main()
