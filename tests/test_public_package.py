from importlib.resources import files
import unittest

import yaml

from tabbiomed.data import dataset_info, dataset_names
from tabbiomed.models import BENCHMARK_MODELS, get_model_spec


class PublicPackage(unittest.TestCase):
    def test_paper_scope_and_selected_collection_endpoints(self):
        names = dataset_names()
        self.assertEqual(len(names), 94)
        self.assertEqual(len(set(names)), 94)
        self.assertEqual(
            {n for n in names if n.startswith("SIDER_")},
            {"SIDER_gastro", "SIDER_nervous"},
        )
        for name in names:
            metadata = dataset_info(name)["metadata"]
            self.assertGreater(metadata["num_samples"], 0)
            self.assertGreater(metadata["num_features"], 0)
        with self.assertRaises(ValueError):
            dataset_info("not_a_benchmark_task")

    def test_packaged_selection_budgets(self):
        path = files("tabbiomed").joinpath("configs/benchmark.yaml")
        config = yaml.safe_load(path.read_text())
        self.assertEqual(config["n_trials"], 100)
        self.assertEqual(config["trajectory_n_trials"], 200)
        self.assertEqual(config["hpo_evaluation_budgets"], [25, 50, 100, 200])
        self.assertEqual(config["time_limit"], 21600)
        self.assertEqual(config["cv_time_limit_per_fold"], 7200)
        self.assertEqual(config["cv_outer_folds"], 5)
        self.assertEqual(config["cv_inner_folds"], 3)

    def test_builtin_model_roster_and_task_support(self):
        self.assertTrue(BENCHMARK_MODELS)
        for name in BENCHMARK_MODELS:
            spec = get_model_spec(name)
            self.assertIn("binary", spec.tasks)
            self.assertIn("classification", spec.tasks)
            if name in {"HyperFast", "TabPFN-Wide", "TabOmicsv2.5x1", "TabOmicsv2.5x8"}:
                self.assertNotIn("regression", spec.tasks)
            else:
                self.assertIn("regression", spec.tasks)
        for name in ("LinearModel", "NeoLsSVM"):
            self.assertFalse(get_model_spec(name).hpo_enabled)


if __name__ == "__main__":
    unittest.main()
