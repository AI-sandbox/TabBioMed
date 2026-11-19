import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import optuna
import torch

from tabbiomed.data import catalog
from tabbiomed.models import get_search_space
from tabbiomed.models.model_definitions.torch_pca import TorchPCA
from tabbiomed.profiles import (
    PCA_MODELS,
    PCA_SEARCH_SPACES,
    default_output,
    profile_config,
)
from tabbiomed.results.manifest import create_manifest
from tabbiomed.slurm import prepare, run_task, submit
from tabbiomed.tabbiomed_datasets import TabularDataset
from tabbiomed.trainer.train import CONFIG, run_dataset
from tabbiomed.trainer.utils.parameter_config import (
    define_trial_parameters_from_config,
    load_model_search_space,
    merge_default_params_with_search_space,
)
from tabbiomed.utils import load_yaml_config


class PCAProfile(unittest.TestCase):
    def test_profile_preserves_budgets_and_separates_outputs(self):
        main = load_yaml_config(CONFIG)
        selected = profile_config(main, "pca500")
        self.assertEqual(profile_config(main, "main"), main)
        self.assertNotIn("search_space_dir", main)
        self.assertEqual(selected.pop("search_space_dir"), str(PCA_SEARCH_SPACES))
        self.assertEqual(selected, main)
        self.assertEqual(default_output("pca500"), Path("results/raw_pca500"))
        self.assertEqual(default_output("main"), Path("results/raw"))

    def test_default_and_sampled_configurations_share_fixed_pca(self):
        for model in PCA_MODELS:
            with self.subTest(model=model):
                main = get_search_space(model)
                space = load_model_search_space(model, PCA_SEARCH_SPACES)
                self.assertTrue(all(space[name] == spec for name, spec in main.items()))
                defaults = merge_default_params_with_search_space(
                    {"apply_pca": False, "pca_max_features": 2000},
                    model,
                    config_dir=PCA_SEARCH_SPACES,
                )
                values = {
                    name: spec["choices"][0]
                    if "choices" in spec
                    else float(spec["low"])
                    for name, spec in space.items()
                }
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Choices for a categorical distribution.*"
                    )
                    sampled = define_trial_parameters_from_config(
                        optuna.trial.FixedTrial(values),
                        model,
                        config_dir=PCA_SEARCH_SPACES,
                        dataset_metadata={"objective": "binary", "use_gpu": True},
                    )
                for parameters in (defaults, sampled):
                    self.assertIs(parameters["apply_pca"], True)
                    self.assertEqual(parameters["pca_max_features"], 500)
                    self.assertEqual(parameters["pca_fit"], "lowrank")
                if model == "TabICLv2":
                    for parameters in (defaults, sampled):
                        self.assertIs(parameters["pca_impute_missing"], True)
                        self.assertEqual(parameters["pca_imputation_strategy"], "mean")

    def test_full_plan_has_108_complete_pairs_and_forwards_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("tabbiomed.slurm.subprocess.check_output") as sbatch:
                scripts = prepare(
                    root, profile="pca500", gpu_partition="gpu", gpu_type="a5500"
                )
                submit(root)
                sbatch.assert_not_called()
            plan = json.loads((root / "manifest.json").read_text())
            tasks = plan["tasks"]
            self.assertEqual(len(tasks), 108)
            self.assertEqual(len({t["dataset"] for t in tasks}), 36)
            self.assertEqual({t["model"] for t in tasks}, set(PCA_MODELS))
            reference = catalog()
            self.assertTrue(
                all(
                    reference[t["dataset"]]["metadata"]["num_features"] > 2000
                    for t in tasks
                )
            )
            self.assertEqual(len(scripts), 1)
            script = scripts[0].read_text()
            for value in (
                "--time=24:00:00",
                "--cpus-per-task=10",
                "--mem=128G",
                "--gres=gpu:a5500:1",
            ):
                self.assertIn(value, script)
            self.assertEqual(Path(plan["output"]).name, "raw_pca500")
            with patch("tabbiomed.slurm.subprocess.run") as run:
                run_task(root / "manifest.json", 0)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--profile") + 1], "pca500")

    def test_profile_rejects_unmatched_models_and_small_datasets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "PCA-500 supports only"):
                prepare(root / "model", profile="pca500", models=["RandomForest"])
            with self.assertRaisesRegex(ValueError, "more than 2,000 features"):
                prepare(root / "data", profile="pca500", datasets=["pima_diabetes"])
            self.assertEqual(list(root.iterdir()), [])

    def test_pca_uses_training_means_and_limits_actual_components(self):
        rng = np.random.default_rng(42)
        for rows, active in ((8, 10), (12, 3)):
            with self.subTest(rows=rows, active=active):
                x = np.zeros((rows, 2005), dtype=np.float32)
                x[:, :active] = rng.normal(size=(rows, active))
                x[0, 0] = np.nan
                x[:, -1] = np.nan
                pca = TorchPCA(n_components=500, fit="lowrank")
                transformed = pca.fit_transform_array(x, "cpu")
                self.assertEqual(transformed.shape, (rows, min(rows, active)))
                self.assertTrue(np.isfinite(transformed).all())
                self.assertEqual(pca.nonfinite_fill_[-1].item(), 0.0)
                self.assertAlmostEqual(
                    pca.nonfinite_fill_[0].item(), np.nanmean(x[:, 0]), places=6
                )
                means = pca.nonfinite_fill_.clone()
                held = np.full((3, 2005), 100.0, dtype=np.float32)
                held[:, -1] = np.nan
                self.assertTrue(np.isfinite(pca.transform_array(held)).all())
                self.assertTrue(torch.equal(means, pca.nonfinite_fill_))

    def test_runner_passes_pca_settings_and_records_export_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            x = np.zeros((40, 2005), dtype=np.float32)
            y = np.tile([0, 1], 20)
            splits = dict(
                train=list(range(24)), val=list(range(24, 32)), test=list(range(32, 40))
            )
            metadata = dict(task="binary", num_features=2005)
            for name, value in (("X", x), ("y", y), ("split_indices", splits)):
                np.save(data / f"{name}.npy", value)
            (data / "metadata.json").write_text(json.dumps(metadata))
            dataset = TabularDataset(
                name="ALLAML", X=x, y=y, cat_idx=[], task="binary", split_indices=splits
            )
            with patch("tabbiomed.trainer.selection.run_experiment") as evaluate:
                directory = run_dataset(
                    dataset,
                    metadata,
                    model="TabPFNv2.5",
                    data_directory=data,
                    output=root / "raw",
                    profile="pca500",
                )
            configuration = evaluate.call_args.kwargs["config"]
            space = load_model_search_space(
                "TabPFNv2.5", configuration["search_space_dir"]
            )
            self.assertEqual(space["pca_max_features"]["choices"], [500])
            self.assertEqual(space["apply_pca"]["choices"], [True])
            report = json.loads((directory / "report.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(report["scope"], "pca_500")
            self.assertEqual(manifest["evaluations"][0]["scope"], "pca_500")
            self.assertEqual(report["input_identity"]["search_space"], space)

    def test_collector_preserves_main_and_pca_scopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, scope in (("run_main", None), ("run_pca", "pca_500")):
                directory = root / "raw/TabICLv2/ALLAML" / name
                directory.mkdir(parents=True)
                report = dict(
                    model="TabICLv2",
                    dataset="ALLAML",
                    status="complete",
                    input_identity={"data_root": str(root / "data/ALLAML")},
                )
                if scope:
                    report["scope"] = scope
                (directory / "report.json").write_text(json.dumps(report))
                (directory / ".run.lock").touch()
            manifest = create_manifest(root / "raw", root / "runs.json")
            entries = json.loads(manifest.read_text())["evaluations"]
            self.assertEqual(
                {e.get("scope", "main") for e in entries}, {"main", "pca_500"}
            )


if __name__ == "__main__":
    unittest.main()
