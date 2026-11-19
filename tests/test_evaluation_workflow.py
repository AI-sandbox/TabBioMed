import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyRegressor

from tabbiomed.models import register_model
from tabbiomed.models.base_model import BaseModel
from tabbiomed.tabbiomed_datasets import TabularDataset
from tabbiomed.trainer.train import run_dataset, CONFIG
from tabbiomed.utils import load_yaml_config
from tabbiomed.results.consolidation import consolidate
from tabbiomed.results.hpo_read import read_history


class ConstantModel(BaseModel):
    required_parameters = ("constant",)

    @classmethod
    def get_base_params(cls, args):
        return {"offset": 0.0}

    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = DummyRegressor(
            strategy="constant", constant=params["constant"] + params["offset"]
        )

    @classmethod
    def get_default_params(cls):
        return {"constant": 10.0}

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


class CrashingConstantModel(ConstantModel):
    def fit(self, X, y, X_val=None, y_val=None):
        if self.params["constant"] == 1.0:
            import os
            import resource
            import signal

            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.kill(os.getpid(), signal.SIGSEGV)
        return super().fit(X, y, X_val, y_val)


register_model(
    "WorkflowConstant",
    ConstantModel,
    search_space={"constant": {"type": "categorical", "choices": [1.0, 2.0, 3.0]}},
    tasks=("regression",),
)


register_model(
    "WorkflowCrash",
    CrashingConstantModel,
    search_space={"constant": {"type": "categorical", "choices": [1.0, 2.0, 3.0]}},
    tasks=("regression",),
)


register_model(
    "WorkflowConditional",
    ConstantModel,
    search_space={
        "constant": {"type": "categorical", "choices": [1.0, 2.0, 3.0]},
        "offset": {
            "type": "conditional",
            "depends_on": "constant",
            1.0: 0.0,
            2.0: 0.0,
            3.0: 0.0,
        },
    },
    tasks=("regression",),
)


class EvaluationWorkflow(unittest.TestCase):
    def make_dataset(self, root, cv=False):
        name = "workflow_data"
        directory = root / name
        directory.mkdir()
        x = np.arange(120, dtype=float).reshape(40, 3)
        y = np.full(40, 2.0)
        metadata = dict(
            name=name,
            task="regression",
            num_features=3,
            num_samples=40,
            cat_idx=[],
            READY=True,
        )
        if cv:
            metadata["split_strategy"] = "cv"
            folds = []
            for i in range(5):
                test = list(range(8 * i, 8 * i + 8))
                pool = [j for j in range(40) if j not in test]
                folds.append(dict(train=pool[:24], val=pool[24:], test=test))
            splits = dict(cv_folds=folds, groups=np.repeat(np.arange(20), 2).tolist())
        else:
            splits = dict(
                train=list(range(24)), val=list(range(24, 32)), test=list(range(32, 40))
            )
        np.save(directory / "X.npy", x)
        np.save(directory / "y.npy", y)
        np.save(directory / "split_indices.npy", splits)
        (directory / "metadata.json").write_text(json.dumps(metadata))
        dataset = TabularDataset(
            name=name,
            X=x,
            y=y,
            cat_idx=[],
            task="regression",
            num_classes=None,
            split_indices=splits,
            groups=np.asarray(splits["groups"]) if cv else None,
        )
        return dataset, metadata, directory

    def check_default_workflow(self, cv):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, metadata, data = self.make_dataset(root, cv)
            X = original.X.astype(object)
            X[:, 1] = np.tile(["a", "b", "c", "d"], 10)
            metadata["cat_idx"] = [1]
            np.save(data / "X.npy", X)
            (data / "metadata.json").write_text(json.dumps(metadata))
            dataset = TabularDataset(
                name=original.name,
                X=X,
                y=original.y,
                cat_idx=[1],
                task="regression",
                num_features=3,
                num_samples=40,
                split_indices=original.split_indices,
                groups=original.groups,
            )
            preparations = []
            prepare = TabularDataset.prepare_dataset

            def record_preparation(ds, *args, **kwargs):
                self.assertIsNone(ds.cat_dims)
                result = prepare(ds, *args, **kwargs)
                preparations.append(ds.X_train.shape[1])
                return result

            with patch.object(TabularDataset, "prepare_dataset", record_preparation):
                directory = run_dataset(
                    dataset,
                    metadata,
                    model="LinearModel",
                    data_directory=data,
                    output=root / "raw",
                )
            count = 5 if cv else 1
            self.assertEqual(preparations, [6] * count)
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["hpo_enabled"])
            units = report["cv_fold_metrics"] if cv else [report]
            self.assertEqual(len(units), count)
            for unit in units:
                self.assertTrue(unit["default"]["fixed_split_test_evaluated"])
                self.assertTrue(np.isfinite(unit["default"]["test_metrics"]["mse"]))
                self.assertIsNone(unit["best"])
            public = root / "results"
            receipt = consolidate(directory / "manifest.json", public, execute=True)
            self.assertEqual(receipt["status"], "installed")
            pair = public / "LinearModel" / dataset.name
            self.assertEqual(len(list(pair.glob("predictions_*default.pt"))), count)
            self.assertTrue((pair / "timing.json").is_file())

    def check_workflow(self, cv, model="WorkflowConstant"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ds, metadata, data = self.make_dataset(root, cv)
            config = load_yaml_config(CONFIG)
            config.update(
                n_trials=2,
                trajectory_n_trials=3,
                hpo_evaluation_budgets=[25, 50, 100, 200],
                time_limit=60,
                cv_time_limit_per_fold=60,
            )
            directory = run_dataset(
                ds,
                metadata,
                model=model,
                data_directory=data,
                output=root / "raw",
                config=config,
            )
            report = json.loads((directory / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            units = report["cv_fold_metrics"] if cv else [report]
            self.assertEqual(len(units), 5 if cv else 1)
            for unit in units:
                self.assertTrue(unit["default"]["fixed_split_test_evaluated"])
                self.assertTrue(unit["best"]["fixed_split_test_evaluated"])
                self.assertEqual(unit["best"]["parameters"]["constant"], 2.0)
                for variant in ("default", "best"):
                    self.assertEqual(unit[variant]["parameters"]["offset"], 0.0)
                    self.assertEqual(unit[variant]["parameters"]["scaler"], "standard")
                self.assertGreater(
                    unit["default"]["test_metrics"]["mse"],
                    unit["best"]["test_metrics"]["mse"],
                )
                if cv:
                    self.assertEqual(unit["hpo_validation"]["actual_folds"], 3)
                    for inner in unit["hpo_validation"]["folds"]:
                        train_groups = set(ds.groups[inner["train"]])
                        val_groups = set(ds.groups[inner["val"]])
                        self.assertFalse(train_groups & val_groups)

            public = root / "results"
            receipt = consolidate(directory / "manifest.json", public, execute=True)
            self.assertEqual(receipt["status"], "installed")
            pair = public / model / ds.name
            exported = json.loads((pair / "report.json").read_text())
            self.assertEqual(exported["artifact_type"], "evaluation_summary")
            self.assertTrue((pair / "timing.json").is_file())
            from tabbiomed.data.identity import split_digest
            from tabbiomed.analysis.results import valid_split_identity

            timing = json.loads((pair / "timing.json").read_text())
            fingerprint = split_digest(ds.split_indices, ds.num_samples, 5 if cv else 0)
            self.assertEqual(timing["dataset_split_sha256"], fingerprint)
            self.assertTrue(
                valid_split_identity(timing, {"splits_sha256": fingerprint})
            )
            self.assertTrue((pair / "hpo/trials.jsonl").is_file())
            self.assertTrue((pair / "hpo/selections.json").is_file())
            official, budgets, _, errors = read_history(
                public, model, ds.name, "regression"
            )
            self.assertFalse(errors)
            self.assertEqual(len(official), len(units))
            self.assertEqual(len(budgets), 4 * len(units))
            for part in pair.rglob("*.json"):
                self.assertNotIn(str(root), part.read_text())
            self.assertEqual(len(list(pair.glob("predictions_*best.pt"))), len(units))
            if model == "WorkflowCrash":
                events = [
                    json.loads(line)
                    for line in (directory / "hpo_trials.jsonl")
                    .read_text()
                    .splitlines()
                ]
                trials = [e for e in events if e["event"] == "trial_finished"]
                self.assertEqual(len(trials), 3)
                self.assertIn("SIGSEGV", trials[1]["error"])
                self.assertEqual(trials[2]["status"], "success")

    def test_registered_model_fixed_split_and_consolidation(self):
        self.check_workflow(False)

    def test_conditional_search_space_identity_and_consolidation(self):
        self.check_workflow(False, model="WorkflowConditional")

    def test_registered_model_nested_cv_and_consolidation(self):
        self.check_workflow(True)

    def test_native_trial_crash_continues_and_retains_test_results(self):
        self.check_workflow(False, model="WorkflowCrash")

    def test_initial_report_is_written_under_the_run_lock(self):
        from tabbiomed.trainer.train import dump_report

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ds, metadata, data = self.make_dataset(root)
            checked = []

            def write_locked(report, directory):
                with (directory / ".run.lock").open() as lock:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                checked.append(directory)
                dump_report(report, directory)

            with (
                patch("tabbiomed.trainer.train.dump_report", side_effect=write_locked),
                patch(
                    "tabbiomed.trainer.selection.run_experiment",
                    side_effect=RuntimeError("controller interrupted"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "controller interrupted"):
                    run_dataset(
                        ds, metadata, model="WorkflowConstant",
                        data_directory=data, output=root / "raw",
                    )
            self.assertEqual(len(checked), 1)

    def test_interrupted_cv_can_publish_explicit_time_limit_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ds, metadata, data = self.make_dataset(root, cv=True)
            with patch(
                "tabbiomed.trainer.selection.run_experiment",
                side_effect=RuntimeError("controller interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "controller interrupted"):
                    run_dataset(
                        ds,
                        metadata,
                        model="WorkflowConstant",
                        data_directory=data,
                        output=root / "raw",
                    )
            run = next((root / "raw").glob("*/*/run_*"))
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["evaluations"][0]["failures"] = {
                name: {"reason": "time_limit"} for name in ("default", "best")
            }
            manifest_path.write_text(json.dumps(manifest))
            consolidate(manifest_path, root / "public", root / "logs", execute=True)
            exported = json.loads(
                (root / "public/WorkflowConstant/workflow_data/report.json").read_text()
            )
            self.assertEqual(len(exported["cv_fold_metrics"]), 5)
            for unit in exported["cv_fold_metrics"]:
                for variant in ("default", "best"):
                    self.assertEqual(unit[variant]["failure"], {"reason": "time_limit"})
            self.assertFalse(exported["cv_aggregated_metrics"]["best"]["test"])

    def test_default_only_categorical_cv(self):
        self.check_default_workflow(cv=True)

    def test_default_only_categorical_fixed_split(self):
        self.check_default_workflow(cv=False)


if __name__ == "__main__":
    unittest.main()
