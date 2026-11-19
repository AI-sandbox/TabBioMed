import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tabbiomed.evaluator.metrics import compute_metrics
from tabbiomed.results.consolidation import consolidate
from tabbiomed.results.hpo import prepare_history
from tabbiomed.results.io import digest, write_json


class PredictionValidation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "predictions.pt"
        self.labels = self.root / "labels.npy"

    def validate(
        self,
        labels,
        payload,
        *,
        task="regression",
        num_classes=1,
        metric="mse",
        score=0.0,
    ):
        from tabbiomed.results.predictions import validate_predictions

        np.save(self.labels, labels)
        torch.save(payload, self.path)
        return validate_predictions(
            self.path,
            self.labels,
            dict(task=task, num_classes=num_classes),
            {metric: score},
            metric,
        )

    def test_numpy_tensor_and_single_output_column_predictions_are_valid(self):
        labels = np.array([1.0, 2.0])
        for predictions in (
            labels.copy(),
            labels.reshape(-1, 1),
            torch.tensor(labels, requires_grad=True),
        ):
            with self.subTest(predictions=predictions):
                self.validate(labels, dict(y_pred=predictions, y_proba=None))

    def test_multioutput_predictions_require_the_actual_output_shape(self):
        labels = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.validate(
            labels, dict(y_pred=labels - 1), task="multiregression", score=1.0
        )
        for predictions in (
            labels.ravel(),
            labels[:, :1],
            labels.T.reshape(1, -1),
            labels[:, :, None],
        ):
            with self.subTest(shape=predictions.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    self.validate(
                        labels, dict(y_pred=predictions), task="multiregression"
                    )

    def test_corrupt_and_malformed_payloads_are_rejected(self):
        from tabbiomed.results.predictions import validate_predictions

        np.save(self.labels, [1.0, 2.0])
        self.path.write_bytes(b"not a Torch prediction file")
        with self.assertRaisesRegex(ValueError, "read"):
            validate_predictions(
                self.path, self.labels, {"task": "regression"}, {"mse": 0.0}, "mse"
            )
        for payload in (
            [1.0, 2.0],
            {},
            dict(y_pred=None),
            dict(y_pred=["one", "two"]),
            dict(y_pred=[1.0, np.nan]),
            dict(y_pred=[1.0, np.inf]),
            dict(y_pred=[1.0, 2.0, 3.0]),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.validate(np.array([1.0, 2.0]), payload)

    def test_classification_accepts_full_probabilities_when_test_lacks_a_class(self):
        labels = np.array([0, 1])
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                probabilities = np.array(
                    [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=dtype
                )
                payload = dict(y_pred=labels.copy(), y_proba=probabilities)
                score = compute_metrics(
                    labels, labels, probabilities, "classification"
                )["log_loss"]
                self.validate(
                    labels,
                    payload,
                    task="classification",
                    num_classes=3,
                    metric="log_loss",
                    score=score,
                )

    def test_invalid_class_labels_and_probability_contents_are_rejected(self):
        labels = np.array([0, 1])
        base = dict(y_pred=labels.copy(), y_proba=np.array([[0.9, 0.1], [0.1, 0.9]]))
        cases = []
        for predictions in ([-1, 1], [0, 2], [0.5, 1]):
            cases.append({**base, "y_pred": predictions})
        for probabilities in (
            None,
            [0.1, 0.9],
            [[1.0], [1.0]],
            [[0.1, 0.9, 0.0], [0.1, 0.9, 0.0]],
            [[0.9, 0.1]],
            [[np.nan, 0.1], [0.1, 0.9]],
            [[np.inf, 0.1], [0.1, 0.9]],
            [[1.1, -0.1], [0.1, 0.9]],
            [[0.2, 0.2], [0.1, 0.9]],
        ):
            cases.append({**base, "y_proba": probabilities})
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.validate(
                        labels,
                        payload,
                        task="binary",
                        num_classes=2,
                        metric="roc_auc",
                        score=1.0,
                    )

    def test_primary_metric_must_agree_with_saved_predictions(self):
        for task, classes, metric, labels, payload, wrong_score in (
            ("regression", 1, "mse", [1.0, 2.0], dict(y_pred=[1.0, 2.0]), 9.0),
            (
                "binary",
                2,
                "roc_auc",
                [0, 1],
                dict(y_pred=[0, 1], y_proba=[[0.9, 0.1], [0.1, 0.9]]),
                0.0,
            ),
        ):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "metric"):
                    self.validate(
                        np.asarray(labels),
                        payload,
                        task=task,
                        num_classes=classes,
                        metric=metric,
                        score=wrong_score,
                    )


class ConsolidatedPredictionValidation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.run = self.root / "raw"
        self.data.mkdir()
        self.run.mkdir()
        (self.run / ".run.lock").touch()
        np.save(self.data / "X.npy", np.arange(6).reshape(-1, 1))
        np.save(self.data / "y.npy", np.arange(6, dtype=float))
        self.metadata = dict(
            name="example", task="regression", num_samples=6, num_features=1, cat_idx=[]
        )
        (self.data / "metadata.json").write_text(json.dumps(self.metadata))
        np.save(
            self.data / "split_indices.npy", dict(train=[0, 1, 2], val=[3], test=[4, 5])
        )
        np.save(self.run / "test_labels_fixed.npy", [4.0, 5.0])
        self.predictions = dict(y_pred=np.array([4.0, 5.0]), y_proba=None)
        self.prediction_path = self.run / "predictions_default.pt"
        torch.save(self.predictions, self.prediction_path)
        self.report = dict(
            model="LinearModel",
            dataset="example",
            seed=42,
            status="complete",
            hpo_enabled=False,
            primary_metric="mse",
            config={},
            default=dict(
                parameters={},
                test_metrics={"mse": 0.0},
                fixed_split_test_evaluated=True,
                train_val_merged_for_fit=True,
                train_wall_time_seconds=1.0,
            ),
        )
        files = {
            name: dict(
                size=(self.data / name).stat().st_size, sha256=digest(self.data / name)
            )
            for name in ("X.npy", "y.npy", "metadata.json", "split_indices.npy")
        }
        self.report["input_identity"] = dict(
            model="LinearModel",
            dataset="example",
            seed=42,
            data_root=str(self.data),
            data_files=files,
            search_space={},
        )
        (self.run / "report.json").write_text(json.dumps(self.report))
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    evaluations=[dict(report="raw/report.json", data_root="data")],
                )
            )
        )

    def test_direct_manifest_rejects_an_active_run(self):
        with (self.run / ".run.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for execute in (False, True):
                with self.subTest(execute=execute):
                    output = self.root / "results"
                    with self.assertRaisesRegex(ValueError, "still active"):
                        consolidate(self.manifest, output, execute=execute)
                    self.assertFalse(output.exists())

    def test_source_run_stays_locked_until_installation_finishes(self):
        from tabbiomed.results.consolidation import install_directory

        checked = []

        def install_locked(source, destination):
            with (self.run / ".run.lock").open() as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checked.append(destination)
            install_directory(source, destination)

        with patch(
            "tabbiomed.results.consolidation.install_directory",
            side_effect=install_locked,
        ):
            receipt = consolidate(self.manifest, self.root / "results", execute=True)
        self.assertEqual(receipt["status"], "installed")
        self.assertEqual(len(checked), 1)
        with (self.run / ".run.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_source_lock_is_released_after_validation_failure(self):
        self.prediction_path.write_bytes(b"not predictions")
        with self.assertRaises(ValueError):
            consolidate(self.manifest, self.root / "results", execute=True)
        with (self.run / ".run.lock").open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_source_without_a_run_lock_is_rejected(self):
        (self.run / ".run.lock").unlink()
        with self.assertRaisesRegex(ValueError, "Expected a benchmark run lock"):
            consolidate(self.manifest, self.root / "results")

    def test_consolidation_rejects_corrupt_files_before_installation(self):
        self.prediction_path.write_bytes(b"not predictions")
        output = self.root / "results"
        with self.assertRaises(ValueError):
            consolidate(self.manifest, output, execute=True)
        self.assertFalse(output.exists())

    def test_consolidation_rejects_wrong_rows_and_inconsistent_metrics(self):
        for predictions in ([4.0], [5.0, 4.0], [4.0, np.nan]):
            with self.subTest(predictions=predictions):
                torch.save(
                    dict(y_pred=np.array(predictions), y_proba=None),
                    self.prediction_path,
                )
                output = self.root / "results"
                with self.assertRaises(ValueError):
                    consolidate(self.manifest, output, execute=True)
                self.assertFalse(output.exists())

    def test_valid_predictions_are_copied_without_modification(self):
        before = self.prediction_path.read_bytes()
        receipt = consolidate(self.manifest, self.root / "results", execute=True)
        self.assertEqual(receipt["status"], "installed")
        self.assertEqual(
            (
                self.root / "results/LinearModel/example/predictions_default.pt"
            ).read_bytes(),
            before,
        )

    def test_incomplete_test_evaluations_are_not_published_as_successes(self):
        from tabbiomed.analysis.results import read_result

        for index, changes in enumerate((
            {}, {"fixed_split_test_evaluated": False},
            {"fixed_split_test_evaluated": True, "failure": {"reason": "gpu_oom"}},
        )):
            with self.subTest(changes=changes):
                source = json.loads(json.dumps(self.report))
                source["default"].pop("fixed_split_test_evaluated")
                source["default"].update(changes)
                (self.run / "report.json").write_text(json.dumps(source))
                output = self.root / f"results_{index}"
                consolidate(self.manifest, output, execute=True)
                report = json.loads((output / "LinearModel/example/report.json").read_text())
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["default"]["test_metrics"], {})
                row = read_result("LinearModel", "example", {"task": "regression", "n_folds": 0}, root=output)
                self.assertIsNone(row["metric_value"])
                self.assertEqual(row["metric_source"], "failed")

    def test_hpo_export_rejects_corrupt_budget_predictions(self):
        public = self.root / "public"
        public.mkdir()
        np.save(public / "test_labels_fixed.npy", [4.0, 5.0])
        torch.save(self.predictions, public / "predictions_best.pt")
        torch.save(self.predictions, self.run / "predictions_best.pt")
        best = dict(
            trial_id=0,
            parameters={"width": 1},
            test_metrics={"mse": 0.0},
            fixed_split_test_evaluated=True,
            predictions="predictions_best.pt",
        )
        main = dict(
            model="LinearModel",
            dataset="example",
            seed=42,
            hpo_enabled=True,
            primary_metric="mse",
            best=best,
        )
        write_json(self.run / "report.json", main)
        report_sha = write_json(public / "report.json", main)
        hpo = dict(
            official_attempt_cap=1,
            attempted_configurations=1,
            budget_seconds=10.0,
            search_cutoff_seconds=10.0,
        )
        write_json(
            public / "timing.json",
            dict(
                report_sha256=report_sha,
                dataset_split_sha256="splits",
                units=[dict(fold_id=None, hpo=hpo)],
            ),
        )
        events = [
            dict(
                event="search_started",
                search_id="s",
                fold_id=None,
                hpo_elapsed_seconds=0.0,
            )
        ]
        for tid, score in ((0, 0.1), (1, 0.05)):
            common = dict(search_id="s", fold_id=None, trial_id=tid)
            events.append(
                dict(common, event="trial_started", hpo_elapsed_seconds=float(tid))
            )
            events.append(
                dict(
                    common,
                    event="trial_finished",
                    hpo_elapsed_seconds=float(tid + 1),
                    status="success",
                    parameters={"width": tid + 1},
                    validation_metrics={"mse": score},
                )
            )
        (self.run / "hpo_trials.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
        reference = self.run / "selection/fixed/reference.json"
        reference.parent.mkdir(parents=True)
        write_json(
            reference,
            dict(status="failed", validation_seconds=0.0, parameters={"width": 3}),
        )
        (self.run / "budget.pt").write_bytes(b"corrupt budget predictions")
        write_json(
            self.run / "hpo_budget_results.json",
            dict(
                search_id="s",
                budgets=[
                    dict(
                        status="success",
                        selected_trial_id=1,
                        parameters={"width": 2},
                        test_metrics={"mse": 0.0},
                        predictions="budget.pt",
                    )
                ],
            ),
        )
        with self.assertRaisesRegex(ValueError, "read"):
            prepare_history(
                public / "report.json",
                self.run / "report.json",
                split_sha256="splits",
                metadata=self.metadata,
            )


if __name__ == "__main__":
    unittest.main()
