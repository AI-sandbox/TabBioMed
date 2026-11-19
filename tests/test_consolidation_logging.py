import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tabbiomed.results.consolidation import consolidate, Evaluation
from tabbiomed.results.sources import Evidence, validate_labels
from tabbiomed.results.selection import inferred_failure
from tabbiomed.results.hpo import normalized_trial


class ConsolidationLogging(unittest.TestCase):
    def fixture(self, root, manifest_name="manifest.json"):
        manifest = root / manifest_name
        manifest.write_text("{}\n")
        evidence = Evidence()
        evidence.pin(manifest)
        evaluation = Evaluation(
            destination=Path("LinearModel/example"),
            report={"status": "complete", "hpo_enabled": False},
            timing={},
            artifacts={},
            evidence=evidence,
            identity={},
            metadata={},
        )
        return manifest, evaluation

    def test_default_writes_only_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evaluation = self.fixture(root)
            output = root / "results"
            with patch(
                "tabbiomed.results.consolidation.prepare_manifest",
                return_value=[evaluation],
            ):
                receipt = consolidate(manifest, output)
                self.assertEqual(receipt["status"], "validated")
                self.assertFalse(output.exists())
                receipt = consolidate(manifest, output, execute=True)
            self.assertEqual(receipt["status"], "installed")
            self.assertTrue((output / "LinearModel/example/report.json").is_file())
            self.assertTrue((output / "LinearModel/example/timing.json").is_file())
            for name in ("preflight.json", "prepared.json", "installed.json"):
                self.assertFalse(list(root.rglob(name)))

    def test_optional_logs_can_be_inside_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evaluation = self.fixture(root)
            output = root / "results"
            logs = output / ".logs"
            with patch(
                "tabbiomed.results.consolidation.prepare_manifest",
                return_value=[evaluation],
            ):
                consolidate(manifest, output, logs)
                consolidate(manifest, output, logs, execute=True)
            for name in ("preflight.json", "prepared.json", "installed.json"):
                self.assertTrue((logs / name).is_file())
            installed = json.loads((logs / "installed.json").read_text())
            self.assertEqual(installed["status"], "installed")

    def test_logs_cannot_overlap_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evaluation = self.fixture(root)
            output = root / "results"
            with patch(
                "tabbiomed.results.consolidation.prepare_manifest",
                return_value=[evaluation],
            ):
                with self.assertRaisesRegex(ValueError, "overlap evaluation"):
                    consolidate(manifest, output, output / evaluation.destination)
            self.assertFalse(output.exists())

    def test_logs_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, evaluation = self.fixture(root, "preflight.json")
            before = manifest.read_bytes()
            with patch(
                "tabbiomed.results.consolidation.prepare_manifest",
                return_value=[evaluation],
            ):
                with self.assertRaisesRegex(ValueError, "overwrite inputs"):
                    consolidate(manifest, root / "results", root)
            self.assertEqual(manifest.read_bytes(), before)


class TestEvaluationAvailability(unittest.TestCase):
    def test_consolidation_history_and_analysis_agree_on_success(self):
        from tabbiomed.results.selection import test_available
        from tabbiomed.results.hpo import valid_test
        from tabbiomed.analysis.results import _test_metric

        valid = dict(test_metrics={"mse": 0.0}, fixed_split_test_evaluated=True)
        missing_flag = dict(test_metrics={"mse": 0.0})
        cases = [(valid, True), (missing_flag, False), (None, False), ([], False)]
        for flag in (False, None, 0, 1, "true"):
            cases.append(({**valid, "fixed_split_test_evaluated": flag}, False))
        for field in ("failure", "error", "post_hpo_test_error"):
            cases.append(({**valid, field: {"reason": "gpu_oom"}}, False))
        for metrics in ({}, {"mse": float("nan")}, {"mse": float("inf")}, {"mse": True}, [], None):
            cases.append(({**valid, "test_metrics": metrics}, False))
        for block, expected in cases:
            with self.subTest(block=block):
                self.assertIs(test_available(block, "mse"), expected)
                self.assertIs(valid_test(block, "mse"), expected)
                self.assertEqual(_test_metric(block, "mse"), 0.0 if expected else None)


class FailureClassification(unittest.TestCase):
    def test_timeout_status_does_not_require_an_error_message(self):
        for error in (None, "Evaluation failed"):
            with self.subTest(error=error):
                self.assertEqual(
                    inferred_failure({"status": "timeout", "error": error}),
                    {"reason": "time_limit"},
                )

    def test_timeout_messages_are_recognized_in_both_error_fields(self):
        for field in ("error", "post_hpo_test_error"):
            for message in (
                "TimeoutError: evaluation stopped",
                "Time limit exceeded",
                "Evaluation process reached its deadline",
            ):
                with self.subTest(field=field, message=message):
                    self.assertEqual(
                        inferred_failure({field: message}), {"reason": "time_limit"}
                    )

    def test_explicit_and_other_failure_causes_are_preserved(self):
        failure = {"reason": "host_ram_oom", "memory_limit_gib": 128}
        self.assertEqual(
            inferred_failure({"status": "timeout", "failure": failure}), failure
        )
        for error, reason in (
            ("SIGSEGV, signal 11", "segmentation_fault"),
            ("CUDA out of memory", "gpu_oom"),
            ("out of memory", "out_of_memory"),
            ("Invalid input", "evaluation_failed"),
            (None, "not_available"),
        ):
            with self.subTest(error=error):
                self.assertEqual(inferred_failure({"error": error}), {"reason": reason})

    def test_hpo_exports_classify_sampled_and_default_timeouts(self):
        for default in (False, True):
            with self.subTest(default=default):
                record = dict(
                    trial_id=1,
                    status="timeout",
                    parameters={},
                    validation_metrics={},
                    started_at_hpo_seconds=0,
                    hpo_elapsed_seconds=12,
                    trial_wall_time_seconds=12,
                    validation_seconds=12,
                )
                exported = normalized_trial(record, None, default=default)
                self.assertEqual(exported["status"], "timeout")
                self.assertEqual(exported["failure"], {"reason": "time_limit"})
                self.assertEqual(exported["trial_id"], -1 if default else 1)


class TestLabelValidation(unittest.TestCase):
    def test_single_and_multiple_targets_preserve_test_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for shape in ((8,), (8, 1), (8, 3)):
                with self.subTest(shape=shape):
                    targets = np.arange(np.prod(shape)).reshape(shape)
                    rows = [6, 1, 4]
                    np.save(root / "targets.npy", targets)
                    np.save(root / "labels.npy", targets[rows])
                    validate_labels(root / "labels.npy", root / "targets.npy", rows)

    def test_flattened_or_incomplete_multioutput_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = np.arange(24).reshape(8, 3)
            rows = [6, 1, 4]
            np.save(root / "targets.npy", targets)
            for labels in (targets[rows].ravel(), targets[rows, :2], targets[rows][:2]):
                with self.subTest(shape=labels.shape):
                    np.save(root / "labels.npy", labels)
                    with self.assertRaisesRegex(ValueError, "stored test split size"):
                        validate_labels(root / "labels.npy", root / "targets.npy", rows)

    def test_multioutput_values_and_order_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = np.arange(24).reshape(8, 3)
            rows = [6, 1, 4]
            np.save(root / "targets.npy", targets)
            changed = targets[rows].copy()
            changed[1, 2] = -1
            for labels in (changed, targets[rows][::-1], targets[rows][:, ::-1]):
                np.save(root / "labels.npy", labels)
                with self.assertRaisesRegex(ValueError, "label values differ"):
                    validate_labels(root / "labels.npy", root / "targets.npy", rows)


if __name__ == "__main__":
    unittest.main()
