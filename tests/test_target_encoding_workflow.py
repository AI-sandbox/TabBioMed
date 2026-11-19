import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tabbiomed.results.consolidation import consolidate
from tabbiomed.results.sources import same_inputs
from tabbiomed.tabbiomed_datasets import TabularDataset
from tabbiomed.trainer.train import run_dataset


class TargetEncodingWorkflow(unittest.TestCase):
    def evaluate(self, root, labels, *, cv=False):
        data = root / "data"
        data.mkdir()
        y = np.tile(labels, 20)
        x = np.column_stack((np.arange(40, dtype=float), np.arange(40) % 3))
        metadata = dict(
            name="encoded_data",
            task="binary",
            num_samples=40,
            num_features=2,
            num_classes=2,
            cat_idx=[],
        )
        if cv:
            folds = []
            for fold in range(5):
                test = list(range(fold * 8, (fold + 1) * 8))
                pool = [i for i in range(40) if i not in test]
                folds.append(dict(train=pool[:24], val=pool[24:], test=test))
            splits = dict(cv_folds=folds)
            metadata["split_strategy"] = "cv"
        else:
            splits = dict(
                train=list(range(24)),
                val=list(range(24, 32)),
                test=list(range(32, 40)),
            )
        np.save(data / "X.npy", x)
        np.save(data / "y.npy", y)
        np.save(data / "split_indices.npy", splits)
        (data / "metadata.json").write_text(json.dumps(metadata))
        dataset = TabularDataset(
            name="encoded_data",
            X=x,
            y=y,
            task="binary",
            cat_idx=[],
            num_classes=2,
            split_indices=splits,
        )
        dataset.target_encode()
        run = run_dataset(
            dataset,
            metadata,
            model="LinearModel",
            data_directory=data,
            output=root / "raw",
        )
        return run, data, dataset

    def test_encoded_numeric_and_string_targets_round_trip(self):
        for labels in ([1, 2], ["negative", "positive"]):
            with self.subTest(
                labels=labels
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                run, data, dataset = self.evaluate(root, labels)
                raw = json.loads((run / "report.json").read_text())
                self.assertEqual(raw["input_identity"]["target_classes"], labels)
                receipt = consolidate(
                    run / "manifest.json", root / "results", execute=True
                )
                self.assertEqual(receipt["status"], "installed")
                exported = root / "results/LinearModel/encoded_data"
                np.testing.assert_array_equal(
                    np.load(exported / "test_labels_fixed.npy"), dataset.y_test
                )
                np.testing.assert_array_equal(
                    np.load(data / "y.npy"), np.tile(labels, 20)
                )

    def test_encoded_cv_targets_round_trip_all_outer_folds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, dataset = self.evaluate(root, [1, 2], cv=True)
            receipt = consolidate(run / "manifest.json", root / "results", execute=True)
            self.assertEqual(receipt["status"], "installed")
            exported = root / "results/LinearModel/encoded_data"
            for index, fold in enumerate(dataset.split_indices["cv_folds"]):
                np.testing.assert_array_equal(
                    np.load(exported / f"test_labels_fold_{index}.npy"),
                    dataset.y[fold["test"]],
                )

    def test_normalized_targets_do_not_record_an_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, _ = self.evaluate(root, [0, 1])
            report = json.loads((run / "report.json").read_text())
            self.assertNotIn("target_classes", report["input_identity"])
            receipt = consolidate(run / "manifest.json", root / "results", execute=True)
            self.assertEqual(receipt["status"], "installed")

    def test_tampered_classes_or_encoded_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, _, _ = self.evaluate(root, [1, 2])
            report_path = run / "report.json"
            report = json.loads(report_path.read_text())
            for classes in ([2, 1], [1, 3], [1], {"classes": [1, 2]}):
                with self.subTest(classes=classes):
                    report["input_identity"]["target_classes"] = classes
                    report_path.write_text(json.dumps(report))
                    with self.assertRaisesRegex(ValueError, "target classes"):
                        consolidate(
                            run / "manifest.json", root / "results", execute=True
                        )
                    self.assertFalse((root / "results").exists())
            report["input_identity"]["target_classes"] = [1, 2]
            report_path.write_text(json.dumps(report))
            labels = np.load(run / "test_labels_fixed.npy")
            np.save(run / "test_labels_fixed.npy", 1 - labels)
            with self.assertRaisesRegex(ValueError, "Test label values"):
                consolidate(run / "manifest.json", root / "results", execute=True)
            self.assertFalse((root / "results").exists())

    def test_different_target_encodings_cannot_be_combined(self):
        identity = dict(model="Example", dataset="example", seed=42, data_files={})
        with self.assertRaisesRegex(ValueError, "target_classes"):
            same_inputs(identity, {**identity, "target_classes": [1, 2]})


if __name__ == "__main__":
    unittest.main()
