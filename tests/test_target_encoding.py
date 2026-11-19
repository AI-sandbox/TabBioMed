import unittest

import numpy as np

from tabbiomed.tabbiomed_datasets import TabularDataset


class TargetEncoding(unittest.TestCase):
    def dataset(self, y, *, task="binary", splits=None):
        if splits is None:
            splits = dict(train=[3, 0], val=[1], test=[2])
        return TabularDataset(
            name="example",
            X=np.arange(len(y), dtype=float).reshape(-1, 1),
            y=y,
            cat_idx=[],
            task=task,
            num_classes=(
                len(np.unique(y)) if task in ("binary", "classification") else None
            ),
            split_indices=splits,
        )

    def check_splits(self, dataset):
        for part in ("train", "val", "test"):
            rows = np.asarray(dataset.split_indices[part], dtype=int)
            np.testing.assert_array_equal(
                getattr(dataset, "y_" + part), dataset.y[rows]
            )

    def test_classification_labels_and_splits_are_encoded_together(self):
        cases = (
            (np.array(["no", "yes", "no", "yes"]), "binary", [0, 1, 0, 1]),
            (np.array([1, 2, 1, 2]), "binary", [0, 1, 0, 1]),
            (np.array([-2.5, 7.0, -2.5, 7.0]), "binary", [0, 1, 0, 1]),
            (np.array([True, False, True, False]), "binary", [1, 0, 1, 0]),
            (np.array(["z", "a", "m", "a"]), "classification", [2, 0, 1, 0]),
            (np.array([10, 30, 20, 10]), "classification", [0, 2, 1, 0]),
        )
        for y, task, expected in cases:
            for splits in (
                dict(train=[3, 0], val=[1], test=[2]),
                dict(train=[3, 0], val=[2, 1], test=[]),
            ):
                with self.subTest(labels=y, task=task, splits=splits):
                    original = y.copy()
                    dataset = self.dataset(y, task=task, splits=splits)
                    dataset.target_encode()
                    np.testing.assert_array_equal(dataset.y, expected)
                    self.check_splits(dataset)
                    np.testing.assert_array_equal(y, original)

    def test_repeated_encoding_preserves_original_inverse_mapping(self):
        for y in (np.array(["no", "yes", "no", "yes"]), np.array([1, 2, 1, 2])):
            with self.subTest(labels=y):
                dataset = self.dataset(y)
                dataset.target_encode()
                encoder = dataset._target_encoder
                dataset.target_encode()
                self.assertIs(dataset._target_encoder, encoder)
                np.testing.assert_array_equal(encoder.inverse_transform(dataset.y), y)
                self.check_splits(dataset)

    def test_cv_folds_receive_encoded_parent_targets(self):
        y = np.array(["yes", "no", "yes", "no", "yes", "no"])
        splits = {
            "cv_folds": [
                dict(train=[3, 4], val=[5], test=[0, 1, 2]),
                dict(train=[0, 1], val=[2], test=[3, 4, 5]),
            ]
        }
        parent = self.dataset(y, splits=splits)
        parent.target_encode()
        np.testing.assert_array_equal(parent.y, [1, 0, 1, 0, 1, 0])
        for fold, _, _ in parent.get_cv_splits():
            self.check_splits(fold)
            np.testing.assert_array_equal(fold.y, parent.y)

    def test_normalized_readonly_targets_remain_unchanged(self):
        for dtype in (np.int64, np.float32, np.uint8):
            with self.subTest(dtype=dtype):
                y = np.array([0, 1, 0, 1], dtype=dtype)
                y.setflags(write=False)
                dataset = self.dataset(y)
                dataset.target_encode()
                self.assertIs(dataset.y, y)
                self.assertFalse(y.flags.writeable)
                self.check_splits(dataset)

    def test_regression_targets_are_not_encoded(self):
        for task, y in (
            ("regression", np.array([-0.25, 2.5, 0.75, 1.25])),
            ("multiregression", np.arange(8, dtype=float).reshape(4, 2) / 3),
        ):
            with self.subTest(task=task):
                dataset = self.dataset(y, task=task)
                dataset.target_encode()
                self.assertIs(dataset.y, y)
                self.check_splits(dataset)


if __name__ == "__main__":
    unittest.main()
