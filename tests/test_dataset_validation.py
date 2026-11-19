import unittest

import numpy as np

from tabbiomed.tabbiomed_datasets import TabularDataset


class DatasetValidation(unittest.TestCase):
    def dataset(self, splits, cat_idx=()):
        return TabularDataset(
            name="example",
            X=np.arange(24).reshape(8, 3),
            y=np.arange(8) % 2,
            cat_idx=list(cat_idx),
            task="binary",
            split_indices=splits,
        )

    def valid_split(self):
        return dict(train=[3, 1, 2, 0], val=[5, 4], test=[7, 6])

    def check_rejected(self, split, message):
        for cv in (False, True):
            with self.subTest(cv=cv):
                splits = {"cv_folds": [self.valid_split(), split]} if cv else split
                with self.assertRaisesRegex(ValueError, message):
                    self.dataset(splits)

    def test_invalid_row_indices_are_rejected_in_every_partition(self):
        for part in ("train", "val", "test"):
            for rows, message in (
                ([-1], "outside"),
                ([8], "outside"),
                ([0, 0], "duplicate"),
                ([0.5], "integers"),
                ([0.0], "integers"),
                ([False, True], "integers"),
                ([0, True], "integers"),
                (["0"], "integers"),
                ([[0]], "one-dimensional"),
                (0, "one-dimensional"),
                (np.array([2**63], dtype=np.uint64), "outside"),
            ):
                with self.subTest(part=part, rows=rows):
                    split = {**self.valid_split(), part: rows}
                    self.check_rejected(split, message)

    def test_partition_overlaps_are_rejected(self):
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            with self.subTest(left=left, right=right):
                split = self.valid_split()
                split[right] = [split[left][0]]
                self.check_rejected(split, "disjoint")

    def test_invalid_categorical_indices_are_rejected(self):
        for cat_idx, message in (
            ([-1], "outside"),
            ([3], "outside"),
            ([0, 0], "duplicate"),
            ([0.5], "integers"),
            ([0.0], "integers"),
            ([False], "integers"),
            ([0, True], "integers"),
            (["0"], "integers"),
            ([[0]], "one-dimensional"),
        ):
            for cv in (False, True):
                with self.subTest(cat_idx=cat_idx, cv=cv):
                    split = self.valid_split()
                    splits = {"cv_folds": [split]} if cv else split
                    with self.assertRaisesRegex(ValueError, "cat_idx.*" + message):
                        self.dataset(splits, cat_idx)

    def test_integer_indices_preserve_order_and_input_arrays(self):
        for dtype in (np.int32, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                splits = {
                    part: np.asarray(rows, dtype=dtype)
                    for part, rows in self.valid_split().items()
                }
                dataset = self.dataset(splits, [np.int32(2), np.int64(0)])
                self.assertIs(dataset.split_indices, splits)
                self.assertEqual(dataset.cat_idx, [2, 0])
                for part, rows in splits.items():
                    np.testing.assert_array_equal(
                        getattr(dataset, "X_" + part), dataset.X[rows]
                    )
                    np.testing.assert_array_equal(
                        getattr(dataset, "y_" + part), dataset.y[rows]
                    )
                    np.testing.assert_array_equal(rows, self.valid_split()[part])
                    self.assertEqual(rows.dtype, dtype)

    def test_inner_fold_subsets_and_empty_test_partitions_remain_valid(self):
        split = dict(train=[3, 1], val=[5], test=[])
        dataset = self.dataset(split)
        self.assertEqual(dataset.X_train.shape, (2, 3))
        self.assertEqual(dataset.X_test.shape, (0, 3))
        parent = self.dataset({"cv_folds": [split, self.valid_split()]})
        folds = list(parent.get_cv_splits())
        self.assertEqual(len(folds), 2)
        np.testing.assert_array_equal(folds[0][0].X_train, dataset.X_train)

    def test_empty_cv_training_or_validation_is_rejected_before_yielding(self):
        for part in ("train", "val"):
            for fold_id in range(3):
                with self.subTest(part=part, fold_id=fold_id):
                    folds = [self.valid_split() for _ in range(3)]
                    folds[fold_id][part] = []
                    parent = self.dataset({"cv_folds": folds})
                    iterator = parent.get_cv_splits()
                    with self.assertRaisesRegex(
                        ValueError, f"CV fold {fold_id} has empty {part} indices"
                    ):
                        next(iterator)

    def test_cv_iteration_preserves_all_fold_memberships(self):
        folds = [
            {
                part: [(row + shift) % 8 for row in rows]
                for part, rows in self.valid_split().items()
            }
            for shift in range(3)
        ]
        parent = self.dataset({"cv_folds": folds})
        yielded = list(parent.get_cv_splits())
        self.assertEqual(len(yielded), len(folds))
        for fold_id, (dataset, train, val) in enumerate(yielded):
            self.assertEqual(dataset.split_indices, folds[fold_id])
            np.testing.assert_array_equal(train, folds[fold_id]["train"])
            np.testing.assert_array_equal(val, folds[fold_id]["val"])
            np.testing.assert_array_equal(
                dataset.X_test, parent.X[folds[fold_id]["test"]]
            )

    def test_malformed_split_structures_are_rejected(self):
        for splits in (
            None,
            [],
            {"train": [0], "val": [1]},
            {"cv_folds": None},
            {"cv_folds": []},
            {"cv_folds": [None]},
            {"cv_folds": [dict(train=[0], val=[1])]},
        ):
            with self.subTest(splits=splits):
                with self.assertRaises(ValueError):
                    self.dataset(splits)


if __name__ == "__main__":
    unittest.main()
