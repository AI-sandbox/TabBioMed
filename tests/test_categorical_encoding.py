import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabbiomed.tabbiomed_datasets import TabularDataset


class CategoricalEncoding(unittest.TestCase):
    def prepare(self, raw, splits, strategy, cat_idx=None):
        dataset = TabularDataset(
            name="example",
            X=raw,
            y=np.arange(len(raw)) % 2,
            cat_idx=[0] if cat_idx is None else cat_idx,
            task="binary",
            num_classes=2,
            num_features=raw.shape[1],
            num_samples=len(raw),
            split_indices=splits,
        )
        dataset.prepare_dataset(
            inpute_mv=False,
            encode_cat=True,
            scale_features=False,
            cat_encoding_strategy=strategy,
        )
        return dataset

    def test_binary_missing_and_unseen_values_do_not_match_known_values(self):
        cases = (
            (
                [0, 1, np.nan, 0, None, 2, 1],
                dict(train=[0, 1, 2, 3], val=[4, 5], test=[6]),
                ([0, 1, -1, 0], [-1, -1], [1]),
            ),
            (
                ["no", "yes", "yes", "no", None, np.nan, pd.NA, "other", "no", "yes"],
                dict(train=[0, 1, 2, 3], val=[4, 5, 6, 7], test=[8, 9]),
                ([0, 1, 1, 0], [-1, -1, -1, -1], [0, 1]),
            ),
            (
                ["0", "1", "0", "1", "0.0", "1.0", "nan", "2", None],
                dict(train=[0, 1, 2, 3], val=[4, 5, 6], test=[7, 8]),
                ([0, 1, 0, 1], [0, 1, -1], [-1, -1]),
            ),
        )
        for strategy in ("onehot", "label"):
            for values, splits, expected in cases:
                with self.subTest(strategy=strategy, values=values):
                    raw = np.asarray(values, dtype=object).reshape(-1, 1)
                    original = pd.DataFrame(raw.copy())
                    dataset = self.prepare(raw, splits, strategy)
                    for part, target in zip(("train", "val", "test"), expected):
                        actual = getattr(dataset, "X_" + part).astype(float)
                        np.testing.assert_array_equal(actual[:, 0], target)
                        self.assertTrue(np.isfinite(actual).all())
                    pd.testing.assert_frame_equal(pd.DataFrame(raw), original)

    def test_reserved_code_survives_unsigned_and_string_storage(self):
        for dtype in (np.int8, np.uint8, np.uint64, "U1", "S1"):
            for strategy in ("onehot", "label"):
                with self.subTest(dtype=dtype, strategy=strategy):
                    raw = np.array([[0], [1], [2]], dtype=dtype)
                    original = raw.copy()
                    dataset = self.prepare(
                        raw, dict(train=[0, 1], val=[2], test=[]), strategy
                    )
                    np.testing.assert_array_equal(dataset.X_train[:, 0], [0, 1])
                    self.assertEqual(float(dataset.X_val[0, 0]), -1)
                    np.testing.assert_array_equal(raw, original)

    def test_nonbinary_unsigned_categories_keep_negative_unknown_codes(self):
        for dtype, offset in ((np.uint8, 0), (np.uint64, 2**63)):
            with self.subTest(dtype=dtype):
                raw = np.array(
                    [[offset + value] for value in (0, 1, 2, 9, 2, 9)], dtype=dtype
                )
                original = raw.copy()
                dataset = self.prepare(
                    raw, dict(train=[0, 1, 2], val=[3], test=[4, 5]), "label"
                )
                np.testing.assert_array_equal(dataset.X_train[:, 0], [0, 1, 2])
                np.testing.assert_array_equal(dataset.X_val[:, 0], [-1])
                np.testing.assert_array_equal(dataset.X_test[:, 0], [2, -1])
                self.assertEqual(dataset.cat_dims, [3])
                np.testing.assert_array_equal(raw, original)

    def test_label_codes_are_not_truncated_by_fixed_width_strings(self):
        for dtype in ("U1", "S1"):
            with self.subTest(dtype=dtype):
                raw = np.array(list("abcdefghijkzk"), dtype=dtype).reshape(-1, 1)
                original = raw.copy()
                dataset = self.prepare(
                    raw, dict(train=list(range(11)), val=[11], test=[12]), "label"
                )
                np.testing.assert_array_equal(dataset.X_train[:, 0], np.arange(11))
                np.testing.assert_array_equal(dataset.X_val[:, 0], [-1])
                np.testing.assert_array_equal(dataset.X_test[:, 0], [10])
                self.assertEqual(dataset.cat_dims, [11])
                np.testing.assert_array_equal(raw, original)

    def test_label_codes_do_not_overflow_narrow_signed_storage(self):
        raw = np.array(list(range(-128, 128)) + [-128, 127], dtype=np.int8).reshape(
            -1, 1
        )
        original = raw.copy()
        dataset = self.prepare(
            raw, dict(train=list(range(256)), val=[256], test=[257]), "label"
        )
        np.testing.assert_array_equal(dataset.X_train[:, 0], np.arange(256))
        np.testing.assert_array_equal(dataset.X_val[:, 0], [0])
        np.testing.assert_array_equal(dataset.X_test[:, 0], [255])
        np.testing.assert_array_equal(raw, original)

    def test_binary_codes_survive_mixed_categorical_encoding(self):
        raw = np.array(
            [
                [0, "a", 10],
                [1, "b", 20],
                [np.nan, "c", 30],
                [0, "a", 40],
                [1, "new", 50],
                [2, "a", 60],
            ],
            dtype=object,
        )
        splits = dict(train=[0, 1, 2, 3], val=[4], test=[5])
        for strategy in ("onehot", "label"):
            with self.subTest(strategy=strategy):
                dataset = self.prepare(raw.copy(), splits, strategy, cat_idx=[0, 1])
                binary_column = 1 if strategy == "onehot" else 0
                np.testing.assert_array_equal(
                    dataset.X_train[:, binary_column].astype(float), [0, 1, -1, 0]
                )
                self.assertEqual(float(dataset.X_val[0, binary_column]), 1)
                self.assertEqual(float(dataset.X_test[0, binary_column]), -1)
                if strategy == "onehot":
                    self.assertEqual(dataset.X_train.shape[1], 5)
                    np.testing.assert_array_equal(dataset.X_val[0, 2:], [0, 0, 0])
                else:
                    self.assertEqual(dataset.X_train.shape[1], 3)
                    self.assertEqual(float(dataset.X_val[0, 1]), -1)

    def test_label_missing_codes_are_independent_of_other_heldout_values(self):
        cases = (
            (["normal", "normal", np.nan], [np.nan, "degraded"], [1, -1]),
            (["a", None, np.nan], [None, np.nan, "z"], [1, 2, -1]),
            ([0, 2, 4, np.nan], [np.nan, 8, 2], [3, -1, 1]),
            (["a", "b", "c"], [np.nan, "z", "a"], [-1, -1, 0]),
        )
        for train, heldout, expected in cases:
            for part in ("val", "test"):
                with self.subTest(train=train, part=part):
                    raw = np.asarray(train + heldout, dtype=object).reshape(-1, 1)
                    splits = dict(train=list(range(len(train))), val=[], test=[])
                    splits[part] = list(range(len(train), len(raw)))
                    dataset = self.prepare(raw.copy(), splits, "label")
                    np.testing.assert_array_equal(
                        getattr(dataset, "X_" + part)[:, 0], expected
                    )
                    encoder = dataset._label_encoders[0]
                    np.testing.assert_array_equal(
                        dataset.X_train[:, 0], encoder.transform(raw[: len(train), 0])
                    )
                    self.assertEqual(dataset.cat_dims, [len(encoder.classes_)])
                    for value, code in zip(heldout, expected):
                        single_raw = np.asarray(train + [value], dtype=object).reshape(
                            -1, 1
                        )
                        single = self.prepare(
                            single_raw, {**splits, part: [len(train)]}, "label"
                        )
                        self.assertEqual(getattr(single, "X_" + part)[0, 0], code)

    def test_label_cardinalities_follow_categorical_column_order(self):
        raw = np.array(
            [
                ["a", 0.1, "no", "u"],
                ["b", 0.2, "yes", "v"],
                ["c", 0.3, "no", "w"],
                ["a", 0.4, "yes", "x"],
                ["unseen", 0.5, "no", "u"],
                ["b", 0.6, "yes", "v"],
            ],
            dtype=object,
        )
        splits = dict(train=[0, 1, 2, 3], val=[4], test=[5])
        cardinalities = {0: 3, 2: 2, 3: 4}
        encoded = None
        for order in ([0, 2, 3], [3, 2, 0], [2, 0, 3]):
            with self.subTest(order=order):
                dataset = self.prepare(raw.copy(), splits, "label", cat_idx=order)
                expected = [cardinalities[col] for col in order]
                self.assertEqual(dataset.cat_idx, order)
                self.assertEqual(dataset.cat_dims, expected)
                self.assertEqual(dataset.get_metadata()["cat_dims"], expected)
                for col, count in zip(order, dataset.cat_dims):
                    codes = dataset.X_train[:, col].astype(int)
                    self.assertEqual(len(np.unique(codes[codes >= 0])), count)
                self.assertEqual(float(dataset.X_val[0, 0]), -1)
                if encoded is None:
                    encoded = dataset.X.copy()
                else:
                    np.testing.assert_array_equal(dataset.X, encoded)

    def test_tabm_uses_reserved_embedding_slot_for_binary_unknowns(self):
        source = (
            Path(__import__("tabbiomed").__file__).parent
            / "models/model_definitions/tabm.py"
        )
        spec = importlib.util.spec_from_file_location("test_tabm_encoding", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            "sys.modules",
            {"tabm": SimpleNamespace(), "rtdl_num_embeddings": SimpleNamespace()},
        ):
            spec.loader.exec_module(module)
        model = module.TabM.__new__(module.TabM)
        model.cat_idx = [0]
        model._binary_cat_idx = set()
        model.n_cat_features = 1
        model.n_num_features = 0
        train = np.array([[0], [1], [-1]])
        model.cat_cardinalities = model._compute_cat_cardinalities(train)
        _, categories = model._prepare_data(train)
        self.assertEqual(model.cat_cardinalities, [3])
        np.testing.assert_array_equal(categories.numpy()[:, 0], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
