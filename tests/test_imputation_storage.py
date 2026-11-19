import unittest

import numpy as np

from tabbiomed.tabbiomed_datasets import TabularDataset


class ImputationStorage(unittest.TestCase):
    def dataset(self, raw, cat_idx=(), splits=None):
        return TabularDataset(
            name="example",
            X=raw,
            y=np.arange(len(raw), dtype=float),
            cat_idx=list(cat_idx),
            task="regression",
            split_indices=(
                splits
                if splits is not None
                else dict(train=[0, 1, 2], val=[3], test=[4])
            ),
        )

    def test_fixed_width_imputation_preserves_fractions_and_categories(self):
        for dtype in ("U3", "S3"):
            for mixed in (False, True):
                for strategy, settings, fill in (
                    ("mean", {}, 10.5),
                    ("median", {}, 10.5),
                    ("knn", {"n_neighbors": 2}, 10.5),
                    ("constant", {"fill_value": -1234.567}, -1234.567),
                ):
                    with self.subTest(dtype=dtype, mixed=mixed, strategy=strategy):
                        values = [
                            [v, c]
                            for v, c in zip(
                                ("1", "20", "nan", "nan", "4"),
                                ("yes", "no", "yes", "no", "yes"),
                            )
                        ]
                        raw = np.array(values, dtype=dtype)
                        if not mixed:
                            raw = raw[:, :1].copy()
                        original = raw.copy()
                        raw.setflags(write=False)
                        ds = self.dataset(raw, cat_idx=[1] if mixed else [])
                        ds.impute_missing_values(strategy, **settings)
                        expected = np.array([1.0, 20.0, fill, fill, 4.0])
                        np.testing.assert_allclose(ds.X[:, 0].astype(float), expected)
                        for part, rows in ds.split_indices.items():
                            np.testing.assert_allclose(
                                getattr(ds, "X_" + part)[:, 0].astype(float),
                                expected[rows],
                            )
                            if mixed:
                                np.testing.assert_array_equal(
                                    getattr(ds, "X_" + part)[:, 1], original[rows, 1]
                                )
                        np.testing.assert_array_equal(raw, original)

    def test_heldout_missing_values_and_empty_test_partition(self):
        for dtype in ("U3", "S3"):
            with self.subTest(dtype=dtype):
                raw = np.array([["1"], ["20"], ["nan"]], dtype=dtype)
                ds = self.dataset(raw, splits=dict(train=[0, 1], val=[2], test=[]))
                ds.impute_missing_values("mean")
                np.testing.assert_allclose(ds.X[:, 0].astype(float), [1, 20, 10.5])
                self.assertEqual(ds.X_test.shape, (0, 1))

    def test_existing_numeric_storage_and_source_values_are_preserved(self):
        for dtype in (np.float32, np.float64, object):
            with self.subTest(dtype=dtype):
                raw = np.array([[1], [20], [np.nan], [np.nan], [4]], dtype=dtype)
                original = raw.copy()
                raw.setflags(write=False)
                ds = self.dataset(raw)
                ds.impute_missing_values("mean")
                self.assertEqual(ds.X.dtype, np.dtype(dtype))
                np.testing.assert_allclose(
                    ds.X[:, 0].astype(float), [1, 20, 10.5, 10.5, 4]
                )
                np.testing.assert_allclose(
                    raw.astype(float), original.astype(float), equal_nan=True
                )


if __name__ == "__main__":
    unittest.main()
