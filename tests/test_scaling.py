import unittest

import numpy as np

from tabbiomed.tabbiomed_datasets import TabularDataset
from tabbiomed.trainer.utils.preprocessing import scale_splits_fit_on_train


class ScalingStorage(unittest.TestCase):
    def test_unknown_scalers_fail_even_when_no_columns_need_scaling(self):
        for strategy in ("qunatile", "StandardScaler", "none", None, 1):
            for raw, categorical in (
                (np.array([[2.0], [3.0], [4.0]]), []),
                (np.array([[0.0], [1.0], [0.0]]), []),
                (np.array([[2.0], [3.0], [4.0]]), [0]),
            ):
                with self.subTest(strategy=strategy, categorical=categorical):
                    with self.assertRaisesRegex(ValueError, "Unsupported scaler"):
                        scale_splits_fit_on_train(
                            raw,
                            raw[:1],
                            raw[:1],
                            cat_idx=categorical,
                            scaler_type=strategy,
                        )

    def test_numeric_strings_retain_fractional_and_negative_scaled_values(self):
        for dtype in ("U1", "S1"):
            with self.subTest(dtype=dtype):
                raw = np.array([[1], [2], [3], [4], [0]], dtype=dtype)
                original = raw.copy()
                raw.setflags(write=False)
                train, val, test, scaler, plan = scale_splits_fit_on_train(
                    raw[:3], raw[3:4], raw[4:], cat_idx=[]
                )
                expected = (np.array([1, 2, 3, 4, 0]) - 2) / np.sqrt(2 / 3)
                np.testing.assert_allclose(
                    np.vstack((train, val, test))[:, 0].astype(float), expected
                )
                np.testing.assert_allclose(scaler.mean_, [2])
                self.assertEqual(plan.scale_indices, [0])
                np.testing.assert_array_equal(raw, original)

    def test_mixed_strings_preserve_categorical_and_binary_columns(self):
        for dtype in ("U1", "S1"):
            with self.subTest(dtype=dtype):
                raw = np.array(
                    [
                        ["a", "1", "0"],
                        ["b", "2", "1"],
                        ["a", "3", "0"],
                        ["z", "4", "1"],
                    ],
                    dtype=dtype,
                )
                original = raw.copy()
                train, val, test, _, plan = scale_splits_fit_on_train(
                    raw[:3], raw[3:], raw[:0], cat_idx=[0]
                )
                combined = np.vstack((train, val))
                np.testing.assert_array_equal(combined[:, [0, 2]], raw[:, [0, 2]])
                np.testing.assert_allclose(
                    combined[:, 1].astype(float), (np.arange(1, 5) - 2) / np.sqrt(2 / 3)
                )
                self.assertEqual(plan.scale_indices, [1])
                self.assertEqual(plan.binary_indices, [2])
                self.assertEqual(test.shape, (0, 3))
                np.testing.assert_array_equal(raw, original)

    def test_quantile_scaling_preserves_string_input_precision(self):
        for dtype in ("U6", "S6"):
            with self.subTest(dtype=dtype):
                raw = np.array(
                    [*map(str, range(2, 1002)), "500.5", "2000"], dtype=dtype
                ).reshape(-1, 1)
                train, val, test, _, _ = scale_splits_fit_on_train(
                    raw[:1000],
                    raw[1000:1001],
                    raw[1001:],
                    cat_idx=[],
                    scaler_type="quantile",
                    random_state=42,
                )
                np.testing.assert_allclose(
                    train[:, 0].astype(float), np.linspace(0, 1, 1000), atol=1e-12
                )
                np.testing.assert_allclose(val[:, 0].astype(float), [(500.5 - 2) / 999])
                np.testing.assert_array_equal(test[:, 0], [1.0])

    def test_existing_numeric_types_preserve_values_and_excluded_columns(self):
        for dtype in (np.float32, np.float64, np.int32, np.uint8, object):
            with self.subTest(dtype=dtype):
                raw = np.array([[1, 0], [2, 1], [3, 0], [4, 1], [0, 0]], dtype=dtype)
                original = raw.copy()
                train, val, test, _, plan = scale_splits_fit_on_train(
                    raw[:3], raw[3:4], raw[4:], cat_idx=[]
                )
                combined = np.vstack((train, val, test))
                np.testing.assert_allclose(
                    combined[:, 0].astype(float),
                    (np.array([1, 2, 3, 4, 0]) - 2) / np.sqrt(2 / 3),
                    rtol=1e-6,
                )
                np.testing.assert_array_equal(combined[:, 1], raw[:, 1])
                self.assertEqual(plan.binary_indices, [1])
                if dtype in (np.float32, np.float64, object):
                    self.assertEqual(train.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(raw, original)

    def test_dataset_scaling_preserves_scaled_storage(self):
        for dtype in ("U1", "S1"):
            with self.subTest(dtype=dtype):
                raw = np.array([[1], [2], [3], [4], [0]], dtype=dtype)
                original = raw.copy()
                dataset = TabularDataset(
                    name="example",
                    X=raw,
                    y=np.arange(5, dtype=float),
                    cat_idx=[],
                    task="regression",
                    split_indices=dict(train=[0, 1, 2], val=[3], test=[4]),
                )
                dataset.scale_features()
                expected = (np.array([1, 2, 3, 4, 0]) - 2) / np.sqrt(2 / 3)
                np.testing.assert_allclose(dataset.X[:, 0].astype(float), expected)
                np.testing.assert_allclose(
                    dataset.X_train[:, 0].astype(float), expected[:3]
                )
                np.testing.assert_allclose(
                    dataset.X_val[:, 0].astype(float), expected[3:4]
                )
                np.testing.assert_allclose(
                    dataset.X_test[:, 0].astype(float), expected[4:]
                )
                np.testing.assert_array_equal(raw, original)


if __name__ == "__main__":
    unittest.main()
