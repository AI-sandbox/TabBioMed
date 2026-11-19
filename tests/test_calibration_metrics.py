import unittest

import numpy as np

from tabbiomed.analysis.calibration import (
    compute_metrics,
    compute_metrics_multiclass,
    ece,
)


class CalibrationMetrics(unittest.TestCase):
    def test_single_class_binary_metrics_keep_defined_scores(self):
        from sklearn.metrics import log_loss

        p = np.array([0.1, 0.2, 0.7, 0.9])
        for label in (0, 1):
            with self.subTest(label=label):
                y = np.full(len(p), label)
                result = compute_metrics(y, p)
                for field in (
                    "roc_auc_pooled",
                    "brier_skill",
                    "cal_slope",
                    "cal_intercept",
                ):
                    self.assertTrue(np.isnan(result[field]), field)
                self.assertAlmostEqual(result["brier"], np.mean((p - y) ** 2))
                self.assertAlmostEqual(
                    result["log_loss_pooled"], log_loss(y, p, labels=[0, 1])
                )
                self.assertAlmostEqual(
                    result["ece_uniform"], np.mean(np.abs(p - label))
                )
                self.assertEqual(result["n_pos"], label * len(p))
                if label:
                    self.assertEqual(result["pr_auc_pooled"], 1.0)
                else:
                    self.assertTrue(np.isnan(result["pr_auc_pooled"]))

    def test_single_class_reliability_uses_positive_class_one(self):
        from unittest.mock import patch
        from tabbiomed.analysis.figures.calibration import reliability

        p = np.array([0.1, 0.2, 0.7, 0.9])
        for label in (0, 1):
            with self.subTest(label=label):
                y = np.full(len(p), label)
                loaded = (
                    y,
                    np.column_stack([1 - p, p]),
                    "best",
                    "fixed",
                    np.zeros(len(y)),
                )
                with patch(
                    "tabbiomed.analysis.figures.calibration.cs.load_pair",
                    return_value=loaded,
                ):
                    _, fraction, probabilities, multiclass, metrics = reliability(
                        "example", "Example", 15, "uniform", "best"
                    )
                np.testing.assert_array_equal(probabilities, p)
                np.testing.assert_array_equal(fraction, np.full(len(fraction), label))
                self.assertFalse(multiclass)
                self.assertEqual(metrics["n_pos"], label * len(p))
                self.assertTrue(np.isnan(metrics["roc_auc_pooled"]))

    def test_constant_probabilities_form_one_quantile_bin(self):
        y = np.array([0, 1, 0, 1])
        for value in (0.0, 0.2, 0.9, 1.0):
            for bins in (1, 15, 100):
                with self.subTest(probability=value, bins=bins):
                    p = np.full(len(y), value)
                    expected = abs(0.5 - value)
                    self.assertAlmostEqual(
                        ece(y, p, n_bins=bins, strategy="quantile"), expected
                    )
                    self.assertAlmostEqual(
                        ece(y, p, n_bins=bins, strategy="uniform"), expected
                    )
        self.assertAlmostEqual(
            ece(np.array([1]), np.array([0.7]), strategy="quantile"), 0.3
        )

    def test_binary_and_multiclass_metric_exports_keep_constant_confidence_error(self):
        binary = compute_metrics(np.array([0, 1]), np.array([0.9, 0.9]))
        self.assertAlmostEqual(binary["ece_quantile"], 0.4)
        self.assertAlmostEqual(binary["ece_uniform"], 0.4)
        y = np.array([0, 1, 2, 0])
        multiclass = compute_metrics_multiclass(
            y, np.tile([0.1, 0.2, 0.7], (len(y), 1))
        )
        self.assertAlmostEqual(multiclass["ece_quantile"], 0.45)
        self.assertAlmostEqual(multiclass["ece_uniform"], 0.45)

    def test_nonconstant_and_repeated_probabilities_keep_their_weighted_bins(self):
        for y, p, bins, expected in (
            ([0, 1, 1, 0], [0.1, 0.3, 0.8, 0.9], 2, 0.325),
            ([0, 1, 1, 1], [0.0, 0.0, 1.0, 1.0], 15, 0.25),
        ):
            for strategy in ("uniform", "quantile"):
                with self.subTest(strategy=strategy, probabilities=p):
                    self.assertAlmostEqual(
                        ece(np.array(y), np.array(p), n_bins=bins, strategy=strategy),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
