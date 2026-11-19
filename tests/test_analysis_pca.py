import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tabbiomed.analysis.studies import pca


class PCAComparison(unittest.TestCase):
    def write_reports(self, root, models):
        for index, model in enumerate(models):
            main = {
                "model": model,
                "dataset": "example_dataset",
                "primary_metric": "roc_auc",
                "seed": 42,
                "search_space": {"alpha": {"type": "float", "low": 0.1, "high": 1.0}},
                "default": {
                    "test_metrics": {"roc_auc": 0.6},
                    "fixed_split_test_evaluated": True,
                },
                "best": {
                    "test_metrics": {"roc_auc": 0.8},
                    "fixed_split_test_evaluated": True,
                },
            }
            reduced = {
                **main,
                "search_space": {
                    **main["search_space"],
                    "apply_pca": {"choices": [True]},
                    "pca_max_features": {"choices": [500]},
                    "pca_fit": {"choices": ["lowrank"]},
                },
                "best": {
                    "test_metrics": {"roc_auc": 0.9 if index % 2 == 0 else 0.7},
                    "fixed_split_test_evaluated": True,
                },
            }
            directory = root / model / "example_dataset"
            (directory / "pca_500").mkdir(parents=True)
            (directory / "report.json").write_text(json.dumps(main))
            (directory / "pca_500/report.json").write_text(json.dumps(reduced))
            (directory / "pca_500/pca_components.json").write_text("{}")

    def run_analysis(self, root, output, arguments):
        inventory = {
            "example_dataset": {
                "num_features": 3000,
                "num_samples": 1000,
                "task": "binary",
            }
        }
        with (
            patch.object(
                sys,
                "argv",
                [
                    "pca",
                    "--results-root",
                    str(root),
                    "--output",
                    str(output),
                    *arguments,
                ],
            ),
            patch.object(pca, "ensure_results"),
            patch.object(pca, "dataset_inventory", return_value=inventory),
        ):
            pca.main()
        return json.loads((output / "analysis.json").read_text())

    def test_custom_model_lists_control_reports_tables_and_figures(self):
        for models in (
            ("Example_A",),
            ("Example_C", "Example_A", "ExampleB", "ExampleD", "ExampleE"),
        ):
            with (
                self.subTest(models=models),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary) / "results"
                output = Path(temporary) / "plots"
                self.write_reports(root, models)
                manifest = self.run_analysis(root, output, ["--models", *models])
                self.assertEqual(manifest["models"], list(models))
                self.assertEqual(manifest["model_dataset_pairs"], len(models))
                self.assertIsNone(manifest["pfn_direction_agreement"])
                with (output / "comparisons.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row["model"] for row in rows], list(models))
                for index, row in enumerate(rows):
                    self.assertAlmostEqual(
                        float(row["delta"]), 0.1 if index % 2 == 0 else -0.1
                    )
                table = (output / "tfm_pcs_table.tex").read_text()
                self.assertIn("@{}lcrr" + "r" * len(models) + "@{}", table)
                self.assertIn("\\multicolumn{" + str(4 + len(models)) + "}", table)
                headers = ["\\makecell{" + pca.tex_name(model) for model in models]
                self.assertEqual(
                    [table.index(header) for header in headers],
                    sorted(table.index(header) for header in headers),
                )
                self.assertNotIn("PCA_TABLE_", table)
                summary = (output / "tfms_pca_summary.tex").read_text()
                for model in models:
                    self.assertIn(pca.tex_name(model), summary)
                for suffix in ("pdf", "png", "svg"):
                    self.assertTrue(
                        (output / f"pca_paired_outcomes.{suffix}").is_file()
                    )

    def test_failed_default_does_not_discard_tuned_test_result(self):
        model = "Example"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            output = Path(temporary) / "plots"
            self.write_reports(root, [model])
            path = root / model / "example_dataset" / "report.json"
            report = json.loads(path.read_text())
            failure = {"reason": "segmentation_fault"}
            report.update(
                status="incomplete",
                failure=failure,
                default={"fixed_split_test_evaluated": False, "failure": failure},
            )
            path.write_text(json.dumps(report))
            self.run_analysis(root, output, ["--models", model])
            with (output / "paired_comparisons.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["main_variant"], "tuned")
            self.assertAlmostEqual(float(rows[0]["delta"]), 0.1)

    def test_default_models_are_preserved(self):
        models = ("TabICLv2", "TabPFNv2.5", "RealTabPFNv2.5")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            output = Path(temporary) / "plots"
            self.write_reports(root, models)
            manifest = self.run_analysis(root, output, [])
        self.assertEqual(manifest["models"], list(models))
        self.assertEqual(
            manifest["pfn_direction_agreement"],
            {"shared_paired_datasets": 1, "same_direction": 0},
        )

    def test_duplicate_models_are_rejected_before_loading_results(self):
        with (
            patch.object(
                sys,
                "argv",
                ["pca", "--output", "/tmp/pca", "--models", "Example", "Example"],
            ),
            patch.object(pca, "ensure_results") as load,
            patch("sys.stderr"),
            self.assertRaises(SystemExit) as error,
        ):
            pca.main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
