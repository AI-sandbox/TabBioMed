import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabbiomed.analysis.common import model_hpo_policy
from tabbiomed.analysis.results import read_result
from tabbiomed.analysis.studies import a100, failures, hpo_sensitivity, pretraining


class AnalysisCapabilities(unittest.TestCase):
    def rows(self):
        return pd.DataFrame(
            [
                dict(
                    model="NewDefault",
                    dataset="example",
                    task="binary",
                    hpo_enabled=False,
                    default_metric=0.8,
                    tuned_metric=np.nan,
                ),
                dict(
                    model="RandomForest",
                    dataset="example",
                    task="binary",
                    hpo_enabled=True,
                    default_metric=0.7,
                    tuned_metric=0.9,
                ),
            ]
        )

    def test_hpo_policy_rejects_inconsistent_or_implicit_values(self):
        for policies in ([True, False], [1, 1], [None, None]):
            with self.subTest(policies=policies):
                frame = pd.DataFrame(
                    {"model": ["Example", "Example"], "hpo_enabled": policies}
                )
                with self.assertRaises(ValueError):
                    model_hpo_policy(frame)

    def test_budget_analysis_has_no_expected_tuned_score_for_default_only_model(self):
        milestones = pd.DataFrame(
            [dict(cap=100, model="RandomForest", dataset="example", test=0.9)]
        )
        frame, missing = hpo_sensitivity.elo_inputs(
            self.rows(), milestones, ["NewDefault", "RandomForest"], 100
        )
        self.assertEqual(
            set(frame.competitor),
            {"NewDefault::default", "RandomForest::default", "RandomForest::tuned"},
        )
        self.assertEqual(missing["missing_input_cells"], 0)

    def test_retry_and_pretraining_rankings_retain_default_only_model(self):
        models, entries, frame, _ = a100.elo_table(self.rows())
        self.assertEqual(models.loc["NewDefault", "variant"], "default")
        self.assertNotIn("NewDefault::tuned", entries.index)
        original, _ = pretraining.elo_tables(frame)
        self.assertEqual(original.loc["NewDefault", "variant"], "default")

    def test_failure_collection_counts_only_the_evaluated_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = root / "NewDefault/example"
            pair.mkdir(parents=True)
            report = dict(
                artifact_type="evaluation_summary",
                schema_version=1,
                model="NewDefault",
                dataset="example",
                primary_metric="roc_auc",
                hpo_enabled=False,
                default=dict(
                    fixed_split_test_evaluated=True, test_metrics={"roc_auc": 0.8}
                ),
                best=None,
            )
            path = pair / "report.json"
            path.write_text(json.dumps(report))
            row = read_result(
                "NewDefault", "example", dict(task="binary", n_folds=0), root=root
            )
            self.assertFalse(row["hpo_enabled"])
            (root / "metadata.json").write_text("{}")
            with (
                patch.object(
                    failures, "build_results", return_value=pd.DataFrame([row])
                ),
                patch.object(failures, "dataset_directory", return_value=root),
                patch.object(
                    failures,
                    "analysis_dataset_metadata",
                    return_value={"label": "Single-cell"},
                ),
            ):
                _, variants, _ = failures.collect(
                    {"example": {"n_folds": 0}}, root, None
                )
            self.assertEqual(variants.variant.tolist(), ["default"])
            self.assertTrue(variants.completed.all())


if __name__ == "__main__":
    unittest.main()
