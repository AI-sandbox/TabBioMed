import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tabbiomed.trainer.hpo_trajectory import evaluate_hpo_budgets, budget_results_path


class BudgetResultReuse(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.candidate = dict(
            status="success",
            parameters={"width": 1},
            validation_metrics={"mse": 1.0},
            validation_seconds=1.0,
        )
        self.dataset = SimpleNamespace(
            y_test=np.array([0.0, 1.0]),
            selection_context=dict(
                candidate=self.candidate, budget=10.0, baseline=None
            ),
        )
        self.trials = [
            dict(
                trial_id=0,
                status="success",
                parameters={"width": 2},
                validation_metrics={"mse": 2.0},
                started_at_hpo_seconds=0.0,
                hpo_elapsed_seconds=1.0,
            )
        ]
        self.events = [
            dict(
                model="Example",
                dataset="Example",
                search_id="search",
                hpo_elapsed_seconds=1.0,
            )
        ]
        self.predictions = dict(y_pred=np.array([0.0, 1.0]), y_proba=None)

    def save_predictions(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.predictions, path)

    def evaluation(self, *, trial_id=-1, score=3.0, parameters=None, predictions=None):
        return dict(
            trial_id=trial_id,
            parameters=(
                self.candidate["parameters"] if parameters is None else parameters
            ),
            validation_metrics={"mse": 1.0},
            test_metrics={"mse": score},
            fixed_split_test_evaluated=True,
            predictions=predictions,
        )

    def run_budgets(self, baseline, best, *, fold_id=None):
        self.dataset.selection_context["baseline"] = baseline

        def refit(study, reports, dataset, model_class, model_args, predictions):
            reports[0].update(
                fixed_split_test_evaluated=True,
                test_metrics={"mse": 4.0},
                post_hpo_train_wall_time_seconds=0.1,
            )
            predictions.append(self.predictions)

        with (
            patch(
                "tabbiomed.trainer.hpo_trajectory.load_hpo_trials",
                return_value=self.trials,
            ),
            patch(
                "tabbiomed.trainer.hpo_trajectory._load_search_events",
                return_value=self.events,
            ),
            patch(
                "tabbiomed.trainer.hpo_trajectory.apply_fixed_split_test_eval_for_best_trial",
                side_effect=refit,
            ) as final_fit,
        ):
            evaluate_hpo_budgets(
                self.root,
                [25, 50, 100, 200],
                "mse",
                object,
                SimpleNamespace(seed=42),
                self.dataset,
                best_rep=best,
                fold_id=fold_id,
            )
        return (
            json.loads(budget_results_path(self.root, fold_id).read_text())["budgets"],
            final_fit.call_count,
        )

    def assert_results(self, rows, score, predictions):
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["selected_trial_id"], -1)
            self.assertEqual(row["parameters"], self.candidate["parameters"])
            self.assertEqual(
                row["validation_metrics"], self.candidate["validation_metrics"]
            )
            self.assertEqual(row["test_metrics"], {"mse": score})
            self.assertEqual(row["predictions"], predictions)
            payload = torch.load(self.root / row["predictions"], weights_only=False)
            np.testing.assert_array_equal(payload["y_pred"], self.predictions["y_pred"])

    def test_failed_default_reuses_successful_tuned_evaluation(self):
        for fold_id in (None, 2):
            with self.subTest(fold_id=fold_id):
                pointer = "selection/predictions_selected.pt"
                self.save_predictions(pointer)
                failed = dict(
                    error="CUDA out of memory",
                    fixed_split_test_evaluated=False,
                    test_metrics={},
                )
                best = self.evaluation(predictions=pointer)
                snapshot = copy.deepcopy(best)
                rows, refits = self.run_budgets(failed, best, fold_id=fold_id)
                self.assert_results(rows, 3.0, pointer)
                self.assertEqual(refits, 0)
                self.assertEqual(best, snapshot)

    def test_successful_default_keeps_priority_without_comparing_test_scores(self):
        self.save_predictions("predictions_default.pt")
        self.save_predictions("predictions_best.pt")
        baseline = self.evaluation(score=10.0)
        rows, refits = self.run_budgets(baseline, self.evaluation(score=1.0))
        self.assert_results(rows, 10.0, "predictions_default.pt")
        self.assertEqual(refits, 0)

    def test_missing_default_predictions_use_available_tuned_predictions(self):
        self.save_predictions("predictions_best.pt")
        rows, refits = self.run_budgets(self.evaluation(score=10.0), self.evaluation())
        self.assert_results(rows, 3.0, "predictions_best.pt")
        self.assertEqual(refits, 0)

    def test_invalid_cached_evaluations_refit_selected_configuration_once(self):
        bests = (
            dict(
                error="CUDA out of memory",
                fixed_split_test_evaluated=False,
                test_metrics={},
            ),
            {**self.evaluation(), "post_hpo_test_error": "Evaluation failed"},
            {**self.evaluation(), "test_metrics": {"mse": float("nan")}},
            self.evaluation(parameters={"width": 99}),
            self.evaluation(trial_id=7),
            self.evaluation(predictions="missing.pt"),
        )
        for best in bests:
            with self.subTest(best=best):
                budget_results_path(self.root).unlink(missing_ok=True)
                self.save_predictions("predictions_best.pt")
                rows, refits = self.run_budgets(dict(error="Default failed"), best)
                self.assert_results(rows, 4.0, "predictions_budget_trial_-1.pt")
                self.assertEqual(refits, 1)

    def test_resume_reuses_valid_budget_results_and_replaces_missing_predictions(self):
        rows, refits = self.run_budgets(None, None)
        self.assertEqual(refits, 1)
        rows, refits = self.run_budgets(None, None)
        self.assertEqual(refits, 0)
        self.assert_results(rows, 4.0, "predictions_budget_trial_-1.pt")
        (self.root / "predictions_budget_trial_-1.pt").unlink()
        self.save_predictions("predictions_best.pt")
        rows, refits = self.run_budgets(None, self.evaluation())
        self.assert_results(rows, 3.0, "predictions_best.pt")
        self.assertEqual(refits, 0)

    def test_resume_does_not_reuse_results_for_different_parameters(self):
        self.run_budgets(None, None)
        path = budget_results_path(self.root)
        report = json.loads(path.read_text())
        for row in report["budgets"]:
            row["parameters"] = {"width": 99}
        path.write_text(json.dumps(report))
        self.save_predictions("predictions_best.pt")
        rows, refits = self.run_budgets(None, self.evaluation())
        self.assert_results(rows, 3.0, "predictions_best.pt")
        self.assertEqual(refits, 0)

    def test_sampled_winner_reuses_its_successful_main_evaluation(self):
        self.trials[0]["validation_metrics"] = {"mse": 0.5}
        self.save_predictions("predictions_best.pt")
        best = self.evaluation(trial_id=0, parameters={"width": 2})
        rows, refits = self.run_budgets(dict(error="Default failed"), best)
        self.assertEqual(refits, 0)
        for row in rows:
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["selected_trial_id"], 0)
            self.assertEqual(row["parameters"], {"width": 2})
            self.assertEqual(row["test_metrics"], {"mse": 3.0})
            self.assertEqual(row["predictions"], "predictions_best.pt")


if __name__ == "__main__":
    unittest.main()
