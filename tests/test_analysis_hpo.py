import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tabbiomed.results.hpo import select_prefix as export_select
from tabbiomed.results.hpo_read import read_history, select_prefix


class PublicHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pair = self.root / "A/task"
        (self.pair / "hpo").mkdir(parents=True)
        best = dict(
            parameters={"width": 2},
            validation_metrics={"mse": 2.0},
            test_metrics={"mse": 5.0},
            fixed_split_test_evaluated=True,
        )
        report = dict(
            model="A", dataset="task", hpo_enabled=True, primary_metric="mse", best=best
        )
        report_sha = self.write(self.pair / "report.json", report)
        timing = dict(
            report_sha256=report_sha,
            units=[
                dict(
                    fold_id=None,
                    hpo=dict(
                        official_attempt_cap=100,
                        attempted_configurations=1,
                        budget_seconds=20.0,
                        search_cutoff_seconds=18.0,
                        selection_wall_seconds=2.0,
                    ),
                )
            ],
        )
        timing_sha = self.write(self.pair / "timing.json", timing)
        self.trials = [
            dict(
                fold_id=None,
                trial_id=-1,
                parameters={"width": 1},
                status="success",
                duration_seconds=1.0,
                validation_metrics={"mse": 3.0},
            ),
            dict(
                fold_id=None,
                trial_id=0,
                parameters={"width": 2},
                status="success",
                started_seconds=0.0,
                finished_seconds=1.0,
                validation_metrics={"mse": 2.0},
            ),
        ]
        self.save_trials()
        self.selections = dict(
            schema_version=1,
            model="A",
            dataset="task",
            primary_metric="mse",
            report_sha256=report_sha,
            timing_sha256=timing_sha,
            units=[
                dict(
                    fold_id=None,
                    budgets=[
                        dict(
                            attempt_cap=cap,
                            actual_attempts=1,
                            cap_reached=False,
                            selected_trial_id=0,
                            status="success",
                            validation_metrics={"mse": 2.0},
                            test_metrics={"mse": 5.0},
                            predictions=None,
                        )
                        for cap in [25, 50, 100, 200]
                    ],
                )
            ],
        )
        self.save_selections()

    def write(self, path, value):
        data = json.dumps(value).encode()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def save_trials(self):
        (self.pair / "hpo/trials.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in self.trials)
        )

    def save_selections(self):
        self.write(self.pair / "hpo/selections.json", self.selections)

    def test_reader_needs_only_consolidated_files(self):
        official, rows, hashes, issues = read_history(
            self.root, "A", "task", "regression"
        )
        self.assertEqual(official[0]["scored"], 1)
        self.assertEqual([v["test"] for v in rows], [5.0] * 4)
        self.assertEqual(issues, [])
        self.assertTrue(all(str(self.pair) in p for p in hashes))

    def test_tampered_selection_and_stale_timing_are_rejected(self):
        self.selections["units"][0]["budgets"][0]["selected_trial_id"] = -1
        self.save_selections()
        with self.assertRaisesRegex(ValueError, "validation trajectory"):
            read_history(self.root, "A", "task", "regression")
        self.selections["timing_sha256"] = "wrong"
        self.save_selections()
        with self.assertRaisesRegex(ValueError, "identities"):
            read_history(self.root, "A", "task", "regression")

    def test_deadline_overrun_is_counted_and_cannot_win(self):
        self.trials.append(
            dict(
                fold_id=None,
                trial_id=1,
                parameters={"width": 3},
                status="success",
                started_seconds=2.0,
                finished_seconds=30.0,
                validation_metrics={"mse": 0.0},
            )
        )
        self.save_trials()
        for row in self.selections["units"][0]["budgets"]:
            row["actual_attempts"] = 2
        self.save_selections()
        official, rows, _, _ = read_history(self.root, "A", "task", "regression")
        self.assertEqual(official[0]["recorded_attempts"], 2)
        self.assertEqual(official[0]["scored"], 1)
        self.assertTrue(all(v["selected_trial_id"] == 0 for v in rows))

    def test_default_source_finishing_after_deadline_cannot_win(self):
        self.trials[0].update(
            duration_seconds=0.0, finished_seconds=30.0, validation_metrics={"mse": 1.0}
        )
        self.trials.append(
            dict(
                fold_id=None,
                trial_id=1,
                parameters={"width": 1},
                status="success",
                started_seconds=2.0,
                finished_seconds=30.0,
                validation_metrics={"mse": 1.0},
            )
        )
        self.save_trials()
        for row in self.selections["units"][0]["budgets"]:
            row["actual_attempts"] = 2
        self.save_selections()
        official, rows, _, _ = read_history(self.root, "A", "task", "regression")
        self.assertEqual(official[0]["scored"], 1)
        self.assertTrue(all(row["selected_trial_id"] == 0 for row in rows))

    def test_reader_accepts_correct_accuracy_f1_and_r2_selections(self):
        for metric in ("accuracy", "f1", "r2"):
            with self.subTest(metric=metric):
                report = json.loads((self.pair / "report.json").read_text())
                report["primary_metric"] = metric
                report["best"].update(
                    validation_metrics={metric: 0.9}, test_metrics={metric: 0.8}
                )
                report_sha = self.write(self.pair / "report.json", report)
                timing = json.loads((self.pair / "timing.json").read_text())
                timing["report_sha256"] = report_sha
                timing_sha = self.write(self.pair / "timing.json", timing)
                self.selections.update(
                    primary_metric=metric,
                    report_sha256=report_sha,
                    timing_sha256=timing_sha,
                )
                for row in self.selections["units"][0]["budgets"]:
                    row.update(
                        validation_metrics={metric: 0.9}, test_metrics={metric: 0.8}
                    )
                self.trials[0]["validation_metrics"] = {metric: 0.1}
                self.trials[1]["validation_metrics"] = {metric: 0.9}
                self.save_trials()
                self.save_selections()
                _, rows, _, issues = read_history(
                    self.root,
                    "A",
                    "task",
                    "regression" if metric == "r2" else "classification",
                )
                self.assertEqual(issues, [])
                self.assertEqual([row["selected_trial_id"] for row in rows], [0] * 4)
                self.assertEqual([row["validation"] for row in rows], [0.9] * 4)
                self.assertEqual([row["test"] for row in rows], [0.8] * 4)

    def test_duplicate_trial_identity_is_rejected(self):
        self.trials.append(self.trials[-1])
        self.save_trials()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_history(self.root, "A", "task", "regression")


class MetricDirections(unittest.TestCase):
    def trials(self, metric):
        return [
            dict(
                trial_id=i,
                status="success",
                parameters={},
                validation_metrics={metric: score},
                finished_seconds=1.0,
                hpo_elapsed_seconds=1.0,
            )
            for i, score in enumerate((0.1, 0.9))
        ]

    def test_reader_and_exporter_follow_supported_metric_directions(self):
        cases = (
            ("roc_auc", 1),
            ("auc", 1),
            ("accuracy", 1),
            ("f1", 1),
            ("r2", 1),
            ("ACCURACY", 1),
            ("F1", 1),
            ("R2", 1),
            ("mse", 0),
            ("mae", 0),
            ("rmse", 0),
            ("log_loss", 0),
        )
        for metric, expected in cases:
            for cap in (25, 50, 100, 200):
                with self.subTest(metric=metric, cap=cap):
                    trials = self.trials(metric)
                    self.assertEqual(
                        select_prefix(trials, {}, metric, cap, 10.0)["trial_id"],
                        expected,
                    )
                    self.assertEqual(
                        export_select(trials, {}, metric, cap, 10.0)["trial_id"],
                        expected,
                    )

    def test_default_comparison_uses_the_same_direction_and_keeps_ties(self):
        for metric, best, better, worse in (
            ("accuracy", 0.9, 0.95, 0.8),
            ("f1", 0.9, 0.95, 0.8),
            ("r2", 0.9, 0.95, 0.8),
            ("mse", 0.1, 0.05, 0.2),
            ("log_loss", 0.1, 0.05, 0.2),
        ):
            for score, expected in (
                (better, -1),
                (best, -1),
                (worse, 1 if metric in ("accuracy", "f1", "r2") else 0),
            ):
                with self.subTest(metric=metric, default=score):
                    candidate = dict(
                        trial_id=-1,
                        status="success",
                        parameters={},
                        validation_metrics={metric: score},
                    )
                    trials = self.trials(metric)
                    self.assertEqual(
                        select_prefix(trials, candidate, metric, 100, 10.0)["trial_id"],
                        expected,
                    )
                    self.assertEqual(
                        export_select(trials, candidate, metric, 100, 10.0)["trial_id"],
                        expected,
                    )


class SensitivityDirections(unittest.TestCase):
    def test_overtuning_uses_recorded_metric_instead_of_task(self):
        import pandas as pd
        from tabbiomed.analysis.studies.hpo_history import paired_statistics

        for task, metric, maximize in (
            ("regression", "r2", True),
            ("classification", "accuracy", True),
            ("classification", "f1", True),
            ("binary", "roc_auc", True),
            ("binary", "log_loss", False),
            ("regression", "mse", False),
        ):
            with self.subTest(task=task, metric=metric):
                rows = []
                for cap in (25, 50, 100, 200):
                    change = (0.1 if maximize else -0.1) if cap > 25 else 0.0
                    rows.append(
                        dict(
                            model="Example",
                            dataset="example",
                            task=task,
                            metric=metric,
                            cap=cap,
                            validation=0.7 + change,
                            test=0.7 - change,
                        )
                    )
                summary, pairs = paired_statistics(pd.DataFrame(rows))
                self.assertTrue(
                    summary.loc[summary.cap.gt(25), "validation_net_win"].eq(100).all()
                )
                self.assertTrue(
                    summary.loc[summary.cap.gt(25), "test_net_win"].eq(-100).all()
                )
                self.assertTrue(pairs.loc[pairs.cap.gt(25), "overtuning"].all())
                self.assertFalse(pairs.loc[pairs.cap.eq(25), "overtuning"].any())

    def test_budget_comparisons_reject_changed_objectives(self):
        import pandas as pd
        from tabbiomed.analysis.studies.hpo_history import paired_statistics

        frame = pd.DataFrame(
            [
                dict(
                    model="Example",
                    dataset="example",
                    task="regression",
                    cap=cap,
                    metric="r2" if cap == 25 else "mse",
                    validation=0.7,
                    test=0.6,
                )
                for cap in (25, 50, 100, 200)
            ]
        )
        with self.assertRaisesRegex(ValueError, "same metric"):
            paired_statistics(frame)


if __name__ == "__main__":
    unittest.main()
