from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tabbiomed.results.hpo import normalized_trial, select_prefix as export_select
from tabbiomed.results.hpo_read import select_prefix as public_select
from tabbiomed.results.selection import verify_selection
from tabbiomed.trainer.selection import select


class DefaultDeadlines(unittest.TestCase):
    def trial(self, finished, *, trial_id=0, started=0.0, score=1.0):
        return dict(
            trial_id=trial_id,
            status="success",
            parameters={"width": trial_id + 1},
            validation_metrics={"mse": score},
            started_at_hpo_seconds=started,
            hpo_elapsed_seconds=finished,
        )

    def candidate(self, trial, *, charge=0.0):
        return dict(
            status=trial["status"],
            parameters=trial["parameters"],
            validation_metrics=trial["validation_metrics"],
            validation_seconds=charge,
            source_trial_id=trial["trial_id"],
            source_completion_seconds=trial["hpo_elapsed_seconds"],
        )

    def test_late_default_cannot_win_at_any_configuration_cap(self):
        trial = self.trial(12.0)
        for cap in (25, 50, 100, 200):
            with self.subTest(cap=cap):
                result = select([trial], self.candidate(trial), "mse", cap, 10.0)
                self.assertFalse(result["default_selected"])
                self.assertIsNone(result["selected"])
                self.assertEqual(result["retained_attempts"], 1)
                self.assertEqual(result["trial_completion"]["n_successful_trials"], 0)
                self.assertEqual(
                    result["trial_completion"]["unsuccessful_trial_ids"], [0]
                )

    def test_default_is_eligible_up_to_the_remaining_search_deadline(self):
        for charge, finished, expected in (
            (0.0, 9.0, True),
            (0.0, 10.0, True),
            (0.0, 10.001, False),
            (2.0, 8.0, True),
            (2.0, 9.0, False),
        ):
            with self.subTest(charge=charge, finished=finished):
                trial = self.trial(finished)
                result = select(
                    [trial], self.candidate(trial, charge=charge), "mse", 100, 10.0
                )
                self.assertEqual(result["default_selected"], expected)
                self.assertEqual(result["validation_seconds"], charge)
                self.assertEqual(result["hpo_cutoff_seconds"], 10.0 - charge)
                if expected:
                    self.assertEqual(result["selected"]["trial_id"], -1)

    def test_invalid_or_missing_source_completion_cannot_win(self):
        for completion in (None, -1.0, float("nan"), float("inf"), True, "1"):
            with self.subTest(completion=completion):
                candidate = {
                    **self.candidate(self.trial(1.0)),
                    "source_completion_seconds": completion,
                }
                self.assertIsNone(select([], candidate, "mse", 100, 10.0)["selected"])
        candidate.pop("source_completion_seconds")
        self.assertIsNone(select([], candidate, "mse", 100, 10.0)["selected"])

    def test_independent_default_uses_its_validation_charge(self):
        for charge, expected in ((8.0, True), (10.0, True), (12.0, False)):
            with self.subTest(charge=charge):
                candidate = dict(
                    status="success",
                    parameters={},
                    validation_metrics={"mse": 1.0},
                    validation_seconds=charge,
                )
                result = select([], candidate, "mse", 100, 10.0)
                self.assertEqual(result["default_selected"], expected)

    def test_export_and_reader_agree_with_the_runner(self):
        for charge, finished in ((0.0, 9.0), (0.0, 12.0), (2.0, 8.0), (2.0, 9.0)):
            for cap in (25, 50, 100, 200):
                with self.subTest(charge=charge, finished=finished, cap=cap):
                    trial = self.trial(finished)
                    candidate = self.candidate(trial, charge=charge)
                    runner = select([trial], candidate, "mse", cap, 10.0)
                    exported = export_select(
                        [trial], candidate, "mse", cap, 10.0 - charge
                    )
                    public_candidate = normalized_trial(candidate, None, default=True)
                    public_trial = normalized_trial(trial, None)
                    self.assertEqual(public_candidate["finished_seconds"], finished)
                    public = public_select(
                        [public_trial], public_candidate, "mse", cap, 10.0 - charge
                    )
                    expected_id = (
                        runner["selected"]["trial_id"] if runner["selected"] else None
                    )
                    self.assertEqual(
                        exported["trial_id"] if exported else None, expected_id
                    )
                    self.assertEqual(
                        public["trial_id"] if public else None, expected_id
                    )
        independent = dict(
            status="success",
            parameters={},
            validation_metrics={"mse": 1.0},
            validation_seconds=8.0,
        )
        self.assertNotIn(
            "finished_seconds", normalized_trial(independent, None, default=True)
        )

    def test_consolidation_accepts_an_earlier_trial_when_default_finishes_late(self):
        earlier = self.trial(3.0, score=2.0)
        late = self.trial(12.0, trial_id=1, started=3.0)
        candidate = self.candidate(late)
        selection = select([earlier, late], candidate, "mse", 100, 10.0)
        self.assertEqual(selection["selected"]["trial_id"], 0)
        hpo = dict(
            reference_validation_charge_seconds=0.0,
            budget_seconds=10.0,
            search_cutoff_seconds=10.0,
            official_attempt_cap=100,
        )
        with (
            patch(
                "tabbiomed.results.selection.selection_timing", return_value=(hpo, "s")
            ),
            patch(
                "tabbiomed.results.selection.trial_records",
                return_value=([earlier, late], [], "s"),
            ),
        ):
            _, counts = verify_selection(
                {}, candidate, selection, [{}], None, "mse", {"time_limit": 10.0}
            )
        self.assertEqual(counts["n_successful_trials"], 1)


class SweepTimeoutHandling(unittest.TestCase):
    def test_shared_deadline_propagates_without_replacing_the_exception(self):
        from tabbiomed.trainer.evaluation import _run_sweep_with_hpo_step_timeout
        from tabbiomed.trainer.hpo_step_timeout import HPOStepDeadlineExceeded

        for budget in (None, 0, 10):
            with self.subTest(budget=budget):
                error = HPOStepDeadlineExceeded("Selection budget exhausted")
                sweep = SimpleNamespace(run=Mock(side_effect=error))
                with patch(
                    "tabbiomed.trainer.hpo_step_timeout.time.monotonic", return_value=100
                ):
                    with self.assertRaises(HPOStepDeadlineExceeded) as caught:
                        _run_sweep_with_hpo_step_timeout(sweep, budget)
                self.assertIs(caught.exception, error)
                sweep.run.assert_called_once_with()
                if budget:
                    self.assertEqual(sweep.hpo_step_deadline_monotonic, 110)
                else:
                    self.assertFalse(hasattr(sweep, "hpo_step_deadline_monotonic"))

    def test_trial_history_distinguishes_deadlines_failures_and_interruptions(self):
        from tabbiomed.trainer.hpo_step_timeout import HPOStepDeadlineExceeded
        from tabbiomed.trainer.sweep import OptunaSweep

        for error, status in (
            (HPOStepDeadlineExceeded("Selection budget exhausted"), "timeout"),
            (RuntimeError("Fit failed"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
        ):
            with self.subTest(status=status):
                sweep = OptunaSweep.__new__(OptunaSweep)
                sweep.history = Mock()
                sweep.trial_reports = []
                sweep.metric = "mse"
                sweep._objective = Mock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    sweep.objective(SimpleNamespace(number=0))
                self.assertIs(caught.exception, error)
                calls = sweep.history.write.call_args_list
                self.assertEqual(
                    [call.args[0] for call in calls],
                    ["trial_started", "trial_finished"],
                )
                self.assertEqual(calls[-1].kwargs["status"], status)
                self.assertEqual(calls[-1].kwargs["trial_id"], 0)


if __name__ == "__main__":
    unittest.main()
