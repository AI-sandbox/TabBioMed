import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tabbiomed.models import BENCHMARK_MODELS
from tabbiomed.slurm import prepare, submit, run_task


class SlurmWorkflow(unittest.TestCase):
    def test_complete_pair_resources_and_explicit_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = prepare(
                root,
                models=["RandomForest", "TabM"],
                datasets=["pima_diabetes"],
                partition="cpu",
                gpu_partition="gpu",
                gpu_type="a5500",
            )
            self.assertEqual(len(scripts), 2)
            document = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(document["tasks"]), 2)
            for script in scripts:
                text = script.read_text()
                self.assertIn("--time=24:00:00", text)
                self.assertIn("--cpus-per-task=10", text)
                self.assertIn("--mem=128G", text)
                self.assertNotIn("default_only", text)
            with patch("tabbiomed.slurm.subprocess.check_output") as sbatch:
                submit(root)
                sbatch.assert_not_called()
            with patch("tabbiomed.slurm.subprocess.run") as run:
                run_task(root / "manifest.json", 1)
                args = run.call_args.args[0]
                self.assertIn("tabbiomed.trainer", args)
                self.assertIn("--gpu", args)

    def test_invalid_indices_never_start_a_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare(root, models=["RandomForest"], datasets=["pima_diabetes", "lymphography"])
            with patch("tabbiomed.slurm.subprocess.run") as run:
                for index in (-2, -1, 2, 3, 0.5, "1", None, True, False):
                    with self.subTest(index=index):
                        with self.assertRaisesRegex(ValueError, "Task index"):
                            run_task(root / "manifest.json", index)
                        run.assert_not_called()
                for index, dataset in ((0, "pima_diabetes"), (1, "lymphography")):
                    run_task(root / "manifest.json", index)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--dataset") + 1], dataset)
                self.assertEqual(run.call_count, 2)

    def test_partial_submission_receipt_prevents_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare(root, models=["RandomForest", "TabM"], datasets=["pima_diabetes"])
            with patch(
                "tabbiomed.slurm.subprocess.check_output",
                side_effect=["123\n", subprocess.CalledProcessError(1, "sbatch")],
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    submit(root, execute=True)
            with patch(
                "tabbiomed.slurm.subprocess.check_output", return_value="124\n"
            ) as sbatch:
                receipt = submit(root, execute=True)
                sbatch.assert_called_once()
            self.assertEqual([r["job_id"] for r in receipt], ["123", "124"])

    def test_separate_environments_require_explicit_selection(self):
        separate = {"TabPFN-Wide", "TabPFN-3.5"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("tabbiomed.slurm.all_models", list(BENCHMARK_MODELS)):
                prepare(root / "default", datasets=["pima_diabetes"])
            tasks = json.loads((root / "default" / "manifest.json").read_text())["tasks"]
            names = {task["model"] for task in tasks}
            self.assertIn("TabM", names)
            self.assertTrue(separate.isdisjoint(names))

            prepare(root / "explicit", datasets=["pima_diabetes"], models=sorted(separate))
            tasks = json.loads((root / "explicit" / "manifest.json").read_text())["tasks"]
            self.assertEqual({task["model"] for task in tasks}, separate)


if __name__ == "__main__":
    unittest.main()
