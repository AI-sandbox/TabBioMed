import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabbiomed.analysis import downloads
from tabbiomed.analysis.__main__ import COMMANDS, main
from tabbiomed.analysis.datasets import load_metadata
from tabbiomed.analysis.elo.api import apply_missing_policy, rate
from tabbiomed.analysis.reproduce import paper_commands
from tabbiomed.analysis.results import read_result, valid_split_identity
from tabbiomed.data import catalog


class AnalysisWorkflow(unittest.TestCase):
    def test_local_results_never_download_published_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.dict(os.environ, {"TBM_RESULTS": str(root)}):
                with patch.object(downloads, "download_results") as download:
                    self.assertEqual(downloads.ensure_results(root), root)
                    download.assert_not_called()
                    with self.assertRaises(FileNotFoundError):
                        downloads.ensure_results(root / "missing")
                    self.assertFalse((root / "missing").exists())

    def test_cli_sets_results_before_importing_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            settings = {}

            def load(name):
                settings["module"] = name
                settings["results"] = os.environ["TBM_RESULTS"]

                def generate():
                    settings["arguments"] = list(sys.argv)

                return SimpleNamespace(main=generate)

            with (
                patch.dict(os.environ),
                patch("tabbiomed.analysis.__main__.import_module", side_effect=load),
            ):
                main(["elo", "--results", str(root), "--output", str(root / "plots")])
            self.assertEqual(settings["module"], "tabbiomed.analysis.figures.elo")
            self.assertEqual(settings["results"], str(root))
            self.assertEqual(
                settings["arguments"][1:], ["--output", str(root / "plots")]
            )

    def test_shared_configuration_applies_before_every_analysis_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for command in COMMANDS:
                with self.subTest(command=command):
                    settings = {}

                    def load(name):
                        settings.update(
                            module=name,
                            plan=os.environ["TBM_ANALYSIS_PLAN"],
                            families=os.environ["TBM_MODEL_FAMILIES"],
                        )
                        return SimpleNamespace(main=lambda: None)

                    with (
                        patch.dict(os.environ),
                        patch(
                            "tabbiomed.analysis.__main__.import_module",
                            side_effect=load,
                        ),
                    ):
                        main(
                            [
                                command,
                                "--plan",
                                str(root / "scope.json"),
                                "--model-families",
                                str(root / "families.json"),
                            ]
                        )
                    self.assertEqual(settings["plan"], str(root / "scope.json"))
                    self.assertEqual(settings["families"], str(root / "families.json"))

    def test_model_scope_uses_plan_and_rejects_missing_families(self):
        from tabbiomed.analysis.common import load_benchmark_models_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            families = dict(
                model_family={
                    "RandomForest": "Tree-based",
                    "NewDefault": "Foundation models",
                    "ClassificationOnly": "Foundation models",
                    "Unused": "Baselines",
                },
                exclude_from_expected_benchmark=[],
                regression_optional_missing_models=["ClassificationOnly"],
            )
            plan = dict(
                models={
                    name: {}
                    for name in ("RandomForest", "NewDefault", "ClassificationOnly")
                }
            )
            family_path = root / "families.json"
            plan_path = root / "scope.json"
            family_path.write_text(json.dumps(families))
            plan_path.write_text(json.dumps(plan))
            config = load_benchmark_models_config(family_path, plan_path=plan_path)
            self.assertEqual(
                set(config.expected_models_binary_classification), set(plan["models"])
            )
            self.assertEqual(
                set(config.expected_models_regression), {"RandomForest", "NewDefault"}
            )
            self.assertNotIn("Unused", config.expected_models_binary_classification)
            plan["models"]["MissingFamily"] = {}
            bad_plan = root / "unknown.json"
            bad_plan.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "missing a family assignment"):
                load_benchmark_models_config(family_path, plan_path=bad_plan)

    def test_observation_groups_preserve_membership_in_dataset_subsets(self):
        from tabbiomed.analysis.figures.elo_domains import observation_partition

        groups = {"shared": ["a", "b"], "outside": ["c", "d"]}
        partition = observation_partition(["a", "e"], groups)
        self.assertEqual(partition, {"a": "shared", "e": "singleton::e"})
        with self.assertRaisesRegex(ValueError, "Duplicate shared-observation"):
            observation_partition(["a"], {"one": ["a"], "two": ["a"]})

    def test_paper_profile_keeps_packaged_scope(self):
        from tabbiomed.analysis import reproduce
        from tabbiomed.analysis.paths import CONFIG

        with (
            patch.dict(
                os.environ,
                {
                    "TBM_ANALYSIS_PLAN": "/tmp/custom-scope.json",
                    "TBM_MODEL_FAMILIES": "/tmp/custom-families.json",
                },
            ),
            patch.object(reproduce, "paper_commands", return_value=[("example",)]),
            patch.object(reproduce.subprocess, "run") as run,
        ):
            reproduce.main([])
        environment = run.call_args.kwargs["env"]
        self.assertEqual(
            environment["TBM_ANALYSIS_PLAN"], str(CONFIG / "benchmark.json")
        )
        self.assertEqual(
            environment["TBM_MODEL_FAMILIES"], str(CONFIG / "model_families.json")
        )

    def test_cli_short_output_option_preserves_requested_destination(self):
        for command in ("dataset-figures", "elo-regimes", "runtime", "upset"):
            for argument in (
                ("-o", "/tmp/requested-plots"),
                ("-o/tmp/requested-plots",),
            ):
                with self.subTest(command=command, argument=argument):
                    arguments = []
                    implementation = SimpleNamespace(
                        main=lambda: arguments.extend(sys.argv[1:])
                    )
                    with patch(
                        "tabbiomed.analysis.__main__.import_module",
                        return_value=implementation,
                    ):
                        main([command, *argument])
                    self.assertEqual(
                        arguments,
                        ["--output-dir", str(Path("/tmp/requested-plots").resolve())],
                    )

    def test_numerical_elo_does_not_import_plotting(self):
        script = "from tabbiomed.analysis.elo.api import rate; import sys; assert 'matplotlib' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_metadata_access_does_not_load_feature_arrays(self):
        metadata = {"task": "binary", "num_features": 2}
        with patch("tabbiomed.data.storage.load_metadata", return_value=metadata):
            with patch("tabbiomed.data.storage.load_arrays") as arrays:
                self.assertEqual(load_metadata("example"), metadata)
                arrays.assert_not_called()

    def test_elo_weights_each_dataset_equally_across_outer_folds(self):
        rows = []
        for dataset, folds, contender in (("cv", 5, 0.7), ("fixed", 1, 0.9)):
            for fold in range(folds):
                for competitor, score in (
                    ("RandomForest::default", 0.8),
                    ("NewModel::tuned", contender),
                ):
                    rows.append(
                        dict(
                            dataset=dataset,
                            fold=fold,
                            task="binary",
                            competitor=competitor,
                            metric_value=score,
                        )
                    )
        table, _, _ = rate(pd.DataFrame(rows), bootstrap=10)
        np.testing.assert_allclose(table.elo, 1000.0, atol=1e-8)

    def test_missing_tuned_uses_random_forest_default(self):
        rows = []
        for dataset, contender in (("a", 0.7), ("b", np.nan)):
            for competitor, score in (
                ("RandomForest::default", 0.8),
                ("RandomForest::tuned", 0.9),
                ("NewModel::tuned", contender),
            ):
                rows.append(
                    dict(
                        dataset=dataset,
                        fold=0,
                        task="binary",
                        competitor=competitor,
                        metric_value=score,
                    )
                )
        frame, details = apply_missing_policy(pd.DataFrame(rows), "rf_imputed")
        selected = frame[
            frame.dataset.eq("b") & frame.competitor.eq("NewModel::tuned")
        ].iloc[0]
        self.assertEqual(selected.metric_value, 0.8)
        self.assertTrue(selected.imputed)
        self.assertEqual(details["imputed_cells"], 1)

    def test_new_model_reader_uses_selected_tuned_despite_lower_test_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "NewModel/example"
            directory.mkdir(parents=True)
            report = dict(
                artifact_type="evaluation_summary",
                schema_version=1,
                hpo_enabled=True,
                model="NewModel",
                dataset="example",
                primary_metric="roc_auc",
                default=dict(
                    fixed_split_test_evaluated=True, test_metrics={"roc_auc": 0.8}
                ),
                best=dict(
                    fixed_split_test_evaluated=True, test_metrics={"roc_auc": 0.7}
                ),
            )
            (directory / "report.json").write_text(json.dumps(report))
            row = read_result(
                "NewModel", "example", dict(task="binary", n_folds=0), root=root
            )
            self.assertEqual(row["selected_variant"], "tuned")
            self.assertEqual(row["metric_value"], 0.7)

    def test_new_default_only_model_has_no_fabricated_tuned_competitor(self):
        from tabbiomed.analysis.figures.pareto import expected_competitors
        from tabbiomed.analysis.figures.elo import read_consolidated_pair
        from tabbiomed.analysis.figures.upset import select_upset_scores

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "NewDefault/example"
            directory.mkdir(parents=True)
            report = dict(
                artifact_type="evaluation_summary",
                schema_version=1,
                model="NewDefault",
                dataset="example",
                hpo_enabled=False,
                primary_metric="roc_auc",
                best=None,
                default=dict(
                    fixed_split_test_evaluated=True, test_metrics={"roc_auc": 0.8}
                ),
            )
            (directory / "report.json").write_text(json.dumps(report))
            info = dict(task="binary", n_folds=0)
            row = read_result("NewDefault", "example", info, root=root)
            self.assertEqual(row["metric_source"], "default_only")
            selected = select_upset_scores(pd.DataFrame([row]))
            self.assertEqual(selected.metric_source.iloc[0], "default_only")
            pair = read_consolidated_pair(
                root, dict(model="NewDefault", dataset="example"), info
            )
            self.assertFalse(pair["hpo_enabled"])
            self.assertNotIn("tuned", pair["split_results"])
            self.assertEqual(
                expected_competitors(["NewDefault"], {"NewDefault"}),
                ["NewDefault::default"],
            )

    def test_split_identity_accepts_equal_contents_across_serializations(self):
        from tabbiomed.data.identity import split_digest
        import hashlib

        lists = dict(train=[0, 1, 2], val=[3], test=[4])
        arrays = {
            key: np.asarray(value, dtype=np.int32) for key, value in lists.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root / "lists.npy", lists)
            np.save(root / "arrays.npy", arrays)
            first = hashlib.sha256((root / "lists.npy").read_bytes()).hexdigest()
            second = hashlib.sha256((root / "arrays.npy").read_bytes()).hexdigest()
        self.assertNotEqual(first, second)
        fingerprint = split_digest(lists, 5, 0)
        self.assertEqual(fingerprint, split_digest(arrays, 5, 0))
        info = dict(splits_sha256=fingerprint, recorded_split_sha256=first)
        self.assertTrue(
            valid_split_identity({"dataset_split_sha256": fingerprint}, info)
        )
        self.assertTrue(valid_split_identity({"dataset_split_sha256": first}, info))
        self.assertFalse(valid_split_identity({"dataset_split_sha256": second}, info))
        changed = dict(train=[0, 1, 3], val=[2], test=[4])
        self.assertFalse(
            valid_split_identity(
                {"dataset_split_sha256": split_digest(changed, 5, 0)}, info
            )
        )
        self.assertFalse(valid_split_identity({}, info))

    def test_analysis_scopes_work_with_different_model_counts(self):
        from dataclasses import replace
        from tabbiomed.analysis.common import load_benchmark_models_config
        from tabbiomed.analysis.figures.elo_regimes import CASES, models_for_case
        from tabbiomed.analysis.ranks import friedman_global_all_tasks

        models = ("RandomForest", "LinearModel", "NeoLsSVM")
        classification_models = (*models, "HyperFast")
        self.assertEqual(set(CASES), {"all_tasks", "classification"})
        self.assertEqual(
            models_for_case(list(classification_models), "all_tasks"), sorted(models)
        )
        self.assertEqual(
            models_for_case(list(classification_models), "classification"),
            sorted(classification_models),
        )
        config = replace(
            load_benchmark_models_config(),
            expected_models_regression=models,
            expected_models_binary_classification=classification_models,
        )
        frame = pd.DataFrame(
            [
                dict(dataset=dataset, task=task, model=model, metric_value=score)
                for dataset, task, scores in (
                    ("binary", "binary", (0.6, 0.7, 0.8)),
                    ("regression", "regression", (3.0, 2.0, 1.0)),
                )
                for model, score in zip(models, scores)
            ]
        )
        result = friedman_global_all_tasks(frame, config)
        self.assertEqual(result["scope_tag"], "global_all_tasks")
        self.assertEqual(result["n_models"], 3)
        self.assertEqual(result["n_datasets"], 2)
        self.assertEqual(result["average_ranks"], dict(zip(models, (3.0, 2.0, 1.0))))

    def test_calibration_arguments_forward_dataset_and_models(self):
        from tabbiomed.analysis.figures import calibration

        models = ["ExampleA", "ExampleB", "ExampleC", "ExampleD", "ExampleE"]
        with (
            patch.object(
                sys,
                "argv",
                [
                    "calibration",
                    "--dataset",
                    "example",
                    "--models",
                    *models,
                    "--n-bins",
                    "12",
                    "--strategy",
                    "quantile",
                    "--variant",
                    "default",
                    "--output-dir",
                    "/tmp/calibration",
                ],
            ),
            patch.object(calibration, "ensure_results"),
            patch.object(calibration, "paper_style"),
            patch.object(calibration, "make_figure") as generate,
        ):
            calibration.main()
        self.assertEqual(generate.call_count, 2)
        for call, histogram in zip(generate.call_args_list, (False, True)):
            self.assertEqual(
                call.args,
                (
                    "example",
                    "example",
                    models,
                    12,
                    "quantile",
                    histogram,
                    "default",
                    Path("/tmp/calibration"),
                ),
            )

    def test_calibration_layout_accepts_one_or_many_models(self):
        from tabbiomed.analysis.figures import calibration

        def reliability(dataset, model, n_bins, strategy, variant):
            return (
                np.array([0.25, 0.75]),
                np.array([0.2, 0.8]),
                np.array([0.1, 0.25, 0.75, 0.9]),
                False,
                dict(model=model, variant_used=variant, n_test=4),
            )

        for count in (1, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                models = [f"Example{i}" for i in range(count)]
                with patch.object(calibration, "reliability", side_effect=reliability):
                    calibration.make_figure(
                        "example",
                        "example",
                        models,
                        20,
                        "uniform",
                        True,
                        "best",
                        output,
                    )
                data = json.loads(
                    (output / "calibration_example_hist.json").read_text()
                )
                self.assertEqual([row["model"] for row in data["panels"]], models)
                self.assertTrue((output / "calibration_example_hist.pdf").is_file())
                self.assertTrue((output / "calibration_example_hist.png").is_file())

    def test_paper_assets_and_reproduction_stages_are_packaged(self):
        from tabbiomed.analysis.paths import CONFIG, TEMPLATES

        profile = json.loads((CONFIG / "benchmark.json").read_text())
        self.assertEqual(set(profile["datasets"]), set(catalog()))
        self.assertEqual(len(profile["models"]), 23)
        from tabbiomed.models import BENCHMARK_MODELS

        self.assertEqual(set(profile["models"]), set(BENCHMARK_MODELS))
        self.assertTrue((TEMPLATES / "pca_deltas.tex").is_file())
        mapping = json.loads((CONFIG / "artifacts.json").read_text())
        self.assertTrue(mapping)
        package = Path(__import__("tabbiomed").__file__).parent
        for item in mapping:
            for source in item["sources"]:
                path = Path(source)
                self.assertEqual(path.parts[0], "tabbiomed")
                self.assertTrue((package.joinpath(*path.parts[1:])).is_file(), source)
        stages = [command[0] for command in paper_commands(Path("outputs"))]
        self.assertEqual(len(stages), 25)
        self.assertLess(
            stages.index("tabbiomed.analysis.studies.hpo_history"),
            stages.index("tabbiomed.analysis.studies.hpo_sensitivity"),
        )


if __name__ == "__main__":
    unittest.main()
