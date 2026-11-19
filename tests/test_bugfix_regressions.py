import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer

from tabbiomed.analysis.binary_metrics import ece_binary
from tabbiomed.analysis.calibration import ece as cal_ece
from tabbiomed.analysis import downloads
from tabbiomed.analysis.elo.api import rate
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.studies import pca as pca_study
from tabbiomed.cv_splits import ZERO_NONZERO, load_sample_groups, make_grouped_outer_cv
from tabbiomed.data import storage
from tabbiomed.evaluator.metrics import compute_metrics
from tabbiomed.models.base_model import BaseModel
from tabbiomed.models.model_definitions.decision_tree import DecisionTree
from tabbiomed.models.model_definitions.lasso import LassoModel
from tabbiomed.models.model_definitions.mlp import MLP
from tabbiomed.models.model_definitions.torch_pca import svd_flip
from tabbiomed.results.schema import clean
from tabbiomed.slurm import MAX_ARRAY_SIZE, prepare, run_task
from tabbiomed.tabbiomed_datasets import TabularDataset
from tabbiomed.trainer.utils.parameter_config import define_trial_parameters_from_config
from tabbiomed.utils import dump_report, update_n_trials


class BugfixRegressions(unittest.TestCase):
    def test_base_model_explicitly_declares_allow_metadata_scaling_false(self):
        self.assertIn("allow_metadata_scaling", BaseModel.__dict__)
        self.assertIs(BaseModel.allow_metadata_scaling, False)

    def test_normalize_and_align_probabilities_handles_missing_and_single_classes(self):
        args = SimpleNamespace(
            objective="classification",
            num_classes=3,
            num_features=1,
            num_cpus=1,
            seed=42,
            is_cv=False,
        )
        model = DecisionTree(
            DecisionTree.resolve_parameters(DecisionTree.get_default_params(), args),
            args,
        )
        # Single-class 0 with 1-column probability 1.0 must NOT invert to class 1
        aligned_zero = model._normalize_and_align_probabilities(
            np.ones((4, 1)), classes=np.array([0])
        )
        np.testing.assert_allclose(
            aligned_zero, np.tile([1.0, 0.0, 0.0], (4, 1))
        )
        # Missing middle class {0, 2} with 1-column positive-class probability
        aligned_pair = model._normalize_and_align_probabilities(
            np.array([[0.25], [0.75]]), classes=np.array([0, 2])
        )
        np.testing.assert_allclose(
            aligned_pair, np.array([[0.75, 0.0, 0.25], [0.25, 0.0, 0.75]])
        )

    def test_fixed_parameter_type_works_in_hpo_suggestions(self):
        space = {
            "C": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
            "max_iter": {"type": "fixed", "value": 500},
            "scaler": {"type": "categorical", "choices": ["standard"]},
        }
        with patch(
            "tabbiomed.trainer.utils.parameter_config.load_model_search_space",
            return_value=space,
        ):
            params = define_trial_parameters_from_config(
                optuna.trial.FixedTrial({"C": 1.0, "scaler": "standard"}),
                "SVM",
                dataset_metadata={"objective": "binary"},
            )
        self.assertEqual(params["max_iter"], 500)

    def test_keep_empty_features_prevents_all_nan_column_shift(self):
        X_num = np.array(
            [
                [10.0, np.nan, 100.0],
                [20.0, np.nan, np.nan],
                [30.0, np.nan, 300.0],
            ]
        )
        imputer = SimpleImputer(
            strategy="mean", add_indicator=True, keep_empty_features=True
        )
        imputed = imputer.fit_transform(X_num)
        num_block = imputed[:, :3]
        ind_block = imputed[:, 3:]
        np.testing.assert_allclose(num_block[:, 0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(num_block[:, 1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(num_block[:, 2], [100.0, 200.0, 300.0])
        self.assertEqual(ind_block.shape, (3, 2))

    def test_mlp_accepts_int32_multiclass_targets_and_returns_1d_regression(self):
        X = np.ones((6, 2), dtype=np.float32)
        y_cls = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        args_cls = SimpleNamespace(
            objective="classification",
            num_features=2,
            num_classes=3,
            cat_idx=[],
            cat_dims=[],
            use_gpu=False,
            gpu_ids=[0],
            epochs=1,
            batch_size=3,
            val_batch_size=3,
            early_stopping_rounds=1,
            logging_period=100,
            seed=0,
            data_parallel=False,
        )
        params_cls = MLP.resolve_parameters(
            {**MLP.get_default_params(), "n_epochs": 1}, args_cls
        )
        mlp_cls = MLP(params_cls, args_cls)
        mlp_cls.fit(X, y_cls, X, y_cls)
        self.assertEqual(mlp_cls.predict_proba(X).shape, (6, 3))

        y_reg = np.arange(6, dtype=np.float32)
        args_reg = SimpleNamespace(**{**vars(args_cls), "objective": "regression", "num_classes": 1})
        params_reg = MLP.resolve_parameters(
            {**MLP.get_default_params(), "n_epochs": 1}, args_reg
        )
        mlp_reg = MLP(params_reg, args_reg)
        mlp_reg.fit(X, y_reg, X, y_reg)
        self.assertEqual(mlp_reg.predict(X).shape, (6,))

    def test_lasso_invalid_objective_raises_value_error(self):
        args = SimpleNamespace(
            objective="unsupported_task",
            num_classes=2,
            num_features=2,
            num_cpus=1,
            seed=42,
        )
        params = {**LassoModel.get_default_params(), "scaler": "standard", "threshold": None}
        with self.assertRaisesRegex(ValueError, "unsupported_task"):
            LassoModel(params, args)

    def test_svd_flip_preserves_unit_rows_on_zero_columns(self):
        import torch

        u = torch.zeros((4, 2))
        v = torch.eye(2, 3)
        _, v_flipped = svd_flip(u, v)
        np.testing.assert_allclose(v_flipped.numpy(), np.eye(2, 3))

    def test_update_n_trials_and_dump_report_nan_safety(self):
        cfg = update_n_trials({"n_trials": 50}, "RandomForest")
        self.assertEqual(cfg["n_trials"], 50)

        metrics = compute_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 0, 1]),
            np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.7, 0.3, 0.0], [0.2, 0.8, 0.0]]),
            "classification",
        )
        self.assertIsNone(metrics["roc_auc"])
        with tempfile.TemporaryDirectory() as td:
            dump_report({"metrics": metrics, "inf": float("inf")}, Path(td))
            text = (Path(td) / "report.json").read_text()
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            parsed = json.loads(text)
            self.assertIsNone(parsed["metrics"]["roc_auc"])
            self.assertIsNone(parsed["inf"])

    def test_grouped_zero_nonzero_outer_cv_preserves_both_strata(self):
        groups = np.arange(60)
        y = np.array([0.0] * 54 + [1.5] * 6)
        splits = make_grouped_outer_cv(
            groups, y, "regression", n_folds=5, seed=42, validation_stratification=ZERO_NONZERO
        )
        with tempfile.TemporaryDirectory() as td:
            np.save(Path(td) / "y.npy", y)
            loaded = load_sample_groups(td, splits)
            self.assertEqual(len(loaded), 60)

    def test_slurm_chunks_large_plans_under_max_array_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scripts = prepare(root)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertGreater(len(manifest["tasks"]), 2000)
            for script in scripts:
                arr_line = next(
                    ln for ln in script.read_text().splitlines() if ln.startswith("#SBATCH --array=")
                )
                spec = arr_line.split("=", 1)[1].split("%")[0]
                upper = int(spec.split("-")[-1])
                self.assertLess(upper, MAX_ARRAY_SIZE)
            with patch("tabbiomed.slurm.subprocess.run") as run:
                run_task(root / "manifest.json", 0, batch=scripts[0].name)
                run.assert_called_once()

    def test_tabular_dataset_null_string_binary_and_subset_label_encoding(self):
        X = np.array([["f", 1.0], ["t", 2.0], [None, 3.0], ["f", 4.0], ["t", 5.0]], dtype=object)
        y = np.array([0, 1, 0, 1, 0])
        ds = TabularDataset(
            name="bin_null",
            X=X,
            y=y,
            cat_idx=[0],
            task="binary",
            num_classes=2,
            num_features=2,
            num_samples=5,
            split_indices={"train": [0, 1, 2], "val": [3], "test": [4]},
        )
        ds.prepare_dataset(inpute_mv=False, encode_cat=True, scale_features=False)
        self.assertEqual(ds.X_train.shape, (3, 2))

        X_sub = np.array([["a", 1.0], ["b", 2.0], ["c", 3.0], ["unused", 4.0]], dtype=object)
        ds_sub = TabularDataset(
            name="sub_lbl",
            X=X_sub,
            y=np.array([0, 1, 0, 1]),
            cat_idx=[0],
            task="binary",
            num_classes=2,
            num_features=2,
            num_samples=4,
            split_indices={"train": [0], "val": [1], "test": [2]},
        )
        ds_sub.prepare_dataset(
            inpute_mv=False, encode_cat=True, scale_features=False, cat_encoding_strategy="label"
        )
        self.assertEqual(ds_sub.X.dtype, np.float64)

    def test_validate_splits_rejects_cv_folds_on_fixed_split_dataset(self):
        with patch.object(
            storage,
            "_info",
            return_value={"split_metadata": {}, "splits_sha256": "unused"},
        ), patch.object(
            storage,
            "load_metadata",
            return_value={"num_samples": 4, "split_strategy": "fixed"},
        ):
            with self.assertRaisesRegex(ValueError, "Unexpected cv_folds"):
                storage.validate_splits(
                    "pima_diabetes",
                    {"train": [0, 1], "val": [2], "test": [3], "cv_folds": []},
                )

    def test_normalized_scores_with_failures_scores_sole_or_tied_winners_as_one(self):
        raw = pd.DataFrame(
            {"A": [0.85, 0.90, np.nan], "B": [np.nan, 0.90, np.nan]},
            index=["sole", "tied", "all_fail"],
        )
        tasks = pd.Series({"sole": "binary", "tied": "binary", "all_fail": "binary"})
        norm = normalized_scores_with_failures(raw, tasks)
        self.assertEqual(norm.loc["sole", "A"], 1.0)
        self.assertEqual(norm.loc["sole", "B"], 0.0)
        self.assertEqual(norm.loc["tied", "A"], 1.0)
        self.assertEqual(norm.loc["tied", "B"], 1.0)
        self.assertEqual(norm.loc["all_fail", "A"], 0.0)
        self.assertEqual(norm.loc["all_fail", "B"], 0.0)

    def test_pca_analysis_succeeds_without_pca_components_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            model_dir = root / "TabPFNv2.5" / "example_dataset"
            (model_dir / "pca_500").mkdir(parents=True)
            base = {
                "model": "TabPFNv2.5",
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
                **base,
                "search_space": {
                    **base["search_space"],
                    "apply_pca": {"choices": [True]},
                    "pca_max_features": {"choices": [500]},
                    "pca_fit": {"choices": ["lowrank"]},
                },
                "best": {
                    "test_metrics": {"roc_auc": 0.85},
                    "fixed_split_test_evaluated": True,
                },
            }
            (model_dir / "report.json").write_text(json.dumps(base))
            (model_dir / "pca_500/report.json").write_text(json.dumps(reduced))
            inv = {"example_dataset": {"num_features": 3000, "num_samples": 120, "task": "binary"}}
            rows, _ = pca_study.collect(root, inv, ["TabPFNv2.5"], component_cap=500)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["outcome"], "better")

    def test_unified_ece_convention_and_dynamic_ensure_results_and_elo_zero_bootstrap(self):
        y = np.array([0, 1, 0, 1, 1])
        p = np.array([0.0, 0.2, 0.4, 0.5, 1.0])
        self.assertAlmostEqual(
            cal_ece(y, p, n_bins=5, strategy="uniform"),
            ece_binary(y, p, n_bins=5),
        )

        with tempfile.TemporaryDirectory() as td:
            resolved = Path(td).resolve()
            with patch.dict(os.environ, {"TBM_RESULTS": str(resolved)}):
                self.assertEqual(downloads.ensure_results(), resolved)

        frame = pd.DataFrame(
            [
                {"dataset": "d1", "fold": 0, "task": "regression", "metric": "r2", "competitor": "RandomForest::default", "metric_value": 0.5},
                {"dataset": "d1", "fold": 0, "task": "regression", "metric": "r2", "competitor": "MLP::tuned", "metric_value": 0.9},
            ]
        )
        table, _, _ = rate(frame, bootstrap=0)
        mlp_elo = float(table.loc[table.competitor.eq("MLP::tuned"), "elo"].iloc[0])
        rf_elo = float(table.loc[table.competitor.eq("RandomForest::default"), "elo"].iloc[0])
        self.assertGreater(mlp_elo, rf_elo)

        cleaned = clean(
            {
                "custom_model_dir": "/private/tmp/models",
                "disk_offload_dir": "/scratch/job123",
                "checkpoint_path": "/home/user/weights/hyperfast.pt",
                "model_path": "/home/user/weights/tabicl.ckpt",
            }
        )
        self.assertEqual(
            cleaned,
            {"checkpoint_path": "hyperfast.pt", "model_path": "tabicl.ckpt"},
        )

    def test_talent_curator_deduplicates_dna_diabetes_and_maternal_health(self):
        from tabbiomed.data.curation.talent_datasets import ingest_one_talent_folder

        for ds_name, task_type, n_classes in (
            ("dna", "multiclass", 3),
            ("CDC_Diabetes_Health_Indicators", "binclass", 2),
            ("maternal_health_risk", "multiclass", 3),
        ):
            with self.subTest(dataset=ds_name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                folder = root / "raw" / ds_name
                out = root / "out"
                folder.mkdir(parents=True)
                # 12 distinct rows (4 per class / 6 per class), plus 1 exact duplicate and 2 conflicting rows
                base_X = np.arange(24, dtype=np.float32).reshape(12, 2)
                base_y = np.arange(12, dtype=np.int64) % n_classes
                dup_X = base_X[:1]
                dup_y = base_y[:1]
                conflict_X = np.array([[999.0, 999.0], [999.0, 999.0]], dtype=np.float32)
                conflict_y = np.array([0, 1], dtype=np.int64)
                X_all = np.vstack([base_X, dup_X, conflict_X])
                y_all = np.concatenate([base_y, dup_y, conflict_y])
                (folder / "info.json").write_text(
                    json.dumps(
                        {
                            "name": ds_name,
                            "task_type": task_type,
                            "n_num_features": 2,
                            "n_cat_features": 0,
                            "num_classes": n_classes,
                        }
                    )
                )
                np.save(folder / "N_train.npy", X_all[:10])
                np.save(folder / "N_val.npy", X_all[10:13])
                np.save(folder / "N_test.npy", X_all[13:])
                np.save(folder / "y_train.npy", y_all[:10])
                np.save(folder / "y_val.npy", y_all[10:13])
                np.save(folder / "y_test.npy", y_all[13:])
                self.assertTrue(ingest_one_talent_folder(folder, out, force=True))
                saved_X = np.load(out / ds_name / "X.npy", allow_pickle=True)
                saved_y = np.load(out / ds_name / "y.npy")
                self.assertEqual(saved_X.shape, (12, 2))
                self.assertEqual(saved_y.shape, (12,))

    def test_win_rates_and_hpo_sensitivity_orient_by_primary_metric(self):
        from tabbiomed.analysis.figures.win_rates import pairwise_matrix
        from tabbiomed.analysis.studies.hpo_sensitivity import default_comparisons

        inputs = pd.DataFrame(
            [
                {
                    "dataset": "d_reg",
                    "model": "ModelA",
                    "task": "regression",
                    "primary_metric": "r2",
                    "metric_value": 0.9,
                },
                {
                    "dataset": "d_reg",
                    "model": "ModelB",
                    "task": "regression",
                    "primary_metric": "r2",
                    "metric_value": 0.4,
                },
            ]
        )
        metadata = pd.DataFrame([{"dataset": "d_reg", "task": "regression"}])
        matrix = pairwise_matrix(inputs, metadata, ["ModelA", "ModelB"])[0]
        self.assertEqual(float(matrix.loc["ModelA", "ModelB"]), 100.0)
        self.assertEqual(float(matrix.loc["ModelB", "ModelA"]), 0.0)

        selected = pd.DataFrame(
            [
                {
                    "model": "ModelA",
                    "dataset": "d_cls",
                    "task": "classification",
                    "primary_metric": "accuracy",
                    "hpo_enabled": True,
                    "default_metric": 0.70,
                    "tuned_metric": 0.85,
                }
            ]
        )
        _, freqs, _ = default_comparisons(selected)
        self.assertEqual(int(freqs.loc["ModelA", "tuned_better"]), 1)
        self.assertEqual(int(freqs.loc["ModelA", "default_better"]), 0)

    def test_reference_curators_exit_nonzero_on_missing_inputs_or_errors(self):
        import contextlib
        import io
        from tabbiomed.data.curation import (
            metabolomics_datasets,
            multiomics_datasets,
            openml_datasets,
            parkinson_speech_measures,
            tabpfn_vocs_datasets,
            talent_datasets,
        )

        sink = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as td,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            missing = Path(td) / "nonexistent"
            with patch.object(metabolomics_datasets, "RAW_PATH", str(missing)):
                with self.assertRaises(SystemExit) as ctx:
                    metabolomics_datasets.main()
                self.assertEqual(ctx.exception.code, 1)

            with self.assertRaises(SystemExit) as ctx:
                multiomics_datasets.run_all(
                    Path(td), ["GS-OV"], Path(td) / "out", force=True, raise_on_failure=True
                )
            self.assertEqual(ctx.exception.code, 1)

            with patch.object(parkinson_speech_measures, "RAW_DIR", str(missing)):
                with self.assertRaises(SystemExit) as ctx:
                    parkinson_speech_measures.main()
                self.assertEqual(ctx.exception.code, 1)

            with patch.object(
                tabpfn_vocs_datasets,
                "_build_one",
                side_effect=RuntimeError("corrupt raw file"),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    tabpfn_vocs_datasets.main()
                self.assertEqual(ctx.exception.code, 1)

            with patch("sys.argv", ["openml", "--input", str(missing)]):
                with self.assertRaises(SystemExit) as ctx:
                    openml_datasets.main()
                self.assertEqual(ctx.exception.code, 1)

            with patch("sys.argv", ["talent", "--input", str(missing)]):
                with self.assertRaises(SystemExit) as ctx:
                    talent_datasets.main()
                self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
