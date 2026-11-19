import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tabbiomed.data import catalog as read_catalog, ensure_dataset, verify_dataset
from tabbiomed.data import storage
from tabbiomed.data.identity import (
    METADATA_FIELDS,
    array_digest,
    json_digest,
    split_digest,
    text_columns,
)


class DatasetWorkflow(unittest.TestCase):
    def fixture(self, root, *, cv=False):
        name = "example"
        if cv:
            X = np.arange(20, dtype=float).reshape(10, 2)
            X[1, 0] = np.nan
            y = np.column_stack((np.arange(10), -np.arange(10))).astype(float)
            groups = np.repeat(np.arange(5), 2).tolist()
            folds = []
            for fold in range(5):
                tests = [2 * fold, 2 * fold + 1]
                vals = [2 * ((fold + 1) % 5), 2 * ((fold + 1) % 5) + 1]
                train = sorted(set(range(10)) - set(tests + vals), reverse=True)
                folds.append(dict(train=train, val=vals, test=tests))
            splits = dict(cv_folds=folds, groups=groups)
            metadata = dict(
                name=name,
                READY=True,
                task="multiregression",
                num_samples=10,
                num_features=2,
                cat_idx=[],
                split_strategy="cv",
                split_group="individual",
            )
        else:
            X = np.array(
                [[1.0, "a"], [np.nan, "b"], [3.0, None], [4.0, "a"], [5.0, "c"]],
                dtype=object,
            )
            y = np.array([0, 1, 0, 1, 0])
            splits = dict(train=[3, 0, 2], val=[1], test=[4])
            metadata = dict(
                name=name,
                READY=True,
                task="binary",
                num_classes=2,
                num_samples=5,
                num_features=2,
                cat_idx=[1],
            )
        info = dict(
            metadata=metadata,
            metadata_sha256=json_digest(
                {key: metadata.get(key) for key in METADATA_FIELDS}
            ),
            features_sha256=array_digest(X, text_columns(X)),
            targets_sha256=array_digest(y),
            splits_sha256=split_digest(splits, len(y), 5 if cv else 0),
        )
        source = root / "parquet" / name
        source.mkdir(parents=True)
        (source / "metadata.json").write_text(json.dumps(metadata))
        if cv:
            (source / "split_groups.json").write_text(json.dumps(dict(groups=groups)))
        for fold, parts in enumerate(splits["cv_folds"] if cv else [splits]):
            for part in ["train", "val", "test"]:
                rows = parts[part]
                directory = source / "data" / f"fold_{fold}" if cv else source / "data"
                directory.mkdir(parents=True, exist_ok=True)
                fields = dict(
                    row_idx=rows, f_0=X[rows, 0].astype(float), f_1=X[rows, 1]
                )
                if cv:
                    fields.update(target_0=y[rows, 0], target_1=y[rows, 1])
                else:
                    fields["target"] = y[rows]
                pq.write_table(pa.table(fields), directory / (part + ".parquet"))
        return name, X, y, splits, metadata, {name: info}

    def roundtrip(self, cv):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, X, y, splits, metadata, registry = self.fixture(root, cv=cv)
            with (
                patch("tabbiomed.data.catalog.catalog", return_value=registry),
                patch.object(storage, "benchmark_datasets", return_value=registry),
                patch.object(storage, "PARQUET", root / "parquet"),
                patch.object(storage, "DATA_CACHE", root / "cache"),
            ):
                target = ensure_dataset(name, root / "processed")
                actual_X = np.load(target / "X.npy", allow_pickle=True)
                actual_y = np.load(target / "y.npy", allow_pickle=True)
                self.assertEqual(
                    array_digest(actual_X, text_columns(actual_X)),
                    array_digest(X, text_columns(X)),
                )
                np.testing.assert_array_equal(actual_y, y)
                self.assertEqual(
                    np.load(target / "split_indices.npy", allow_pickle=True).item(),
                    splits,
                )
                self.assertEqual(verify_dataset(name, root / "processed"), metadata)
                changed = actual_X.copy()
                changed[0, 0] = 1000
                np.save(target / "X.npy", changed)
                with self.assertRaisesRegex(ValueError, "values or row/column order"):
                    verify_dataset(name, root / "processed")

    def test_fixed_parquet_restores_categorical_values_and_row_order(self):
        self.roundtrip(False)

    def test_cv_parquet_restores_multioutput_labels_and_groups(self):
        self.roundtrip(True)

    def test_dataset_metadata_name_must_match_the_requested_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, _, _, _, metadata, registry = self.fixture(root)
            source = root / "parquet"
            with patch.object(storage, "benchmark_datasets", return_value=registry):
                self.assertEqual(storage.load_metadata(name, source), metadata)
                for value in ("other", "", None, 42):
                    with self.subTest(name=value):
                        changed = dict(metadata, name=value)
                        (source / name / "metadata.json").write_text(json.dumps(changed))
                        with self.assertRaisesRegex(ValueError, "Dataset name"):
                            storage.load_metadata(name, source)
                changed = dict(metadata)
                del changed["name"]
                with self.assertRaisesRegex(ValueError, "Dataset name"):
                    storage._validate_metadata(name, changed)

    def test_training_metadata_is_part_of_data_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, _, _, _, metadata, registry = self.fixture(root)
            with patch.object(storage, "benchmark_datasets", return_value=registry):
                with self.assertRaisesRegex(ValueError, "preprocessing metadata"):
                    storage._validate_metadata(name, dict(metadata, force_scale=True))
                with self.assertRaisesRegex(ValueError, "preprocessing metadata"):
                    storage._validate_metadata(name, dict(metadata, cat_dims=[3]))

    def test_local_splits_require_the_recorded_split_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, _, _, splits, metadata, registry = self.fixture(root, cv=True)
            policy = dict(
                group_split_protocol="grouped_outer_partition_v1",
                validation_stratification="zero_vs_nonzero",
            )
            registry[name]["split_metadata"] = policy
            splits.update(policy)
            with (
                patch("tabbiomed.data.catalog.catalog", return_value=registry),
                patch.object(storage, "benchmark_datasets", return_value=registry),
                patch.object(storage, "PARQUET", root / "parquet"),
                patch.object(storage, "DATA_CACHE", root / "cache"),
            ):
                target = ensure_dataset(name, root / "processed")
                self.assertEqual(storage.load_splits(name, root / "processed"), splits)
                self.assertEqual(verify_dataset(name, root / "processed"), metadata)
                for key in policy:
                    for value in (None, "changed"):
                        with self.subTest(key=key, value=value):
                            changed = dict(splits)
                            if value is None:
                                del changed[key]
                            else:
                                changed[key] = value
                            np.save(target / "split_indices.npy", changed)
                            with self.assertRaisesRegex(ValueError, "split policy"):
                                storage.load_splits(name, root / "processed")
                            with self.assertRaisesRegex(ValueError, "split policy"):
                                storage.data_identity(name, root / "processed")

    def test_cached_splits_require_the_recorded_stratification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, _, _, splits, _, registry = self.fixture(root, cv=True)
            policy = dict(validation_stratification="zero_vs_nonzero")
            registry[name]["split_metadata"] = policy
            splits.update(policy)
            with (
                patch.object(storage, "benchmark_datasets", return_value=registry),
                patch.object(storage, "PARQUET", root / "parquet"),
                patch.object(storage, "DATA_CACHE", root / "cache"),
            ):
                self.assertEqual(storage.load_splits(name), splits)
                directory = storage.dataset_directory(name, arrays=True)
                changed = dict(splits)
                del changed["validation_stratification"]
                np.save(directory / "split_indices.npy", changed)
                with self.assertRaisesRegex(ValueError, "split policy"):
                    storage.load_splits(name)
                with self.assertRaisesRegex(ValueError, "split policy"):
                    storage.dataset_directory(name, arrays=True)

    def test_unexpected_split_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name, _, _, splits, _, registry = self.fixture(root)
            registry["stratified"] = dict(
                split_metadata=dict(validation_stratification="zero_vs_nonzero")
            )
            source = root / "parquet"
            for value in ("zero_vs_nonzero", "unknown"):
                with (
                    self.subTest(value=value),
                    patch.object(storage, "benchmark_datasets", return_value=registry),
                ):
                    with self.assertRaisesRegex(ValueError, "split policy"):
                        storage.validate_splits(
                            name, dict(splits, validation_stratification=value), source
                        )

    def test_catalog_is_available_in_a_fresh_interpreter(self):
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from tabbiomed.tabbiomed_datasets import all_datasets\nfrom tabbiomed.data import catalog\nassert len(all_datasets)==len(catalog())==94\nprint(len(all_datasets))",
            ],
            text=True,
        )
        self.assertEqual(output.strip(), "94")
        self.assertEqual(len(read_catalog()), 94)

    def test_unknown_dataset_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "outside the benchmark"):
                ensure_dataset("unknown", temporary)


class SourceCuration(unittest.TestCase):
    def test_fixed_layout_help_does_not_import_or_execute_curators(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from tabbiomed.data.curation import FIXED_LAYOUT_SOURCES, curate

        for source in FIXED_LAYOUT_SOURCES:
            with (
                self.subTest(source=source),
                patch("runpy.run_module") as execute,
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(SystemExit) as stopped:
                    curate(source, ["--help"])
                self.assertEqual(stopped.exception.code, 0)
                execute.assert_not_called()

    def test_cdhit_preserves_input_order_and_clustering_flags(self):
        from tabbiomed.data.curation import cognano_datasets as source

        def cluster(command, *, check):
            self.assertTrue(check)
            self.assertEqual(command[0], "cd-hit")
            flags = dict(zip(command[1::2], command[2::2]))
            self.assertEqual(
                {
                    key: flags[key]
                    for key in ["-c", "-aS", "-G", "-d", "-sc", "-T", "-M"]
                },
                {
                    "-c": "0.8",
                    "-aS": "0.8",
                    "-G": "1",
                    "-d": "0",
                    "-sc": "1",
                    "-T": "8",
                    "-M": "16000",
                },
            )
            self.assertEqual(
                Path(flags["-i"]).read_text(), ">0\nAAAA\n>1\nAAAT\n>2\nCCCC\n"
            )
            Path(flags["-o"] + ".clstr").write_text(
                ">Cluster 0\n0 4aa, >0... *\n1 4aa, >1... at 90%\n>Cluster 1\n0 4aa, >2... *\n"
            )

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(source, "CDHIT_WORK_DIR", temporary),
            patch("subprocess.run", side_effect=cluster),
        ):
            np.testing.assert_array_equal(
                source._cdhit_groups(["AAAA", "AAAT", "CCCC"], 0.8, "vhh"), [0, 0, 1]
            )


if __name__ == "__main__":
    unittest.main()
