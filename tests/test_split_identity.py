from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tabbiomed.results.consolidation import Evaluation, check_scopes
from tabbiomed.results.io import digest
from tabbiomed.results.sources import Evidence, dataset_splits


class SplitIdentity(unittest.TestCase):
    def splits(self, cv=False):
        if cv:
            folds = []
            for index in range(5):
                test = [2 * index, 2 * index + 1]
                pool = [row for row in range(10) if row not in test]
                folds.append(dict(train=pool[:6], val=pool[6:], test=test))
            return dict(cv_folds=folds, groups=[f"g{i}" for i in range(10)])
        return dict(
            train=list(range(6)), val=[6, 7], test=[8, 9], groups=list(range(10))
        )

    def arrays(self, splits):
        result = {}
        for name, value in reversed(list(splits.items())):
            if name == "cv_folds":
                result[name] = [self.arrays(fold) for fold in value]
            elif name in ("train", "val", "test"):
                result[name] = np.asarray(value, dtype=np.int32)
            else:
                result[name] = value
        return result

    def evaluation(self, root, name, splits, *, pca=False):
        directory = root / "raw" / name
        directory.mkdir(parents=True)
        np.save(directory / "X.npy", np.arange(20).reshape(10, 2))
        np.save(directory / "y.npy", np.arange(10) % 2)
        np.save(directory / "split_indices.npy", splits)
        (directory / "metadata.json").write_text(json.dumps(dict(num_samples=10)))
        files = {
            filename: dict(
                size=(directory / filename).stat().st_size,
                sha256=digest(directory / filename),
            )
            for filename in ("X.npy", "y.npy", "metadata.json", "split_indices.npy")
        }
        identity = dict(model="Model", dataset="example", seed=42, data_files=files)
        evidence = Evidence()
        _, _, scientific = dataset_splits(identity, directory, evidence)
        return Evaluation(
            destination=Path("Model/example/pca_500" if pca else "Model/example"),
            report=dict(
                model="Model", dataset="example", seed=42, primary_metric="mse"
            ),
            timing=dict(dataset_split_sha256=scientific),
            artifacts={},
            evidence=evidence,
            identity=identity,
            metadata={},
        )

    def test_equivalent_split_serializations_share_nested_evaluation_identity(self):
        for cv in (False, True):
            with self.subTest(cv=cv), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                splits = self.splits(cv)
                main = self.evaluation(root, "main", splits)
                pca = self.evaluation(root, "pca", self.arrays(splits), pca=True)
                self.assertNotEqual(
                    main.identity["data_files"]["split_indices.npy"]["sha256"],
                    pca.identity["data_files"]["split_indices.npy"]["sha256"],
                )
                self.assertEqual(main.timing, pca.timing)
                check_scopes([main, pca], root / "results")

    def test_changed_split_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self.splits()
            changed = deepcopy(splits)
            changed["train"][0], changed["val"][0] = (
                changed["val"][0],
                changed["train"][0],
            )
            main = self.evaluation(root, "main", splits)
            pca = self.evaluation(root, "pca", changed, pca=True)
            self.assertNotEqual(main.timing, pca.timing)
            with self.assertRaisesRegex(ValueError, "different stored splits"):
                check_scopes([main, pca], root / "results")

    def test_existing_main_accepts_known_split_identities_and_rejects_others(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pca = self.evaluation(root, "pca", self.splits(), pca=True)
            output = root / "results"
            main = output / "Model/example"
            main.mkdir(parents=True)
            (main / "report.json").write_text(json.dumps(pca.report))
            scientific = pca.timing["dataset_split_sha256"]
            serialized = pca.identity["data_files"]["split_indices.npy"]["sha256"]
            self.assertNotEqual(scientific, serialized)
            for value in (scientific, serialized, "wrong"):
                with self.subTest(split_identity=value):
                    pca.evidence = Evidence()
                    (main / "timing.json").write_text(
                        json.dumps(dict(dataset_split_sha256=value))
                    )
                    if value == "wrong":
                        with self.assertRaisesRegex(
                            ValueError, "different stored splits"
                        ):
                            check_scopes([pca], output)
                    else:
                        check_scopes([pca], output)


if __name__ == "__main__":
    unittest.main()
