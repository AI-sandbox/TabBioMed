import csv
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tabbiomed.data.curation.paths import CONFIG_DIR, GENOMIC_ANNOTATIONS_DIR
from tabbiomed.data.curation.sexcheck_datasets import source_individuals
from tabbiomed.data.curation.plink_subsets import prepare_subsets


class CurationAssets(unittest.TestCase):
    def test_genomic_targets_load_without_workspace_annotations(self):
        root = Path(__file__).resolve().parents[1]
        script = "from tabbiomed.data.curation.genomics import load_labels; import json; print(json.dumps({k: len(v) for k, v in load_labels().items()}))"
        with tempfile.TemporaryDirectory() as workspace:
            counts = json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    cwd=workspace,
                    env={**os.environ, "PYTHONPATH": str(root)},
                    text=True,
                )
            )
        self.assertEqual(
            counts,
            dict(
                pca=2609,
                continental=3558,
                subpopulation=3558,
                coordinates=3558,
                sex=4519,
            ),
        )

    def test_bundled_sex_annotations_match_released_individuals(self):
        with (CONFIG_DIR / "sexcheck_chrX_500_sample_metadata.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        ids = [row["individual"] for row in rows]
        with tempfile.TemporaryDirectory() as workspace:
            raw = Path(workspace)
            (raw / "chrX_500.psam").write_text("#IID\n" + "\n".join(ids) + "\n")
            samples, keep, individuals, targets = source_individuals(
                raw, "sexcheck_chrX_500"
            )
        self.assertEqual(samples, ids)
        self.assertEqual(individuals, ids)
        np.testing.assert_array_equal(keep, np.arange(4026))
        np.testing.assert_array_equal(targets - 1, [int(row["target"]) for row in rows])

    def test_plink_uses_bundled_sex_table(self):
        class FirstCommandChecked(Exception):
            pass

        def check(command, *, cwd, check):
            table = Path(command[command.index("--update-sex") + 1])
            self.assertEqual(table, GENOMIC_ANNOTATIONS_DIR / "sexcheck_labels.txt")
            self.assertTrue(table.is_file())
            self.assertFalse((cwd / "labels").exists())
            raise FirstCommandChecked

        with (
            tempfile.TemporaryDirectory() as workspace,
            patch("subprocess.run", side_effect=check),
        ):
            with self.assertRaises(FirstCommandChecked):
                prepare_subsets(workspace)


if __name__ == "__main__":
    unittest.main()
