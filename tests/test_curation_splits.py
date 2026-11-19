from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np
import pandas as pd

from tabbiomed.data.curation.split_utils import create_group_stratified_split


class CurationSplitRatios(unittest.TestCase):
    def test_grouped_splits_reject_missing_identifiers(self):
        for missing in (None, np.nan, pd.NA):
            for stratified in (False, True):
                with self.subTest(missing=missing, stratified=stratified):
                    groups = np.repeat(np.arange(50), 2).astype(object).tolist()
                    frame = pd.DataFrame(
                        dict(
                            group=groups + [missing],
                            target=(np.arange(101) // 2) % 2,
                        )
                    )
                    config = dict(
                        group_by="group", stratify_by="target" if stratified else None
                    )
                    with self.assertRaisesRegex(ValueError, "missing identifiers"):
                        create_group_stratified_split(frame, config)

    def test_grouped_splits_reject_missing_group_column(self):
        frame = pd.DataFrame(dict(target=np.arange(100) % 2))
        for stratify in (None, "target"):
            with self.subTest(stratify=stratify):
                with self.assertRaisesRegex(ValueError, "Group column.*not found"):
                    create_group_stratified_split(
                        frame, dict(group_by="group", stratify_by=stratify)
                    )

    def test_unequal_fractions_in_every_split_mode(self):
        frame = pd.DataFrame(
            dict(
                group=np.repeat(np.arange(1000), 2),
                target=np.repeat(np.arange(1000) % 2, 2),
            ),
            index=np.arange(2000) + 5000,
        )
        for grouped in (False, True):
            for stratified in (False, True):
                for val_size, test_size in ((0.1, 0.2), (0.2, 0.1)):
                    with self.subTest(
                        grouped=grouped,
                        stratified=stratified,
                        val=val_size,
                        test=test_size,
                    ):
                        config = dict(
                            group_by="group" if grouped else None,
                            stratify_by="target" if stratified else None,
                            train_size=0.7,
                            val_size=val_size,
                            test_size=test_size,
                            random_state=42,
                        )
                        with redirect_stdout(StringIO()):
                            splits = create_group_stratified_split(frame, config)
                        rows = np.concatenate(list(splits.values()))
                        np.testing.assert_array_equal(
                            np.sort(rows), np.arange(len(frame))
                        )
                        for part, fraction in (
                            ("train", 0.7),
                            ("val", val_size),
                            ("test", test_size),
                        ):
                            self.assertAlmostEqual(
                                len(splits[part]) / len(frame), fraction, delta=0.002
                            )
                            if stratified:
                                self.assertAlmostEqual(
                                    frame.iloc[splits[part]]["target"].mean(),
                                    0.5,
                                    delta=0.01,
                                )
                        if grouped:
                            groups = {
                                part: set(frame.iloc[indices]["group"])
                                for part, indices in splits.items()
                            }
                            self.assertFalse(groups["train"] & groups["val"])
                            self.assertFalse(groups["train"] & groups["test"])
                            self.assertFalse(groups["val"] & groups["test"])


if __name__ == "__main__":
    unittest.main()
