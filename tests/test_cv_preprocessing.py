import unittest

import numpy as np

from tabbiomed.cv_splits import make_inner_cv_plan
from tabbiomed.tabbiomed_datasets import TabularDataset
from tabbiomed.trainer.inner_cv import prepare_inner_cv


class EncodingModel:
    @classmethod
    def needs_missing_value_imputation(cls):
        return False

    @classmethod
    def needs_cat_encoding(cls):
        return True


class CVPreprocessing(unittest.TestCase):
    def dataset(self, X, y, splits):
        return TabularDataset(
            name="example",
            X=X,
            y=y,
            cat_idx=[0],
            task="binary",
            num_classes=2,
            num_features=X.shape[1],
            num_samples=len(y),
            split_indices=splits,
        )

    def prepare(self, dataset, strategy):
        dataset.prepare_dataset(
            inpute_mv=False,
            encode_cat=True,
            scale_features=False,
            cat_encoding_strategy=strategy,
        )

    def test_outer_fold_encoding_preserves_raw_data_and_fold_order(self):
        raw = np.array([["a"], ["b"], ["a"], ["b"], ["c"], ["c"]], dtype=object)
        y = np.array([0, 1, 0, 1, 0, 1])
        folds = [
            dict(train=[0, 1, 2, 3], val=[4], test=[5]),
            dict(train=[0, 1, 4, 5], val=[2], test=[3]),
        ]
        for strategy in ("onehot", "label"):
            with self.subTest(strategy=strategy):
                results = []
                for order in ((0, 1), (1, 0)):
                    parent = self.dataset(
                        raw.copy(), y.copy(), {"cv_folds": [folds[i] for i in order]}
                    )
                    prepared = {}
                    for original_id, (fold, _, _) in zip(
                        order, parent.get_cv_splits()
                    ):
                        self.prepare(fold, strategy)
                        np.testing.assert_array_equal(parent.X, raw)
                        prepared[original_id] = [
                            getattr(fold, "X_" + part).copy()
                            for part in ("train", "val", "test")
                        ]
                    results.append(prepared)
                for fold_id in range(2):
                    for first, second in zip(results[0][fold_id], results[1][fold_id]):
                        np.testing.assert_array_equal(first, second)

    def test_inner_folds_match_independent_preprocessing(self):
        y = np.tile([0, 1], 6)
        split = dict(train=list(range(8)), val=[8, 9], test=[10, 11])
        plan = make_inner_cv_plan(
            y, "binary", split, requested_folds=3, seed=42, outer_fold_id=0
        )
        raw = np.tile([["a"], ["b"]], (6, 1)).astype(object)
        raw[plan["folds"][0]["val"][0], 0] = "c"
        for strategy in ("onehot", "label"):
            with self.subTest(strategy=strategy):
                parent = self.dataset(raw.copy(), y.copy(), {"cv_folds": [split]})
                outer = next(parent.get_cv_splits())[0]
                actual_plan = prepare_inner_cv(
                    parent,
                    outer,
                    EncodingModel,
                    requested_folds=3,
                    seed=42,
                    outer_fold_id=0,
                    mv_strategy="knn",
                    cat_encoding_strategy=strategy,
                )
                self.assertEqual(actual_plan, plan)
                for inner, indices in zip(outer.hpo_inner_datasets, plan["folds"]):
                    reference = self.dataset(
                        raw.copy(),
                        y.copy(),
                        dict(
                            train=indices["train"],
                            val=indices["val"],
                            test=split["test"],
                        ),
                    )
                    self.prepare(reference, strategy)
                    for part in ("train", "val"):
                        np.testing.assert_array_equal(
                            getattr(inner, "X_" + part), getattr(reference, "X_" + part)
                        )
                    self.assertEqual(inner.cat_dims, reference.cat_dims)
                    self.assertEqual(inner.cat_idx, reference.cat_idx)
                np.testing.assert_array_equal(parent.X, raw)


if __name__ == "__main__":
    unittest.main()
