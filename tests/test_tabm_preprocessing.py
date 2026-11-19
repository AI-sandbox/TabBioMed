import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("tabm") and importlib.util.find_spec("rtdl_num_embeddings"),
    "requires the neural model dependencies",
)
class TabMBooleanFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tabbiomed.models.model_definitions.tabm import TabM

        cls.model_class = TabM

    def make_model(self, n_features, cat_idx=(), enabled=True):
        args = SimpleNamespace(
            objective="binary", seed=42, use_gpu=False, num_features=n_features,
            cat_idx=list(cat_idx), num_classes=2, num_outputs=1, logging_period=0,
        )
        params = self.model_class.resolve_parameters(
            {
                **self.model_class.get_default_params(),
                "bool_to_cat": enabled, "amp": False, "n_epochs": 1,
                "n_blocks": 1, "d_block": 16, "tabm_k": 2,
                "d_embedding": 4, "num_emb_n_bins": 3, "batch_size": 4,
            },
            args,
        )
        return self.model_class(params, args)

    def data(self):
        return np.column_stack(
            [[0, 1] * 4, np.linspace(0.1, 1.8, 8), [0, 1, 2, 0, 1, 2, 0, 1]]
        )

    def fit(self, model, train, validation=None):
        y = np.arange(len(train)) % 2
        val = train[:2].copy() if validation is None else validation
        model.fit(train, y, val, np.arange(len(val)) % 2)
        return model

    def test_flag_changes_model_inputs(self):
        train = self.data()
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                model = self.fit(self.make_model(3, [2], enabled), train)
                inputs = model._apply_quantile_transform(train, is_train=False)
                numeric, categorical = model._prepare_data(inputs)
                self.assertEqual(model.cat_idx, [2, 0] if enabled else [2])
                self.assertEqual(model.num_idx, [1] if enabled else [0, 1])
                self.assertEqual(numeric.shape[1], 1 if enabled else 2)
                self.assertEqual(categorical.shape[1], 2 if enabled else 1)
                np.testing.assert_array_equal(categorical[:, 0], train[:, 2])
                if enabled:
                    np.testing.assert_array_equal(categorical[:, 1], train[:, 0])
                else:
                    np.testing.assert_array_equal(numeric[:, 0], train[:, 0])
                self.assertEqual(model.args.cat_idx, [2])

    def test_heldout_values_use_training_types_and_reserved_category(self):
        train = self.data()
        original = train.copy()
        heldout = np.array([[0, 0.3, 0], [1, 0.6, 1], [2, 0.9, 2],
                            [0.5, 1.1, -1], [np.nan, 1.4, 0]])
        heldout_original = heldout.copy()
        model = self.fit(self.make_model(3, [2]), train, heldout)
        numeric, categorical = model._prepare_data(heldout)
        np.testing.assert_array_equal(categorical[:, 1], [0, 1, 2, 2, 2])
        np.testing.assert_array_equal(categorical[:, 0], [0, 1, 2, 3, 0])
        self.assertEqual(model.cat_cardinalities, [4, 3])
        self.assertEqual(numeric.shape[1], 1)
        batch = model.predict_proba(heldout)
        individual = np.concatenate([model.predict_proba(row[None]) for row in heldout])
        self.assertTrue(np.isfinite(batch).all())
        np.testing.assert_allclose(batch.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(batch, individual, atol=1e-6)
        np.testing.assert_array_equal(train, original)
        np.testing.assert_array_equal(heldout, heldout_original)

    def test_heldout_binary_values_do_not_change_training_types(self):
        train = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=float)[:, None]
        model = self.fit(self.make_model(1), train, np.array([[0.0], [1.0]]))
        self.assertEqual(model.cat_idx, [])
        self.assertEqual(model.num_idx, [0])

    def test_constant_column_mask_and_existing_categories_are_preserved(self):
        train = np.column_stack([np.ones(8), self.data()])
        original = train.copy()
        model = self.fit(self.make_model(4, [3]), train)
        np.testing.assert_array_equal(model._cols_keep_orig, [1, 2, 3])
        self.assertEqual(model.cat_idx, [2, 0])
        self.assertEqual(model.num_idx, [1])
        self.assertEqual(model.args.cat_idx, [3])
        self.assertEqual(model.predict_proba(train).shape, (8, 2))
        np.testing.assert_array_equal(train, original)

    def test_refit_relearns_binary_columns(self):
        train = self.data()
        model = self.fit(self.make_model(3, [2]), train)
        self.assertEqual(model.cat_idx, [2, 0])
        train[:, 0] = np.linspace(0.1, 0.8, 8)
        self.fit(model, train)
        self.assertEqual(model.cat_idx, [2])
        self.assertEqual(model.num_idx, [0, 1])
        self.assertEqual(model._binary_cat_idx, set())
        self.assertEqual(model.predict_proba(train).shape, (8, 2))

    def test_all_binary_numeric_inputs_can_become_categorical(self):
        train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]] * 2)
        model = self.fit(self.make_model(2), train)
        self.assertEqual(model.num_idx, [])
        self.assertEqual(model.cat_idx, [0, 1])
        self.assertEqual(model.cat_cardinalities, [3, 3])
        numeric, categorical = model._prepare_data(train)
        self.assertIsNone(numeric)
        np.testing.assert_array_equal(categorical, train)
        self.assertEqual(model.predict_proba(train).shape, (8, 2))


if __name__ == "__main__":
    unittest.main()
