import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from tabbiomed.models import get_model_spec, model_name_map, str2model
from tabbiomed.models.model_definitions.kumo_tabular import _context_index, _feature_frame


class KumoTabularTests(unittest.TestCase):
    def test_defaults_and_checkpoint_sizes(self):
        self.assertEqual(model_name_map("Kumo-Tabular-Large"), "Kumo-Tabular")
        for size, count in (("Small", 8), ("Medium", 8), ("Large", 16)):
            name = f"Kumo-Tabular-{size}"
            cls = str2model(name)
            spec = get_model_spec(name)
            self.assertTrue(spec.gpu)
            self.assertTrue(spec.requires_separate_env)
            self.assertFalse(spec.hpo_enabled)
            self.assertEqual(cls.size, size.lower())
            self.assertEqual(cls.get_default_params(), {"num_estimators": count, "max_context_size": 200_000})

    def test_per_member_context(self):
        self.assertIsNone(_context_index(10, 2, 10, 42))
        index = _context_index(10, 3, 7, 42)
        self.assertEqual(index.shape, (3, 7))
        self.assertTrue(torch.equal(index, _context_index(10, 3, 7, 42)))
        for complete_pass in index.flatten()[:20].reshape(2, 10):
            self.assertTrue(torch.equal(complete_pass.sort().values, torch.arange(10)))

    def test_mixed_column_types_and_unsigned_counts(self):
        frame = _feature_frame(np.array([[1.5, "a"], [2.5, "b"]], dtype=object))
        self.assertEqual(list(frame), ["0", "1"])
        self.assertEqual(frame["0"].dtype.kind, "f")
        for value, kind in [(23, "i"), (2 ** 63, "f")]:
            self.assertEqual(_feature_frame(np.array([[value]], dtype=np.uint64))["0"].dtype.kind, kind)

    def test_native_output_conversion(self):
        cls = str2model("Kumo-Tabular")
        args = SimpleNamespace(objective="classification", num_classes=3)
        model = cls(cls.resolve_parameters(cls.get_default_params(), args), args)
        model.device = torch.device("cpu")
        model.subsampled = False
        model.stypes = {"0": "numerical"}
        model.regression = False
        model.model = Mock()
        backend = Mock()
        native = Mock()
        native.numerical = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        native.columns = {backend.Stype.numerical: ["2", "0"]}
        model.model.predict.return_value = native

        X = np.zeros((2, 1))
        with patch.dict("sys.modules", {"sdm": backend}), patch.object(model, "_query_batch_size", return_value=2):
            probs = model.predict_proba(X)
            np.testing.assert_allclose(probs, [[0.2, 0, 0.8], [0.9, 0, 0.1]])
            args.objective = "regression"
            model.regression = True
            native.numerical = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.float16)
            np.testing.assert_array_equal(model.predict(X), [2, 3])

    def test_multioutput_predictions_use_each_target_model(self):
        args = Mock(objective="multiregression")
        cls = str2model("Kumo-Tabular")
        model = cls(cls.resolve_parameters(cls.get_default_params(), args), args)
        model._output_models_ = [Mock(), Mock()]
        for member, predictions in zip(model._output_models_, ([1, 2], [3, 4])):
            member.predict.return_value = np.asarray(predictions)
        np.testing.assert_array_equal(model.predict(np.zeros((2, 1))), [[1, 3], [2, 4]])
