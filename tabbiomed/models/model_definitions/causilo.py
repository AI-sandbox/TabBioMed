"""Causilo 1.5 inference on the full labeled context of each benchmark split."""

from ..base_model import BaseModel


class Causilo(BaseModel):
    n_estimators = 8

    @classmethod
    def get_default_params(cls):
        return {
            "model_version": "1.5",
            "n_estimators": cls.n_estimators,
            "max_context": None,
            "use_kv_cache": False,
        }

    @classmethod
    def needs_missing_value_imputation(cls):
        return False

    @classmethod
    def needs_cat_encoding(cls):
        return False

    @classmethod
    def needs_scaling(cls):
        return False

    @classmethod
    def supports_native_multioutput(cls):
        return False

    def __init__(self, params, args):
        super().__init__(params, args)
        from causilo import CausiloClassifier, CausiloRegressor

        estimator = (
            CausiloClassifier
            if args.objective in ("classification", "binary")
            else CausiloRegressor
        )
        device = "cpu"
        if args.use_gpu:
            device = f"cuda:{args.gpu_ids[0]}" if args.gpu_ids else "cuda"
        self.model = estimator(
            **self.params,
            device=device,
            random_state=args.seed,
        )


class CausiloLight(Causilo):
    n_estimators = 1
