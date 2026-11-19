import numpy as np
import lightgbm as lgb
import re
from loguru import logger
from sklearn.utils.class_weight import compute_sample_weight

from tabbiomed.models.base_model import BaseModel, ModelArgs


class LightGBM(BaseModel):
    @classmethod
    def get_supported_parameters(cls, args):
        aliases = lgb.basic._ConfigAliases._get_all_param_aliases()
        return (
            super().get_supported_parameters(args)
            | set(aliases)
            | {name for names in aliases.values() for name in names}
        )

    @classmethod
    def get_base_params(cls, args):
        return {
            "class_weight": None,
            "n_estimators": args.epochs,
            "seed": args.seed,
        }

    @classmethod
    def configure_parameters(cls, parameters, args, supplied):
        names = [
            key for key in ("seed", "random_seed", "random_state") if key in supplied
        ]
        if len(names) > 1:
            raise ValueError("Specify only one lightgbm seed parameter")
        if names and names[0] != "seed":
            parameters["seed"] = parameters.pop(names[0])
        return parameters

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        self.params = dict(self.params)
        self.class_weight = self.params["class_weight"]
        if (
            args.objective not in ("binary", "classification")
            or isinstance(self.class_weight, str)
            and self.class_weight.lower() == "none"
        ):
            self.class_weight = None

        if not any(
            key in self.params
            for key in ("num_threads", "n_jobs", "nthread", "num_thread")
        ):
            self.params["num_threads"] = args.num_cpus
        self.params["verbosity"] = -1

        if args.use_gpu:
            self.params["device_type"] = "gpu"

        if args.objective in ["regression", "multiregression"]:
            self.params["objective"] = "regression"
            self.params["metric"] = "mse"
        elif args.objective == "classification":
            self.params["objective"] = "multiclass"
            self.params["num_class"] = args.num_classes
            self.params["metric"] = "multiclass"
        elif args.objective == "binary":
            self.params["objective"] = "binary"
            self.params["metric"] = "auc"

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return False

    @classmethod
    def needs_scaling(cls) -> bool:
        return False

    @classmethod
    def uses_validation_during_fit(cls) -> bool:
        return True

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        """LightGBM does not natively support multi-output regression"""
        return False

    @classmethod
    def get_default_params(cls):
        """Default learning rate and L1/L2 regularization follow TabZilla."""
        return {
            "n_estimators": 10000,
            "num_leaves": 31,
            "learning_rate": 0.08,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 1,
            "min_data_in_leaf": 20,
            "extra_trees": False,
            "cat_l2": 1.0,
            "cat_smooth": 10.0,
            "lambda_l1": 1e-3,
            "lambda_l2": 1e-3,
            "class_weight": None,
        }

    def _encode_categoricals(self, X, *, fit=False):
        if fit:
            self.cat_maps = {}
        if not self.args.cat_idx:
            return np.asarray(X)
        X = np.asarray(X, dtype=object).copy()
        for col in self.args.cat_idx or []:
            values = X[:, col].astype(str)
            if fit:
                self.cat_maps[col] = {
                    value: idx for idx, value in enumerate(np.unique(values))
                }
            X[:, col] = [self.cat_maps[col].get(value, -1) for value in values]
        return X.astype(float)

    def fit(self, X, y, X_val=None, y_val=None):
        X = self._encode_categoricals(X, fit=True)
        X_val = self._encode_categoricals(X_val)
        weights = (
            compute_sample_weight(self.class_weight, y)
            if self.class_weight is not None
            else None
        )
        train = lgb.Dataset(
            X, label=y, weight=weights, categorical_feature=self.args.cat_idx
        )

        val = lgb.Dataset(X_val, label=y_val, categorical_feature=self.args.cat_idx)

        num_boost_round = self.params["n_estimators"]

        lgb_params = {
            k: v
            for k, v in self.params.items()
            if k not in ("n_estimators", "class_weight")
        }

        if lgb_params.get("device_type") == "gpu":
            train.free_raw_data = False
            train.params = dict(lgb_params)
            train.construct()
            largest_bin = max(
                train.feature_num_bin(i) for i in range(train.num_feature())
            )
            if largest_bin > 256:
                logger.warning(
                    "LightGBM: {} training bins exceed the OpenCL GPU limit (256), using CPU.",
                    largest_bin,
                )
                lgb_params["device_type"] = "cpu"

        def fit_model():
            return lgb.train(
                lgb_params,
                train,
                num_boost_round=num_boost_round,
                valid_sets=[val],
                valid_names=["eval"],
                callbacks=[
                    lgb.early_stopping(self.args.early_stopping_rounds),
                    lgb.log_evaluation(self.args.logging_period),
                ],
            )

        try:
            self.model = fit_model()
        except lgb.basic.LightGBMError as exc:
            if lgb_params.get("device_type") != "gpu" or not re.search(
                r"bin size \d+ cannot run on GPU", str(exc)
            ):
                raise
            logger.warning(
                "LightGBM OpenCL bin incompatibility ({}), retrying on CPU.", exc
            )
            lgb_params["device_type"] = "cpu"
            self.model = fit_model()
        self.training_device = lgb_params.get("device_type", "cpu")

        return [], []

    def predict(self, X):
        if getattr(self, "_output_models_", None):
            return super().predict(X)
        if self.args.objective in ("regression", "multiregression"):
            self.predictions = self.model.predict(self._encode_categoricals(X))
            return self.predictions
        return super().predict(X)

    def predict_proba(self, X):
        probabilities = self.model.predict(self._encode_categoricals(X))

        if self.args.objective == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), 1)

        self.prediction_probabilities = probabilities
        return self.prediction_probabilities
