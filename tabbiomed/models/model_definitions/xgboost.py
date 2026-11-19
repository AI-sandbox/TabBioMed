import numpy as np
import pandas as pd
import xgboost as xgb

from tabbiomed.models.base_model import BaseModel, ModelArgs


class XGBoost(BaseModel):
    """
    XGBoost (https://xgboost.readthedocs.io/en/stable/)
    """

    @classmethod
    def get_supported_parameters(cls, args):
        return (
            super().get_supported_parameters(args)
            | set(xgb.XGBModel().get_xgb_params())
            | {"enable_categorical", "eta", "nthread", "seed", "num_class"}
        )

    @classmethod
    def get_base_params(cls, args):
        return {
            "n_estimators": args.epochs,
            "seed": args.seed,
        }

    @classmethod
    def configure_parameters(cls, parameters, args, supplied):
        names = [key for key in ("seed", "random_state") if key in supplied]
        if len(names) > 1:
            raise ValueError("Specify only one xgboost seed parameter")
        if names and names[0] != "seed":
            parameters["seed"] = parameters.pop(names[0])
        return parameters

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)
        params = self.params

        xgb_params = {}
        param_mapping = {
            "learning_rate": "eta",
            "alpha": "reg_alpha",
            "lambda": "reg_lambda",
        }

        for key, value in params.items():
            xgb_key = param_mapping.get(key, key)
            xgb_params[xgb_key] = value

        self.params = xgb_params
        if "nthread" not in self.params and "n_jobs" not in self.params:
            self.params["nthread"] = args.num_cpus
        self.params["verbosity"] = 1

        self.params["tree_method"] = "hist"
        if args.use_gpu:
            self.params["device"] = (
                "cuda:" + str(args.gpu_ids[0]) if args.gpu_ids else "cuda:0"
            )

        if args.objective in ["regression", "multiregression"]:
            self.params["objective"] = "reg:squarederror"
            self.params["eval_metric"] = "rmse"
        elif args.objective == "classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = args.num_classes
            self.params["eval_metric"] = "mlogloss"
        elif args.objective == "binary":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "auc"

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
        return True

    @classmethod
    def get_default_params(cls):
        return {
            "n_estimators": 10000,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
            "colsample_bynode": 1.0,
            "alpha": 0.0,
            "lambda": 1.0,
            "grow_policy": "depthwise",
            "scale_pos_weight": 1.0,
        }

    def _to_dmatrix(self, X, y=None, *, fit=False):
        cat_indices = set(self.args.cat_idx or [])
        X = np.asarray(X, dtype=object).copy() if cat_indices else np.asarray(X)
        if fit:
            self.cat_maps = {}
        for col in cat_indices:
            missing = pd.isna(X[:, col])
            values = X[:, col].astype(str)
            if fit:
                self.cat_maps[col] = {
                    value: idx for idx, value in enumerate(np.unique(values[~missing]))
                }
            mapping = self.cat_maps[col]

            X[:, col] = [
                np.nan if is_missing else mapping.get(value, np.nan)
                for value, is_missing in zip(values, missing)
            ]
        feature_types = [
            "c" if col in cat_indices else "q" for col in range(X.shape[1])
        ]
        return xgb.DMatrix(
            X.astype(float) if X.dtype == object else X,
            label=y,
            feature_types=feature_types,
            enable_categorical=True,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        train = self._to_dmatrix(X, y, fit=True)
        val = self._to_dmatrix(X_val, y_val)
        eval_list = [(val, "eval")]

        num_boost_round = self.params["n_estimators"]

        xgb_params = {
            k: v
            for k, v in self.params.items()
            if k not in ("n_estimators", "enable_categorical")
        }

        self.model = xgb.train(
            xgb_params,
            train,
            num_boost_round=num_boost_round,
            evals=eval_list,
            early_stopping_rounds=self.args.early_stopping_rounds,
            verbose_eval=self.args.logging_period,
        )

        return [], []

    def predict(self, X):
        """
        Override predict to handle DMatrix conversion for regression.
        For classification, delegate to base class which calls predict_proba.
        For multi-output regression (fitted via _fit_multioutput), delegate to base class
        which combines predictions from each per-output model stored in _output_models_.
        """

        if hasattr(self, "_output_models_") and self._output_models_:
            return super().predict(X)

        if self.args.objective in ["regression", "multiregression"]:
            X_dmatrix = self._to_dmatrix(X)
            iteration_range = (
                (0, self.model.best_iteration + 1)
                if getattr(self.model, "best_iteration", None) is not None
                else None
            )
            self.predictions = self.model.predict(
                X_dmatrix, iteration_range=iteration_range
            )
            return self.predictions
        else:
            return super().predict(X)

    def predict_proba(self, X):
        X_dmatrix = self._to_dmatrix(X)
        iteration_range = (
            (0, self.model.best_iteration + 1)
            if getattr(self.model, "best_iteration", None) is not None
            else None
        )
        probabilities = self.model.predict(X_dmatrix, iteration_range=iteration_range)

        if self.args.objective == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), 1)

        self.prediction_probabilities = probabilities
        return self.prediction_probabilities
