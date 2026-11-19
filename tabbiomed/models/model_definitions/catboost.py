from inspect import signature

from catboost import CatBoostRegressor, CatBoostClassifier
from tabbiomed.models.base_model import BaseModel, ModelArgs


class CatBoost(BaseModel):
    """
    CatBoost (https://catboost.ai/)
    """

    @classmethod
    def get_supported_parameters(cls, args):
        estimator = (
            CatBoostClassifier
            if args.objective in ("binary", "classification")
            else CatBoostRegressor
        )
        return super().get_supported_parameters(args) | {
            name
            for name, parameter in signature(estimator).parameters.items()
            if parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        }

    @classmethod
    def get_base_params(cls, args):
        return {
            "thread_count": args.num_cpus,
            "iterations": args.epochs,
            "random_seed": args.seed,
        }

    @classmethod
    def configure_parameters(cls, parameters, args, supplied):
        names = [key for key in ("random_seed", "random_state") if key in supplied]
        if len(names) > 1:
            raise ValueError("Specify only one catboost seed parameter")
        if names and names[0] != "random_seed":
            parameters["random_seed"] = parameters.pop(names[0])
        return parameters

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)
        params = self.params

        catboost_params = {}
        for key, value in params.items():
            if key == "depth":
                catboost_params["max_depth"] = value
            elif key == "auto_class_weights":
                if value is None or (
                    isinstance(value, str) and value.lower() == "none"
                ):
                    continue
                else:
                    catboost_params["auto_class_weights"] = value
            else:
                catboost_params[key] = value

        if args.use_gpu:
            catboost_params["colsample_bylevel"] = 1.0

        self.params = catboost_params

        self.params["od_type"] = "Iter"
        self.params["od_wait"] = self.args.early_stopping_rounds
        self.params["verbose"] = self.args.logging_period
        self.params["train_dir"] = (
            "output/CatBoost/" + self.args.dataset + "/catboost_info"
        )

        self.params["allow_writing_files"] = False

        if args.use_gpu:
            self.params["task_type"] = "GPU"
            gpu_ids = self.args.gpu_ids
            if isinstance(gpu_ids, (list, tuple)):
                self.params["devices"] = (
                    ":".join(str(g) for g in gpu_ids) if gpu_ids else "0"
                )
            elif gpu_ids is not None:
                self.params["devices"] = str(gpu_ids)

        self.params["cat_features"] = self.args.cat_idx

        catboost_model_params = {
            k: v for k, v in self.params.items() if k != "threshold"
        }

        if args.objective in ("regression", "multiregression"):
            regression_only_params = {
                k: v
                for k, v in catboost_model_params.items()
                if k != "auto_class_weights"
            }
            if args.objective == "multiregression":
                regression_only_params["loss_function"] = "MultiRMSE"
            self.model = CatBoostRegressor(**regression_only_params)
        elif args.objective == "classification" or args.objective == "binary":
            self.model = CatBoostClassifier(**catboost_model_params)

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
        """CatBoost supports multi-output regression natively via the MultiRMSE objective."""
        return True

    @classmethod
    def get_default_params(cls):
        return {
            "iterations": 10000,
            "learning_rate": 0.03,
            "depth": 6,
            "grow_policy": "SymmetricTree",
            "l2_leaf_reg": 3.0,
            "model_size_reg": 0.5,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "colsample_bylevel": 1.0,
            "one_hot_max_size": 15,
            "max_ctr_complexity": 4,
            "leaf_estimation_iterations": 1,
            "boosting_type": "Plain",
            "max_bin": 254,
            "auto_class_weights": None,
        }

    def fit(self, X, y, X_val=None, y_val=None):
        if self.args.cat_idx:
            X = X.astype(object, copy=True)
            X_val = X_val.astype(object, copy=True)

            for c in self.args.cat_idx:
                X[:, c] = X[:, c].astype(str)
                X_val[:, c] = X_val[:, c].astype(str)

        self.model.fit(X, y, eval_set=(X_val, y_val))
        return [], []

    def predict(self, X):
        if self.args.cat_idx:
            X = X.astype(object, copy=True)
            for c in self.args.cat_idx:
                X[:, c] = X[:, c].astype(str)

        return super().predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for CatBoost.
        Converts categorical features to strings before prediction.
        """
        if self.args.cat_idx:
            X = X.astype(object, copy=True)
            for c in self.args.cat_idx:
                X[:, c] = X[:, c].astype(str)

        return super().predict_proba(X)
