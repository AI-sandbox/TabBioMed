from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from tabbiomed.models.base_model import BaseModel, ModelArgs


class RandomForest(BaseModel):
    """
    Random Forest - Random Forest Regressor/Classifier

    Takes the maximum depth of the trees and the number of estimators as hyperparameter
    """

    required_parameters = ("n_estimators", "max_depth")

    optional_parameters = ("min_samples_leaf",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "bootstrap": True,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        rf_params = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "random_state": args.seed,
            "n_jobs": args.num_cpus,
        }

        if "max_features" in params:
            rf_params["max_features"] = params["max_features"]
        if "min_samples_split" in params:
            rf_params["min_samples_split"] = params["min_samples_split"]
        if "min_samples_leaf" in params:
            rf_params["min_samples_leaf"] = params["min_samples_leaf"]

        bootstrap_value = params["bootstrap"]
        rf_params["bootstrap"] = bootstrap_value

        if "max_samples" in params and params["max_samples"] is not None:
            if bootstrap_value is False:
                rf_params["max_samples"] = None
            else:
                rf_params["max_samples"] = params["max_samples"]
        if "min_impurity_decrease" in params:
            rf_params["min_impurity_decrease"] = params["min_impurity_decrease"]

        if args.objective in ["classification", "binary"]:
            if "class_weight" in params:
                class_weight = params["class_weight"]
                if class_weight is None or (
                    isinstance(class_weight, str) and class_weight.lower() == "none"
                ):
                    pass
                elif isinstance(class_weight, str) and class_weight.lower() in [
                    "balanced",
                    "balanced_subsample",
                ]:
                    rf_params["class_weight"] = class_weight.lower()

        if args.objective in ["regression", "multiregression"]:
            self.model = RandomForestRegressor(**rf_params)
        elif args.objective in ["classification", "binary"]:
            self.model = RandomForestClassifier(**rf_params)

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        """RandomForest handles numerical missing values natively."""
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        return False

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        """RandomForest supports multi-output prediction natively."""
        return True

    @classmethod
    def get_default_params(cls):
        return {
            "n_estimators": 100,
            "max_depth": None,
            "bootstrap": True,
            "max_features": "sqrt",
            "max_samples": None,
            "min_samples_split": 2,
            "min_impurity_decrease": 0.0,
            "class_weight": None,
        }
