from sklearn.linear_model import Ridge, LogisticRegression

from tabbiomed.models.base_model import BaseModel, ModelArgs


class RidgeModel(BaseModel):
    required_parameters = ("alpha",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "max_iter": 10000,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        def convert_bool(value):
            if isinstance(value, str):
                return value.lower() == "true"
            return bool(value)

        def convert_class_weight(value):
            if value is None or (isinstance(value, str) and value.lower() == "null"):
                return None
            elif isinstance(value, str) and value.lower() == "balanced":
                return "balanced"
            return value

        if args.objective in ["regression", "multiregression"]:
            ridge_params = {
                "alpha": params["alpha"],
                "max_iter": params["max_iter"],
            }
            if "fit_intercept" in params:
                ridge_params["fit_intercept"] = convert_bool(params["fit_intercept"])
            self.model = Ridge(**ridge_params)

        elif args.objective in ["classification", "binary"]:
            logreg_params = {
                "penalty": "l2",
                "C": 1 / params["alpha"],
                "max_iter": params["max_iter"],
                "n_jobs": 1,
            }
            if "fit_intercept" in params:
                logreg_params["fit_intercept"] = convert_bool(params["fit_intercept"])
            if "class_weight" in params:
                logreg_params["class_weight"] = convert_class_weight(
                    params["class_weight"]
                )
            self.model = LogisticRegression(**logreg_params)

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return True

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        return True

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        # Deliberately use fewer iterations than Tuned to help Default finish on the largest datasets.
        return {
            "alpha": 1.0,
            "max_iter": 100,
            "fit_intercept": True,
            "class_weight": None,
            "scaler": "standard",
        }
