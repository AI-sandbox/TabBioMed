from sklearn.linear_model import Lasso, LogisticRegression

from tabbiomed.models.base_model import BaseModel, ModelArgs


class LassoModel(BaseModel):
    """
    Lasso Model - Linear Model with L1 regularization

    Takes regularization strength (alpha) as hyperparameter
    """

    required_parameters = ("alpha",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "max_iter": 10000,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        random_state = getattr(args, "random_seed", getattr(args, "seed", None))

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
            lasso_params = {
                "alpha": params["alpha"],
                "max_iter": params["max_iter"],
                "random_state": random_state,
            }
            if "fit_intercept" in params:
                lasso_params["fit_intercept"] = convert_bool(params["fit_intercept"])
            self.model = Lasso(**lasso_params)

        elif args.objective in ["classification", "binary"]:
            logreg_params = {
                "penalty": "l1",
                "C": 1 / params["alpha"],
                "solver": "liblinear" if getattr(args, "is_cv", False) else "saga",
                "max_iter": params["max_iter"],
                "n_jobs": args.num_cpus,
                "random_state": random_state,
            }
            if "fit_intercept" in params:
                logreg_params["fit_intercept"] = convert_bool(params["fit_intercept"])
            if "class_weight" in params:
                logreg_params["class_weight"] = convert_class_weight(
                    params["class_weight"]
                )
            self.model = LogisticRegression(**logreg_params)

        else:
            raise ValueError(f"Invalid objective found: {args.objective}")

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
        }
