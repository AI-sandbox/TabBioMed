from sklearn.linear_model import LinearRegression, LogisticRegression

from tabbiomed.models.base_model import BaseModel, ModelArgs


class LinearModel(BaseModel):
    """
    Linear Model - Ordinary least squares Linear Regression / Logistic Regression
    """

    @classmethod
    def get_base_params(cls, args):
        return {
            "max_iter": 100,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        if args.objective in ["regression", "multiregression"]:
            self.model = LinearRegression(n_jobs=args.num_cpus)
        elif args.objective == "classification":
            self.model = LogisticRegression(
                n_jobs=args.num_cpus,
                max_iter=params["max_iter"],
                solver="lbfgs",
                penalty=None,
            )
        elif args.objective == "binary":
            self.model = LogisticRegression(
                n_jobs=args.num_cpus,
                max_iter=params["max_iter"],
                solver="lbfgs",
                penalty=None,
            )

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
    def get_default_params(cls):
        return {"scaler": "standard", "max_iter": 100}

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        """LinearRegression supports multi-output regression natively, LogisticRegression does not."""
        return True
