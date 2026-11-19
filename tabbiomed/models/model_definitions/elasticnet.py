from sklearn.linear_model import ElasticNet as skElasticNet
from sklearn.linear_model import LogisticRegression as skLogisticRegression

from tabbiomed.models.base_model import BaseModel, ModelArgs


class ElasticNet(BaseModel):
    required_parameters = ("alpha", "l1_ratio")

    @classmethod
    def get_base_params(cls, args):
        return {
            "max_iter": 10000,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        random_state = getattr(args, "random_seed", getattr(args, "seed", None))
        if args.objective in ["regression", "multiregression"]:
            self.model = skElasticNet(
                alpha=params["alpha"],
                l1_ratio=params["l1_ratio"],
                max_iter=params["max_iter"],
                random_state=random_state,
            )
        elif args.objective == "classification":
            self.model = skLogisticRegression(
                penalty="elasticnet",
                C=1 / params["alpha"],
                l1_ratio=params["l1_ratio"],
                solver="saga",
                max_iter=params["max_iter"],
                n_jobs=args.num_cpus,
                random_state=random_state,
            )
        elif args.objective == "binary":
            self.model = skLogisticRegression(
                penalty="elasticnet",
                C=1 / params["alpha"],
                l1_ratio=params["l1_ratio"],
                solver="saga",
                max_iter=params["max_iter"],
                n_jobs=args.num_cpus,
                random_state=random_state,
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
        # Deliberately use fewer iterations than Tuned to help Default finish on the largest datasets.
        return {"alpha": 1.0, "max_iter": 100, "l1_ratio": 0.5, "scaler": "standard"}

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return True
