from loguru import logger
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from tabbiomed.models.base_model import BaseModel, ModelArgs


class KNN(BaseModel):
    required_parameters = ("n_neighbors", "weights", "p")

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        knn_params = {
            "n_neighbors": params["n_neighbors"],
            "n_jobs": args.num_cpus,
        }

        if "weights" in params:
            knn_params["weights"] = params["weights"]
        if "p" in params:
            knn_params["p"] = params["p"]

        if args.objective in ["regression", "multiregression"]:
            self.model = KNeighborsRegressor(**knn_params)
        elif args.objective in ["classification", "binary"]:
            self.model = KNeighborsClassifier(**knn_params)

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
        return {
            "n_neighbors": 5,
            "weights": "uniform",
            "p": 2,
            "scaler": "standard",
        }

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit all supplied training rows, validation rows are excluded."""

        logger.info("KNN: fitting all {} supplied training rows", X.shape[0])
        return super().fit(X, y, None, None)
