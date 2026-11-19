from sklearn.svm import SVR, SVC

from tabbiomed.models.base_model import BaseModel, ModelArgs


class SVM(BaseModel):
    required_parameters = ("C",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "max_iter": -1,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        if args.objective in ("regression", "multiregression"):
            self.model = SVR(C=params["C"], max_iter=params["max_iter"])
        elif args.objective in ("classification", "binary"):
            self.model = SVC(
                C=params["C"], probability=True, max_iter=params["max_iter"]
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
    def supports_native_multioutput(cls) -> bool:
        """In sklearn.svm.SVR documentation, clearly the y arguments in the method fit() needs to be of shape (n_samples,)."""
        return False

    @classmethod
    def get_default_params(cls):
        # Deliberately use fewer iterations than Tuned to help Default finish on the largest datasets.
        return {"C": 1.0, "max_iter": 100, "scaler": "standard"}
