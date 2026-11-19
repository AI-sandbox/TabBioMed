from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from tabbiomed.models.base_model import BaseModel, ModelArgs


class DecisionTree(BaseModel):
    """
    Decision Tree - Decision Tree Regressor/Classifier

    Takes the maximum depth of the tree as hyperparameter
    """

    required_parameters = ("max_depth",)

    @classmethod
    def get_base_params(cls, args):
        return {
            "splitter": "best",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }

    def __init__(self, params, args: ModelArgs):
        super().__init__(params, args)

        if args.objective in ["regression", "multiregression"]:
            self.model = DecisionTreeRegressor(
                max_depth=params["max_depth"],
                splitter=params["splitter"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=args.seed,
            )
        elif args.objective in ["classification", "binary"]:
            self.model = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                splitter=params["splitter"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=args.seed,
            )

    @classmethod
    def needs_missing_value_imputation(cls) -> bool:
        return False

    @classmethod
    def needs_cat_encoding(cls) -> bool:
        return True

    @classmethod
    def needs_scaling(cls) -> bool:
        return False

    @classmethod
    def supports_native_multioutput(cls) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        return {
            "max_depth": None,
            "splitter": "best",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }
