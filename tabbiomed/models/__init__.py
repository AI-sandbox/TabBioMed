"""Lazy model registration and benchmark search spaces."""

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import re


TASKS = ("binary", "classification", "regression", "multiregression")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    implementation: object
    search_space: object
    tasks: tuple[str, ...]
    hpo_enabled: bool
    gpu: bool
    requires_separate_env: bool = False


_registry = {}
_aliases = {}
all_models = []
NO_HPO_MODELS = set()


def _key(name):
    return str(name).lower().replace("-", "").replace("_", "").replace(".", "")


def register_model(
    name,
    model_class,
    *,
    search_space=None,
    aliases=(),
    tasks=TASKS,
    hpo_enabled=True,
    gpu=False,
    requires_separate_env=False,
):
    """Register a BaseModel subclass or a ``module:Class`` import path."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name):
        raise ValueError("Model names must be safe directory names")
    names = {name, *aliases}
    if name in _registry or any(_key(alias) in _aliases for alias in names):
        raise ValueError(f"Model or alias is already registered: {name}")
    if not tasks or not set(tasks) <= set(TASKS):
        raise ValueError(f"Unsupported model tasks: {tasks}")
    if hpo_enabled and search_space is None:
        raise ValueError("Tunable models require a search space")
    if not isinstance(model_class, (type, str)):
        raise TypeError("Expected a model class or module:Class import path")
    if isinstance(model_class, str) and ":" not in model_class:
        raise ValueError("Model import paths must use module:Class")
    _registry[name] = ModelSpec(
        name, model_class, deepcopy(search_space), tuple(tasks), hpo_enabled,
        gpu=bool(gpu), requires_separate_env=bool(requires_separate_env),
    )
    _aliases.update({_key(alias): name for alias in names})
    all_models.append(name)
    if not hpo_enabled:
        NO_HPO_MODELS.add(name)


def model_name_map(name):
    try:
        return _aliases[_key(name)]
    except KeyError:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {', '.join(all_models)}"
        ) from None


def get_model_spec(name):
    return _registry[model_name_map(name)]


def str2model(name):
    implementation = get_model_spec(name).implementation
    if isinstance(implementation, str):
        module, attribute = implementation.split(":", 1)
        implementation = getattr(import_module(module), attribute)
    from .base_model import BaseModel

    if not issubclass(implementation, BaseModel):
        raise TypeError(f"{name} must inherit BaseModel")
    return implementation


def get_search_space(name):
    space = get_model_spec(name).search_space
    if space is None:
        return {}
    if isinstance(space, (str, Path)):
        import yaml

        space = yaml.safe_load(Path(space).read_text())
    if not isinstance(space, dict):
        raise TypeError(f"Expected a parameter mapping for {name}")
    return deepcopy(space)


def get_recommended_model_configs(name):
    if model_name_map(name) != "iLTM":
        return None
    from iltm.hyperparameter_search_space import get_hyperparameter_configs

    configurations = get_hyperparameter_configs()
    if len(configurations) != 25:
        raise ValueError(
            f"Expected 25 iLTM configurations, found {len(configurations)}"
        )
    return deepcopy(configurations)


_BUILTINS = [
    ("LinearModel", "linear", "LinearModel", "linearmodel", ("lm",)),
    ("LassoModel", "lasso", "LassoModel", "lassomodel", ("lasso",)),
    ("RidgeModel", "ridge", "RidgeModel", "ridgemodel", ("ridge",)),
    ("ElasticNet", "elasticnet", "ElasticNet", "elasticnet", ("elastic",)),
    ("KNN", "knn", "KNN", "knn", ()),
    ("DecisionTree", "decision_tree", "DecisionTree", "decisiontree", ("dt",)),
    ("RandomForest", "random_forest", "RandomForest", "randomforest", ("rf",)),
    ("XGBoost", "xgboost", "XGBoost", "xgboost", ("xgb",)),
    ("CatBoost", "catboost", "CatBoost", "catboost", ()),
    ("LightGBM", "lightgbm", "LightGBM", "lightgbm", ("lgbm",)),
    ("MLP", "mlp", "MLP", "mlp", ()),
    ("SVM", "svm", "SVM", "svm", ("supportvector",)),
    ("NeoLsSVM", "neo_ls_svm", "NeoLsSVM", "neolssvm", ()),
    ("xRFM", "xrfm", "xRFM", "xrfm", ()),
    ("RealMLP", "realmlp", "RealMLP", "realmlp", ()),
    ("TabPFNv2.5", "tabpfn_v2_5", "TabPFN", "tabpfnv2.5", ("tabpfn", "tabpfn25")),
    ("TabPFN-Wide", "tabpfn_wide", "TabPFNWide", "tabpfnwide", ()),
    (
        "RealTabPFNv2.5",
        "realtabpfn",
        "RealTabPFN",
        "realtabpfnv2.5",
        ("realtabpfn", "realtabpfn25"),
    ),
    ("TabICLv2", "tabicl", "TabICL", "tabiclv2", ("tabicl",)),
    ("TabDPT", "tabdpt", "TabDPT", "tabdpt", ()),
    ("TabM", "tabm", "TabM", "tabm", ()),
    ("HyperFast", "hyperfast", "HyperFast", "hyperfast", ()),
    ("iLTM", "iltm", "iLTM", None, ()),
    ("Causilo-1.5", "causilo", "Causilo", None, ("causilo",)),
    ("Causilo-1.5-Light", "causilo", "CausiloLight", None, ("causilo-light",)),
]
_CONFIG = Path(__file__).resolve().parents[1] / "configs/model_search_space"
for _name, _module, _class, _config, _names in _BUILTINS:
    register_model(
        _name,
        f"tabbiomed.models.model_definitions.{_module}:{_class}",
        search_space=_CONFIG / f"{_config}.yaml" if _config is not None else {},
        aliases=_names,
        requires_separate_env=_name in {"TabPFN-Wide", "Causilo-1.5", "Causilo-1.5-Light"},
        tasks=TASKS[:2] if _name in ("HyperFast", "TabPFN-Wide") else TASKS,
        hpo_enabled=_name not in (
            "LinearModel", "NeoLsSVM", "Causilo-1.5", "Causilo-1.5-Light"
        ),
        gpu=_name
        not in {
            "DecisionTree",
            "RandomForest",
            "LinearModel",
            "LassoModel",
            "RidgeModel",
            "ElasticNet",
            "KNN",
            "SVM",
            "NeoLsSVM",
        },
    )
BENCHMARK_MODELS = tuple(all_models)
