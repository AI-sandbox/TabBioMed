from typing import Dict, Any, Optional
from copy import deepcopy
import math
import yaml
import optuna
from pathlib import Path

from loguru import logger


def encode_weighted_categorical_parameters(parameters, search_space):
    """Encode effective categorical values for Optuna's enqueue_trial API.

    Weighted categorical sampling below uses a uniform numerical suggestion.
    Enqueue the midpoint of the matching interval to retain the requested value.
    Other parameters are returned unchanged; declared numerical transforms are
    outside the scope of this helper.
    """
    suggestions = dict(parameters)
    for name, spec in search_space.items():
        weights = spec.get("weights")
        if weights is None or name not in suggestions:
            continue
        choices = spec.get("choices", [])
        if (
            len(weights) != len(choices)
            or not choices
            or any(not math.isfinite(w) or w <= 0 for w in weights)
        ):
            raise ValueError(f"Invalid categorical weights for {name}")
        index = choices.index(suggestions[name])
        suggestions[name] = (
            math.fsum(weights[:index]) + weights[index] / 2
        ) / math.fsum(weights)
    return suggestions


def load_model_search_space(model_name: str, config_dir=None) -> Dict[str, Any]:
    from tabbiomed.models import get_search_space

    packaged = Path(__file__).resolve().parents[2] / "configs/model_search_space"
    if config_dir is None or Path(config_dir).resolve() == packaged.resolve():
        return get_search_space(model_name)
    filename = model_name.lower().replace("-", "").replace("_", "") + ".yaml"
    path = Path(config_dir) / filename
    with path.open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a search-space mapping: {path}")
    parent = value.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError(f"Expected a registered model name in extends: {path}")
        value = {**get_search_space(parent), **value}
    return value


def singleton_search_space_overrides(
    model_name: str,
    config_dir: str = None,
) -> Dict[str, Any]:
    """Return search-space params that are fixed to a single value.

    Used to align the default-hyperparameter baseline with experiment search
    spaces that pin flags such as ``apply_pca: [true]`` (e.g. TFM+PCA runs).
    Multi-choice / continuous hyperparameters are left to Optuna and ignored here.
    """
    from tabbiomed.models import get_recommended_model_configs

    if get_recommended_model_configs(model_name) is not None:
        return {}

    space = load_model_search_space(model_name, config_dir=config_dir)
    if not space:
        return {}

    overrides: Dict[str, Any] = {}
    for param_name, param_config in space.items():
        if not isinstance(param_config, dict) or param_config.get("hpo_only", False):
            continue
        ptype = param_config.get("type")
        if ptype == "categorical":
            choices = param_config.get("choices") or []
            if len(choices) == 1:
                overrides[param_name] = choices[0]
        elif ptype == "fixed" and "value" in param_config:
            overrides[param_name] = param_config["value"]
    return overrides


def merge_default_params_with_search_space(
    default_params: Dict[str, Any],
    model_name: str,
    config_dir: str = None,
) -> Dict[str, Any]:
    """Overlay singleton search-space values onto model default params."""
    overrides = singleton_search_space_overrides(model_name, config_dir=config_dir)
    if not overrides:
        return dict(default_params)
    merged = dict(default_params)
    merged.update(overrides)
    logger.info(
        "Default baseline: applying fixed search-space overrides {}: {}",
        model_name,
        overrides,
    )
    return merged


def _normalize_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "").replace("_", "")


def _catboost_gpu_multireg_needs_symmetric_tree(
    model_name: str, dataset_metadata: Optional[Dict[str, Any]]
) -> bool:
    """CatBoost MultiRMSE on GPU does not support Depthwise grow policy."""
    if not dataset_metadata:
        return False
    if _normalize_model_name(model_name) != "catboost":
        return False
    if dataset_metadata.get("objective") != "multiregression":
        return False
    return bool(dataset_metadata.get("use_gpu"))


def _filter_categorical_choices(
    param_name: str,
    choices: list,
    model_name: str,
    dataset_metadata: Optional[Dict[str, Any]],
) -> list:
    if param_name != "grow_policy" or not _catboost_gpu_multireg_needs_symmetric_tree(
        model_name, dataset_metadata
    ):
        return choices

    filtered = [c for c in choices if c == "SymmetricTree"]
    if not filtered:
        raise ValueError(
            "CatBoost GPU multiregression requires grow_policy SymmetricTree, "
            f"but search space choices are {choices!r}."
        )
    if len(filtered) < len(choices):
        logger.info(
            "CatBoost GPU multiregression: sampling grow_policy from ['SymmetricTree'] only "
            "(Depthwise is unsupported with MultiRMSE on GPU)."
        )
    return filtered


def define_trial_parameters_from_config(
    trial: optuna.Trial,
    model_name: str,
    config_dir: str = None,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Define trial parameters based on configuration file instead of model-specific methods.

    Args:
        trial: Optuna trial object
        model_name: Name of the model
        config_dir: Directory containing model search space configurations
        dataset_metadata: Optional dictionary with dataset information for dynamic constraints.
                         KNN requires 'n_train_samples', the actual rows fitted per trial.
                         May contain 'objective', 'use_gpu' (CatBoost grow_policy on GPU multiregression).

    Returns:
        Dictionary of trial parameters
    """
    search_space = load_model_search_space(model_name, config_dir)
    objective = (dataset_metadata or {}).get("objective")
    if objective is not None:
        search_space = {
            name: spec
            for name, spec in search_space.items()
            if "objectives" not in spec or objective in spec["objectives"]
        }
    params = {}

    max_valid_k = None
    if model_name.lower() == "knn":
        max_valid_k = (dataset_metadata or {}).get("n_train_samples")
        if max_valid_k is None:
            raise ValueError(
                "KNN search requires n_train_samples from the actual training split"
            )
        if max_valid_k < 1:
            raise ValueError("KNN search requires at least one actual training row")
        logger.debug(
            f"KNN constraint: max_valid_k={max_valid_k} (actual training rows)"
        )

    for param_name, param_config in search_space.items():
        param_type = param_config.get("type", "float")

        if param_type == "conditional":
            continue

        if param_type == "fixed":
            if "value" not in param_config:
                logger.warning(
                    f"Parameter '{param_name}' is type 'fixed' but missing 'value'. Skipping."
                )
                continue
            params[param_name] = param_config["value"]
            continue

        if "choices" in param_config:
            choices = param_config["choices"]

            if (
                param_name == "n_neighbors"
                and model_name.lower() == "knn"
                and max_valid_k is not None
            ):
                filtered_choices = [c for c in choices if c <= max_valid_k]

                if not filtered_choices:
                    raise ValueError(
                        f"No valid n_neighbors choices for {max_valid_k} actual training rows "
                        f"(max_valid_k={max_valid_k}). Original choices: {choices}"
                    )

                if len(filtered_choices) < len(choices):
                    logger.info(
                        f"Filtered n_neighbors choices from {len(choices)} to {len(filtered_choices)} "
                        f"based on actual training rows (max_valid_k={max_valid_k})"
                    )

                params[param_name] = trial.suggest_categorical(
                    param_name, filtered_choices
                )
            else:
                weights = param_config.get("weights")
                if weights is not None:
                    if (
                        len(weights) != len(choices)
                        or not choices
                        or any(not math.isfinite(w) or w <= 0 for w in weights)
                    ):
                        raise ValueError(
                            f"Invalid categorical weights for {param_name}"
                        )

                    draw = trial.suggest_float(param_name, 0.0, 1.0)
                    total = math.fsum(weights)
                    cumulative = 0.0
                    params[param_name] = choices[-1]
                    for choice, weight in zip(choices, weights):
                        cumulative += weight / total
                        if draw < cumulative:
                            params[param_name] = choice
                            break
                else:
                    choices = _filter_categorical_choices(
                        param_name, choices, model_name, dataset_metadata
                    )
                    params[param_name] = trial.suggest_categorical(param_name, choices)

        elif param_type == "int":
            if "low" not in param_config or "high" not in param_config:
                logger.warning(
                    f"Parameter '{param_name}' is type 'int' but missing 'low' or 'high'. Skipping."
                )
                continue
            low = param_config["low"]
            high = param_config["high"]

            low = int(float(low)) if isinstance(low, str) else int(low)
            high = int(float(high)) if isinstance(high, str) else int(high)
            if param_name == "n_neighbors" and max_valid_k is not None:
                high = min(high, max_valid_k)
                if low > high:
                    raise ValueError(
                        f"No valid n_neighbors range for {max_valid_k} actual training rows"
                    )
            log = param_config.get("log", False)
            step = param_config.get("step", 1)
            params[param_name] = trial.suggest_int(
                param_name, low, high, log=log, step=step
            )

        elif param_type == "float":
            if "low" not in param_config or "high" not in param_config:
                logger.warning(
                    f"Parameter '{param_name}' is type 'float' but missing 'low' or 'high'. Skipping."
                )
                continue
            low = param_config["low"]
            high = param_config["high"]

            low = float(low) if isinstance(low, str) else float(low)
            high = float(high) if isinstance(high, str) else float(high)
            log = param_config.get("log", False)
            params[param_name] = trial.suggest_float(param_name, low, high, log=log)

        elif param_type == "categorical":
            choices = param_config["choices"]

            if (
                param_name == "n_neighbors"
                and model_name.lower() == "knn"
                and max_valid_k is not None
            ):
                filtered_choices = [c for c in choices if c <= max_valid_k]

                if not filtered_choices:
                    raise ValueError(
                        f"No valid n_neighbors choices for {max_valid_k} actual training rows "
                        f"(max_valid_k={max_valid_k}). Original choices: {choices}"
                    )

                if len(filtered_choices) < len(choices):
                    logger.info(
                        f"Filtered n_neighbors choices from {len(choices)} to {len(filtered_choices)} "
                        f"based on actual training rows (max_valid_k={max_valid_k})"
                    )

                params[param_name] = trial.suggest_categorical(
                    param_name, filtered_choices
                )
            else:
                choices = _filter_categorical_choices(
                    param_name, choices, model_name, dataset_metadata
                )
                params[param_name] = trial.suggest_categorical(param_name, choices)

        else:
            logger.warning(
                f"Unknown parameter type '{param_type}' for parameter '{param_name}' in {model_name}"
            )

    for param_name, param_config in search_space.items():
        param_type = param_config.get("type", "float")

        if param_type == "conditional":
            depends_on = param_config.get("depends_on")
            if depends_on is None:
                logger.warning(
                    f"Conditional parameter '{param_name}' missing 'depends_on'. Skipping."
                )
                continue

            if depends_on not in params:
                logger.warning(
                    f"Conditional parameter '{param_name}' depends on '{depends_on}' which was not set. Skipping."
                )
                continue

            dependency_value = params[depends_on]

            if dependency_value not in param_config:
                logger.warning(
                    f"Conditional parameter '{param_name}' has no value for dependency '{depends_on}' = '{dependency_value}'. Skipping."
                )
                continue

            conditional_value = param_config[dependency_value]

            if isinstance(conditional_value, list):
                params[param_name] = trial.suggest_categorical(
                    param_name, conditional_value
                )
            elif isinstance(conditional_value, (int, float, bool, str)):
                params[param_name] = conditional_value
            elif isinstance(conditional_value, dict):
                nested_type = conditional_value.get("type", "float")
                if nested_type == "int":
                    if (
                        "low" not in conditional_value
                        or "high" not in conditional_value
                    ):
                        logger.warning(
                            f"Conditional parameter '{param_name}' nested dict missing 'low' or 'high'. Skipping."
                        )
                        continue
                    low = (
                        int(float(conditional_value["low"]))
                        if isinstance(conditional_value["low"], str)
                        else int(conditional_value["low"])
                    )
                    high = (
                        int(float(conditional_value["high"]))
                        if isinstance(conditional_value["high"], str)
                        else int(conditional_value["high"])
                    )
                    log = conditional_value.get("log", False)
                    params[param_name] = trial.suggest_int(
                        param_name, low, high, log=log
                    )
                elif nested_type == "float":
                    if (
                        "low" not in conditional_value
                        or "high" not in conditional_value
                    ):
                        logger.warning(
                            f"Conditional parameter '{param_name}' nested dict missing 'low' or 'high'. Skipping."
                        )
                        continue
                    low = (
                        float(conditional_value["low"])
                        if isinstance(conditional_value["low"], str)
                        else float(conditional_value["low"])
                    )
                    high = (
                        float(conditional_value["high"])
                        if isinstance(conditional_value["high"], str)
                        else float(conditional_value["high"])
                    )
                    log = conditional_value.get("log", False)
                    params[param_name] = trial.suggest_float(
                        param_name, low, high, log=log
                    )
                elif nested_type == "categorical":
                    if "choices" not in conditional_value:
                        logger.warning(
                            f"Conditional parameter '{param_name}' nested dict missing 'choices'. Skipping."
                        )
                        continue
                    params[param_name] = trial.suggest_categorical(
                        param_name, conditional_value["choices"]
                    )
                else:
                    logger.warning(
                        f"Conditional parameter '{param_name}' has unsupported nested type '{nested_type}'. Skipping."
                    )
            else:
                logger.warning(
                    f"Conditional parameter '{param_name}' has unsupported value type for '{dependency_value}'. Skipping."
                )

    expected = {
        name
        for name, spec in search_space.items()
        if spec.get("type") != "conditional"
        or spec["depends_on"] not in params
        or params[spec["depends_on"]] in spec
    }
    missing = expected - params.keys()
    if missing:
        raise ValueError(
            f"{model_name}: search did not supply parameters {sorted(missing)}"
        )

    transformed = any("weights" in spec for spec in search_space.values())
    for param_name, param_config in search_space.items():
        transform = param_config.get("transform")
        if transform is None or param_name not in params:
            continue
        if transform == "one_minus":
            params[param_name] = 1.0 - params[param_name]
        elif transform == "floor":
            params[param_name] = math.floor(params[param_name])
        else:
            raise ValueError(
                f"Unknown parameter transform {transform!r} for {param_name}"
            )
        transformed = True
    if transformed:
        trial.set_user_attr("sampled_parameters", deepcopy(params))
    return params
