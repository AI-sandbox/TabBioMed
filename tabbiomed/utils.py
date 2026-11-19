from pathlib import Path
from typing import Any

JSONDict = dict[str, Any]
from copy import deepcopy
import json
import math
import os
import tempfile
import numpy as np
import yaml
from loguru import logger
from .trainer.utils.parameter_config import load_model_search_space


def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for (key, value) in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def dump_report(report: JSONDict, output: Path) -> None:
    path = Path(output) / "report.json"
    (fd, temporary) = tempfile.mkstemp(
        prefix=".report.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(fd, path.stat().st_mode & 511 if path.exists() else 436)
        with os.fdopen(fd, "w") as stream:
            json.dump(_convert_numpy_types(report), stream, indent=4, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_yaml_config(path: str) -> dict:
    """
    Loads a yaml file and returns the config as a dictionary

    Args:
        path (str): path to the yaml file

    Returns:
        dict: config as a dictionary

    Raises:
        FileNotFoundError: if the file is not found
    """
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {path}")


def update_n_trials(config: dict, model_name: str) -> dict:
    config_copy = deepcopy(config)
    from .models import NO_HPO_MODELS, get_recommended_model_configs, model_name_map

    if model_name_map(model_name) in NO_HPO_MODELS:
        config_copy["n_trials"] = 0
        return config_copy
    if config_copy["n_trials"] is not None and (
        not isinstance(config_copy["n_trials"], int)
    ):
        if config_copy["n_trials"].lower() in ["inf", "infinite", "unlimited"]:
            config_copy["n_trials"] = None
        else:
            raise ValueError(
                "n_trials in config must be an integer or 'inf' for unlimited trials"
            )
    recommended_configs = get_recommended_model_configs(model_name)
    if recommended_configs is not None:
        limit = config_copy["n_trials"]
        config_copy["n_trials"] = (
            len(recommended_configs)
            if limit is None
            else min(limit, len(recommended_configs))
        )
        return config_copy
    search_space_dir = config_copy.get(
        "search_space_dir",
        str(Path(__file__).resolve().parent / "configs/model_search_space"),
    )
    search_space = load_model_search_space(model_name, config_dir=search_space_dir)
    if not search_space:
        logger.info(
            f"Parameter search space found at {model_name} is empty. Setting n_trials to 1."
        )
        config_copy["n_trials"] = 1
        return config_copy
    search_space.pop("threshold", None)
    if any(
        (
            param_config["type"] in ("float", "conditional")
            for param_config in search_space.values()
        )
    ):
        return config_copy
    n_combinations = 1
    for param_config in search_space.values():
        if param_config["type"] == "categorical":
            n_combinations *= len(param_config["choices"])
        elif param_config["type"] == "int":
            low = int(param_config["low"])
            high = int(param_config["high"])
            step = int(param_config.get("step", 1))
            n_combinations *= (high - low) // step + 1
    if config_copy["n_trials"] is None:
        config_copy["n_trials"] = 10**7
    config_copy["n_trials"] = min(config_copy["n_trials"], n_combinations)
    logger.debug(
        f"found {n_combinations}. Running with {config_copy['n_trials']} trials."
    )
    return config_copy
