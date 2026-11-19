import logging
import random
from pathlib import Path
from typing import Any
import numpy as np
from ..models.base_model import ModelArgs


def build_model_args(args, config) -> ModelArgs:
    model_args: dict[str, Any] = {
        "objective": args.objective,
        "dataset": args.dataset,
        "model": args.model,
        "use_gpu": args.gpu,
        "batch_size": config.get("batch_size", 128),
        "val_batch_size": config.get("val_batch_size", config.get("batch_size", 128)),
        "epochs": config.get("epochs", 10),
        "early_stopping_rounds": config.get("early_stopping_rounds", 10),
        "logging_period": config.get("logging_period", 10),
        "gpu_ids": [0] if args.gpu else None,
        "data_parallel": config.get("data_parallel", False),
        "cat_idx": getattr(args, "cat_idx", None),
        "num_classes": getattr(args, "num_classes", None),
        "num_features": getattr(args, "num_features", None),
        "num_outputs": getattr(args, "num_outputs", 1),
        "optimize_optuna_with": config.get("optimize_optuna_with", None),
        "seed": getattr(args, "seed", 42),
    }
    return ModelArgs(model_args)


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _setup_optuna_logger(experiment_dir: Path) -> logging.FileHandler:
    optuna_log_path = experiment_dir / "optuna.log"
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.propagate = False
    for handler in optuna_logger.handlers[:]:
        optuna_logger.removeHandler(handler)

    file_handler = logging.FileHandler(str(optuna_log_path), mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )

    original_emit = file_handler.emit

    def flushing_emit(record):
        original_emit(record)
        file_handler.flush()

    file_handler.emit = flushing_emit
    optuna_logger.addHandler(file_handler)

    import optuna

    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    return file_handler
