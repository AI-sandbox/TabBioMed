"""Run a complete model and dataset evaluation."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from ..models import (
    NO_HPO_MODELS,
    get_model_spec,
    get_recommended_model_configs,
    model_name_map,
)
from ..tabbiomed_datasets import TabularDataset, all_datasets
from ..data import dataset_info
from ..profiles import (
    PROFILES,
    default_output,
    profile_config,
    profile_scope,
    validate_profile,
)
from ..utils import dump_report, load_yaml_config, update_n_trials
from .artifacts import run_lock
from .cli_setup import _set_reproducibility, build_model_args, _setup_optuna_logger
from .runtime import temporary_runtime_environment

CONFIG = Path(__file__).resolve().parents[1] / "configs/benchmark.yaml"
SEARCH_SPACES = CONFIG.parent / "model_search_space"


def input_identity(args, config, data_directory, *, target_encoder=None):
    from .utils.parameter_config import load_model_search_space

    data_directory = Path(data_directory).resolve()
    files = {}
    for name in ("X.npy", "y.npy", "metadata.json", "split_indices.npy"):
        path = data_directory / name
        stat = path.stat()
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                sha.update(chunk)
        files[name] = dict(
            size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha.hexdigest()
        )
    space = get_recommended_model_configs(args.model)
    if space is None:
        space = load_model_search_space(args.model, config.get("search_space_dir"))
    identity = dict(
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        data_root=str(data_directory),
        data_files=files,
        search_space=space,
        config=deepcopy(config),
    )
    if target_encoder is not None:
        identity["target_classes"] = target_encoder.classes_.tolist()
    return json.loads(json.dumps(identity))


def run_dataset(
    dataset,
    metadata,
    *,
    model,
    data_directory,
    output=None,
    config=None,
    seed=42,
    gpu=False,
    profile="main",
):
    """Evaluate a registered model using the dataset's stored partitions."""
    from .selection import run_experiment
    from .default import run_default_model

    model = model_name_map(model)
    spec = get_model_spec(model)
    if metadata["task"] not in spec.tasks:
        raise ValueError(f"{model} does not support {metadata['task']}")
    validate_profile(profile, model, metadata)
    dataset.target_encode()
    config = profile_config(
        load_yaml_config(CONFIG) if config is None else config, profile
    )
    config.setdefault("search_space_dir", str(SEARCH_SPACES))
    config = update_n_trials(config, model)
    hpo_enabled = model not in NO_HPO_MODELS
    if hpo_enabled and (
        not 0 < config["n_trials"] <= 100
        or config.get("trajectory_n_trials", config["n_trials"]) > 200
    ):
        raise ValueError(
            "Selection permits at most 100 official and 200 collected attempts"
        )
    if not hpo_enabled:
        config.update(n_trials=0, trajectory_n_trials=0, hpo_evaluation_budgets=[])
    args = SimpleNamespace(
        model=model,
        dataset=dataset.name,
        objective=metadata["task"],
        seed=seed,
        gpu=gpu,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        num_outputs=getattr(dataset, "num_outputs", 1),
        cat_idx=dataset.cat_idx,
    )
    _set_reproducibility(seed)
    directory = (
        Path(output if output is not None else default_output(profile)).resolve()
        / model
        / dataset.name
        / datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S_%f")
    )
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(
        model=model,
        dataset=dataset.name,
        seed=seed,
        gpu=gpu,
        config=config,
        primary_metric=config["optimize_optuna_with"][metadata["task"]],
        hpo_enabled=hpo_enabled,
        status="incomplete",
        input_identity=input_identity(
            args,
            config,
            data_directory,
            target_encoder=getattr(dataset, "_target_encoder", None),
        ),
    )
    if metadata.get("split_strategy") == "cv":
        report.update(
            n_outer_folds=5,
            cv_outer_folds=5,
            cv_fold_metrics=[
                dict(fold_id=i, default=None, best=None) for i in range(5)
            ],
        )
    if profile != "main":
        report["scope"] = profile_scope(profile)
    with run_lock(directory):
        dump_report(report, directory)
        manifest = dict(
            schema_version=1,
            evaluations=[
                dict(
                    report=str(directory / "report.json"),
                    data_root=str(Path(data_directory).resolve()),
                )
            ],
        )
        if profile != "main":
            manifest["evaluations"][0]["scope"] = profile_scope(profile)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        handler = _setup_optuna_logger(directory)
        try:
            with temporary_runtime_environment():
                runner = run_experiment if hpo_enabled else run_default_model
                runner(
                    args=args,
                    config=config,
                    report_base=report,
                    dataset=dataset,
                    model_args=build_model_args(args, config),
                    metadata=metadata,
                    experiment_dir=directory,
                    optuna_handler=handler,
                )
        finally:
            logging.getLogger("optuna").removeHandler(handler)
            handler.close()
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=all_datasets)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-module", help="Import a module that registers an additional model"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.model_module:
        importlib.import_module(args.model_module)
    validate_profile(
        args.profile, model_name_map(args.model), dataset_info(args.dataset)["metadata"]
    )
    data_root = args.data_root.expanduser()
    config_path = args.config.expanduser()
    output_path = args.output.expanduser() if args.output is not None else None
    metadata_path = data_root / args.dataset / "metadata.json"
    dataset = TabularDataset.read(args.dataset, datasets_path=str(data_root))
    metadata = json.loads(metadata_path.read_text())
    directory = run_dataset(
        dataset,
        metadata,
        model=args.model,
        data_directory=metadata_path.parent,
        output=output_path,
        config=load_yaml_config(config_path),
        seed=args.seed,
        gpu=args.gpu,
        profile=args.profile,
    )
    print(directory)


if __name__ == "__main__":
    main()
