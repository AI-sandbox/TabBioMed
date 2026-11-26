from __future__ import annotations
from tabbiomed.analysis.paths import PLAN, MODEL_FAMILIES, read_plan
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_INCLUDE_DATASETS = PLAN
DEFAULT_MODEL_FAMILIES = MODEL_FAMILIES
TASKS_DEFAULT = ("binary", "classification", "regression", "multiregression")
PRIMARY_METRIC = {
    "regression": "mse",
    "multiregression": "mse",
    "binary": "roc_auc",
    "classification": "log_loss",
}
HIGHER_IS_BETTER = {
    "roc_auc": True,
    "mse": False,
    "rmse": False,
    "log_loss": False,
    "r2": True,
}


@dataclass(frozen=True)
class BenchmarkModelsConfig:
    model_family: dict[str, str]
    exclude_from_expected_benchmark: frozenset[str]
    regression_optional_missing_models: frozenset[str]
    expected_models_binary_classification: tuple[str, ...]
    expected_models_regression: tuple[str, ...]


def _parse_model_families_payload(data: dict[str, Any]) -> dict[str, str]:
    mf = data.get("model_family")
    if isinstance(mf, dict):
        return {str(k): str(v) for (k, v) in mf.items()}
    raise ValueError("model_families config needs a 'model_family' object")


def load_benchmark_models_config(
    path: Path | None = None, *, plan_path: Path | None = None
) -> BenchmarkModelsConfig:
    data = json.loads(Path(path or DEFAULT_MODEL_FAMILIES).read_text())
    families = _parse_model_families_payload(data)
    excluded = frozenset(data["exclude_from_expected_benchmark"])
    classification_only = frozenset(data["regression_optional_missing_models"])
    plan = read_plan(plan_path or DEFAULT_INCLUDE_DATASETS)
    planned = set(plan["models"])
    missing = planned - set(families)
    if missing:
        raise ValueError(f"Models missing a family assignment: {sorted(missing)}")
    models = tuple(sorted(planned - excluded))
    classification_only = classification_only & set(models)
    return BenchmarkModelsConfig(
        families,
        excluded,
        classification_only,
        models,
        tuple((m for m in models if m not in classification_only)),
    )


def load_include_datasets(path=None):
    return frozenset(read_plan(path or DEFAULT_INCLUDE_DATASETS)["datasets"])


def canon_domain(label: Any) -> str:
    s = str(label or "").strip()
    if not s:
        return "Unlabeled"
    k = s.lower()
    if k == "drug discovery":
        return "Drug Discovery"
    return " ".join((w.capitalize() for w in k.split()))


def analysis_dataset_metadata(
    dataset_name, info, processed_path=None, *, verify_arrays=False
):
    from .datasets import dataset_directory, load_metadata

    metadata = load_metadata(dataset_name, processed_path)
    if metadata["task"] != info["task"]:
        raise ValueError(f"Dataset task differs from benchmark scope: {dataset_name}")
    if verify_arrays:
        dataset_directory(dataset_name, processed_path, arrays=True)
    return metadata


def rank_metric_ascending(task: str) -> bool:
    primary = PRIMARY_METRIC.get(task, "roc_auc")
    higher = HIGHER_IS_BETTER.get(primary, False)
    return not higher


def model_hpo_policy(frame):
    policy = frame[["model", "hpo_enabled"]].drop_duplicates()
    if policy.model.duplicated().any():
        raise ValueError("A model has inconsistent HPO policies across datasets")
    if any(type(value) is not bool for value in policy.hpo_enabled):
        raise ValueError("HPO policy must be explicitly boolean")
    return policy.set_index("model").hpo_enabled
