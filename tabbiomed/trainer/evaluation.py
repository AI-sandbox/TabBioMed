import datetime
from pathlib import Path
from loguru import logger
from ..models.base_model import ModelArgs
from ..tabbiomed_datasets import TabularDataset
from .hpo_step_timeout import set_hpo_step_deadline
from .train_eval import (
    evaluate_fit,
    get_fold_model_args,
    _primary_metric_value_ok,
)
from .utils.parameter_config import merge_default_params_with_search_space


def _run_sweep_with_hpo_step_timeout(sweep, timeout_seconds):
    if timeout_seconds is None or timeout_seconds <= 0:
        return sweep.run()
    set_hpo_step_deadline(sweep, timeout_seconds)
    return sweep.run()


def _set_test_completion_status(
    report: dict, metric: str, expected_folds: int | None = None
) -> None:
    """Completion requires saved-test eligibility for every default and selected fit."""
    errors = []
    if expected_folds is None:
        entries = [("", report)]
    else:
        folds = report.get("cv_fold_metrics", [])
        ids = [fold.get("fold_id") for fold in folds]
        if len(ids) != expected_folds or set(ids) != set(range(expected_folds)):
            errors.append(f"Expected folds 0..{expected_folds - 1}; found {ids}")
        entries = [(f"Fold {fold.get('fold_id')}: ", fold) for fold in folds]
        report["expected_outer_folds"] = expected_folds
    required = ("default", "best") if report.get("hpo_enabled", True) else ("default",)
    for prefix, entry in entries:
        for name in required:
            result = entry.get(name) or {}
            if (
                not _primary_metric_value_ok(result.get("test_metrics") or {}, metric)
                or result.get("fixed_split_test_evaluated") is not True
                or result.get("error")
                or result.get("post_hpo_test_error")
            ):
                detail = result.get("post_hpo_test_error") or result.get("error")
                errors.append(
                    f"{prefix}{name} has no successful test evaluation for {metric}"
                    + (f": {detail}" if detail else "")
                )
    report["status"] = "incomplete" if errors else "complete"
    report["completion_errors"] = errors


def run_default_hyperparameter_baseline(
    model_class: type,
    model_args: ModelArgs,
    dataset: TabularDataset,
    metadata: dict,
    experiment_start: datetime.datetime,
    model_name: str | None = None,
    search_space_dir: str = str(
        Path(__file__).resolve().parents[1] / "configs/model_search_space"
    ),
) -> tuple[dict | None, list]:
    try:
        params = model_class.get_default_params()
    except NotImplementedError:
        return None, []

    if model_name:
        params = merge_default_params_with_search_space(
            params, model_name=model_name, config_dir=search_space_dir
        )

    if metadata.get("split_strategy") == "cv":
        logger.info(
            "Skipping global default baseline for CV dataset (handled per fold)."
        )
        return None, []

    params = model_class.resolve_parameters(params, model_args)

    class _Trial:
        __slots__ = ("number",)

        def __init__(self, n: int):
            self.number = n

    trial = _Trial(-1)
    local_test_preds: list = []
    val_m, test_m, tr, v_inf, t_inf = evaluate_fit(
        model_class=model_class,
        params=params,
        model_args=model_args,
        trial=trial,
        dataset=dataset,
        test_preds=local_test_preds,
        evaluate_test=True,
        merge_train_val_for_fit=True,
    )
    if val_m is None:
        return dict(
            parameters=params,
            trial_id=-1,
            validation_metrics={},
            test_metrics={},
            fixed_split_test_evaluated=False,
            error=getattr(dataset, "last_evaluation_error", "Evaluation failed"),
        ), []
    elapsed = (datetime.datetime.now() - experiment_start).total_seconds()
    rep = {
        "parameters": params,
        "validation_metrics": val_m,
        "test_metrics": test_m or {},
        "trial_id": -1,
        "tuning_time": str(datetime.datetime.now() - experiment_start),
        "train_wall_time_seconds": tr,
        "val_inference_wall_time_seconds": v_inf,
        "test_inference_wall_time_seconds": t_inf,
        "elapsed_seconds": elapsed,
        "fixed_split_test_evaluated": True,
        "train_val_merged_for_fit": not model_class.uses_validation_during_fit(),
    }
    return rep, local_test_preds


def _run_default_baseline_on_fixed_split(
    model_class: type,
    model_args: ModelArgs,
    dataset: TabularDataset,
    cat_encoding_strategy: str,
    mv_imputation_strategy: str,
    fold_id: int,
    experiment_start: datetime.datetime,
    model_name: str | None = None,
    search_space_dir: str = str(
        Path(__file__).resolve().parents[1] / "configs/model_search_space"
    ),
) -> tuple[dict | None, list]:
    try:
        params = model_class.get_default_params()
    except NotImplementedError:
        return None, []

    if model_name:
        params = merge_default_params_with_search_space(
            params, model_name=model_name, config_dir=search_space_dir
        )

    class _Trial:
        __slots__ = ("number",)

        def __init__(self, n: int):
            self.number = n

    trial = _Trial(-1)
    local_test_preds: list = []
    ds = dataset
    ds.prepare_dataset(
        inpute_mv=model_class.needs_missing_value_imputation(),
        encode_cat=model_class.needs_cat_encoding(),
        scale_features=False,
        cat_encoding_strategy=cat_encoding_strategy,
        mv_strategy=mv_imputation_strategy,
    )
    fold_model_args = get_fold_model_args(model_args, ds)
    params = model_class.resolve_parameters(params, fold_model_args)
    val_m, test_m, tr, v_inf, t_inf = evaluate_fit(
        model_class=model_class,
        params=params,
        model_args=fold_model_args,
        trial=trial,
        dataset=ds,
        test_preds=local_test_preds,
        force_standard_scaling=False,
        evaluate_test=True,
        merge_train_val_for_fit=True,
    )
    if val_m is None:
        return dict(
            parameters=params,
            trial_id=-1,
            validation_metrics={},
            test_metrics={},
            fixed_split_test_evaluated=False,
            error=getattr(dataset, "last_evaluation_error", "Evaluation failed"),
        ), []
    elapsed = (datetime.datetime.now() - experiment_start).total_seconds()
    rep = {
        "parameters": params,
        "validation_metrics": val_m,
        "test_metrics": test_m or {},
        "trial_id": -1,
        "fold_id": int(fold_id),
        "tuning_time": str(datetime.datetime.now() - experiment_start),
        "train_wall_time_seconds": tr,
        "val_inference_wall_time_seconds": v_inf,
        "test_inference_wall_time_seconds": t_inf,
        "elapsed_seconds": elapsed,
        "fixed_split_test_evaluated": True,
        "train_val_merged_for_fit": not model_class.uses_validation_during_fit(),
    }
    return rep, local_test_preds
