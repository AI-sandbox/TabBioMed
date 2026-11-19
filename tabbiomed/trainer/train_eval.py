import math
import time
from typing import Any

import numpy as np
import optuna
from loguru import logger

from tabbiomed.trainer.utils.eval_utils import evaluate_split
from tabbiomed.trainer.utils.preprocessing import scale_splits_fit_on_train
from ..models.base_model import ModelArgs
from ..tabbiomed_datasets import TabularDataset


def _concatenate_y_for_train_val_merge(
    y_train: np.ndarray, y_val: np.ndarray
) -> np.ndarray:
    yt = np.asarray(y_train)
    yv = np.asarray(y_val)
    if yt.ndim == 1:
        return np.concatenate([yt, yv])
    if yt.ndim == 2:
        return np.vstack([yt, yv])
    raise ValueError(f"Unsupported y ndim for train+val merge: {yt.ndim}")


def get_fold_model_args(
    model_args: ModelArgs, fold_dataset: TabularDataset
) -> ModelArgs:
    """Build ModelArgs for one fold using encoded fold dataset dimensions."""
    d = {
        "objective": model_args.objective,
        "dataset": model_args.dataset,
        "is_cv": (
            getattr(model_args, "is_cv", False)
            or getattr(fold_dataset, "is_cv_fold", False)
            or "cv_folds" in (getattr(fold_dataset, "split_indices", None) or {})
        ),
        "model": model_args.model,
        "num_cpus": model_args.num_cpus,
        "use_gpu": model_args.use_gpu,
        "epochs": model_args.epochs,
        "early_stopping_rounds": model_args.early_stopping_rounds,
        "logging_period": model_args.logging_period,
        "gpu_ids": model_args.gpu_ids,
        "cat_idx": getattr(fold_dataset, "cat_idx", model_args.cat_idx),
        "num_classes": fold_dataset.num_classes,
        "num_features": fold_dataset.num_features,
        "num_outputs": getattr(fold_dataset, "num_outputs", 1),
        "data_parallel": model_args.data_parallel,
        "batch_size": model_args.batch_size,
        "val_batch_size": getattr(model_args, "val_batch_size", model_args.batch_size),
        "seed": getattr(model_args, "seed", 42),
        "optimize_optuna_with": getattr(model_args, "optimize_optuna_with", None),
    }
    return ModelArgs(d)


def train(
    model_class: type,
    params: dict,
    model_args: ModelArgs,
    trial: Any,
    dataset: TabularDataset,
    test_preds: list,
    force_standard_scaling: bool = False,
    evaluate_test: bool = True,
    merge_train_val_for_fit: bool = False,
) -> tuple[dict | None, dict | None, float | None, float | None, float | None]:
    model_class.validate_parameters(params, model_args)
    _needs_scaling = model_class.needs_scaling() or (
        getattr(dataset, "force_scale", False)
        and getattr(model_class, "allow_metadata_scaling", False)
    )
    _cv_fold_skip_copy = getattr(dataset, "is_cv_fold", False) and not _needs_scaling
    if _cv_fold_skip_copy:
        X_train_base, y_train = dataset.X_train, dataset.y_train
        X_val_base, y_val = dataset.X_val, dataset.y_val
        X_test_base, y_test = dataset.X_test, dataset.y_test
    else:
        X_train_base, y_train = dataset.X_train.copy(), dataset.y_train
        X_val_base, y_val = dataset.X_val.copy(), dataset.y_val
        X_test_base, y_test = dataset.X_test.copy(), dataset.y_test

    if _needs_scaling:
        if not model_class.needs_scaling() and getattr(dataset, "force_scale", False):
            logger.info(
                f"Forcing feature scaling for {dataset.name} as requested in metadata "
                f"(model: {getattr(model_args, 'model', '')})"
            )
        if force_standard_scaling:
            scaler_type = "standard"
            logger.info(
                "CV mode detected: forcing standard scaling (scaler not tuned as hyperparameter)"
            )
        else:
            scaler_type = params["scaler"]
        random_state = getattr(model_args, "seed", None)

        X_train_base, X_val_base, X_test_base, _scaler, _plan = (
            scale_splits_fit_on_train(
                X_train_base,
                X_val_base,
                X_test_base,
                cat_idx=getattr(dataset, "cat_idx", None),
                scaler_type=scaler_type,
                random_state=random_state,
            )
        )

    if X_val_base.shape[0] == 0:
        raise ValueError(
            f"Validation split is empty for dataset '{getattr(dataset, 'name', '')}'. "
            "Please provide non-empty 'val' indices."
        )

    X_train, X_val, X_test = X_train_base, X_val_base, X_test_base

    uses_val_in_fit = model_class.uses_validation_during_fit()
    do_merge = merge_train_val_for_fit and not uses_val_in_fit
    if merge_train_val_for_fit and uses_val_in_fit:
        logger.info(
            f"{model_class.__name__} uses validation during fit; "
            "training on the train split only (merge_train_val_for_fit ignored)."
        )

    X_fit, y_fit = X_train, y_train
    X_fit_val: np.ndarray | None = X_val
    y_fit_val = y_val
    if do_merge:
        logger.info(
            f"{model_class.__name__}: fitting on concatenated train+val; "
            "skipping validation-set evaluation (test split unchanged)."
        )
        X_fit = np.vstack([X_train, X_val])
        y_fit = _concatenate_y_for_train_val_merge(y_train, y_val)
        X_fit_val = None
        y_fit_val = None

    train_wall_time_seconds: float | None = None
    try:
        model = model_class(params, model_args)
        logger.info("Training model")
        t_fit0 = time.perf_counter()
        if model_args.objective == "multiregression":
            model.fit_multioutput(X_fit, y_fit, X_fit_val, y_fit_val)
        else:
            model.fit(X_fit, y_fit, X_fit_val, y_fit_val)
        train_wall_time_seconds = time.perf_counter() - t_fit0
    except Exception as e:
        dataset.last_evaluation_error = f"{type(e).__name__}: {e}"
        import traceback

        logger.error(f"Error training model at trial {trial.number}: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None

    compute_proba = model_args.objective in ("classification", "binary")
    try:
        if do_merge:
            val_metrics: dict = {}
            val_inference_wall_time_seconds: float | None = None
        else:
            val_res = evaluate_split(
                model,
                X_val,
                y_val,
                objective=model_args.objective,
                compute_proba=compute_proba,
            )
            val_inference_wall_time_seconds = val_res.inference_wall_time_seconds
            val_metrics = val_res.metrics

        test_inference_wall_time_seconds: float | None = None
        if evaluate_test and X_test.shape[0] == 0:
            raise ValueError(
                f"Test split is empty for dataset '{getattr(dataset, 'name', '')}' but "
                f"evaluate_test=True. Please provide non-empty 'test' indices."
            )
        if X_test.shape[0] > 0 and evaluate_test:
            test_res = evaluate_split(
                model,
                X_test,
                y_test,
                objective=model_args.objective,
                compute_proba=compute_proba,
            )
            test_inference_wall_time_seconds = test_res.inference_wall_time_seconds
            test_metrics = test_res.metrics
            test_preds.append({"y_pred": test_res.y_pred, "y_proba": test_res.y_proba})
        else:
            test_metrics = {}
            test_preds.append({"y_pred": None, "y_proba": None})

        return (
            val_metrics,
            test_metrics,
            train_wall_time_seconds,
            val_inference_wall_time_seconds,
            test_inference_wall_time_seconds,
        )
    except Exception as e:
        dataset.last_evaluation_error = f"{type(e).__name__}: {e}"
        import traceback

        logger.error(
            f"Error during val/test evaluation at trial {trial.number} "
            f"(e.g. TabPFN preprocessing overflow): {e}"
        )
        logger.error(traceback.format_exc())
        return None, None, train_wall_time_seconds, None, None


def _report_index_for_trial_number(trial_reports: list, trial_number: int) -> int:
    for i, report in enumerate(trial_reports):
        if report.get("trial_id") == trial_number:
            return i
    if 0 <= trial_number < len(trial_reports):
        return trial_number
    return -1


def apply_fixed_split_test_eval_for_best_trial(
    study: optuna.Study,
    trial_reports: list,
    dataset: TabularDataset,
    model_class: type,
    model_args: ModelArgs,
    test_preds: list,
) -> None:
    if dataset.X_test.shape[0] == 0 or not trial_reports:
        return
    try:
        best_trial = study.best_trial
    except (ValueError, AttributeError, RuntimeError):
        return

    tid = best_trial.number
    idx = _report_index_for_trial_number(trial_reports, tid)
    if idx < 0:
        return
    rep = trial_reports[idx]
    if not isinstance(rep, dict) or "error" in rep or "parameters" not in rep:
        return
    default_metric = {"binary": "roc_auc", "classification": "log_loss"}.get(
        model_args.objective, "mse"
    )
    primary_metric = (getattr(model_args, "optimize_optuna_with", None) or {}).get(
        model_args.objective, default_metric
    )
    if rep.get("fixed_split_test_evaluated") and _primary_metric_value_ok(
        rep.get("test_metrics"), primary_metric
    ):
        return
    rep["fixed_split_test_evaluated"] = False

    class _DummyTrial:
        __slots__ = ("number",)

        def __init__(self, n: int):
            self.number = n

    local_preds: list = []
    try:
        _val_m, test_m, tr_sec, v_inf, t_inf = evaluate_fit(
            model_class=model_class,
            params=rep["parameters"],
            model_args=model_args,
            trial=_DummyTrial(tid),
            dataset=dataset,
            test_preds=local_preds,
            evaluate_test=True,
            merge_train_val_for_fit=True,
        )
    except Exception as e:
        rep["post_hpo_test_error"] = str(e)
        logger.error(
            f"Fixed-split post-HPO test evaluation failed for best trial {tid}: {e}"
        )
        import traceback

        logger.error(traceback.format_exc())
        return

    rep["test_metrics"] = test_m or {}
    rep["test_inference_wall_time_seconds"] = t_inf
    if rep.get("inner_cv_metrics"):
        rep["post_hpo_val_inference_wall_time_seconds"] = v_inf
    else:
        rep["val_inference_wall_time_seconds"] = v_inf
    rep["post_hpo_train_wall_time_seconds"] = tr_sec
    rep["train_val_merged_for_fit"] = not model_class.uses_validation_during_fit()
    if not _primary_metric_value_ok(test_m, primary_metric):
        rep["post_hpo_test_error"] = getattr(
            dataset,
            "last_evaluation_error",
            f"Final refit did not produce a finite test metric '{primary_metric}'.",
        )
        logger.error(rep["post_hpo_test_error"])
        return
    rep["fixed_split_test_evaluated"] = True
    rep.pop("post_hpo_test_error", None)
    if local_preds:
        test_preds.append(local_preds[-1])


def aggregate_cv_metrics(all_metrics: list) -> dict:
    if not all_metrics:
        return {}
    metric_names = list(all_metrics[0].keys())
    aggregated = {}
    for metric_name in metric_names:
        metric_values = [m[metric_name] for m in all_metrics if metric_name in m]
        if metric_values:
            aggregated[metric_name] = float(np.mean(metric_values))
            aggregated[f"{metric_name}_std"] = float(np.std(metric_values))
            aggregated[f"{metric_name}_n_folds"] = len(metric_values)
    return aggregated


CV_TIMING_FIELDS_DEFAULT = (
    "train_wall_time_seconds",
    "val_inference_wall_time_seconds",
    "test_inference_wall_time_seconds",
    "elapsed_seconds",
)
CV_TIMING_FIELDS_BEST = CV_TIMING_FIELDS_DEFAULT + ("post_hpo_train_wall_time_seconds",)


def aggregate_cv_report_timing_fields(
    fold_reports: list[dict], which: str, field_names: tuple[str, ...]
) -> dict[str, float | int]:
    """Mean / std / n_folds across folds for scalar timing fields on each fold's report."""
    aggregated: dict[str, float | int] = {}
    for name in field_names:
        values: list[float] = []
        for fr in fold_reports:
            if not isinstance(fr, dict):
                continue
            rep = fr.get(which)
            if not isinstance(rep, dict) or "error" in rep:
                continue
            v = rep.get(name)
            if v is None:
                continue
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                continue
        if values:
            aggregated[name] = float(np.mean(values))
            aggregated[f"{name}_std"] = float(np.std(values))
            aggregated[f"{name}_n_folds"] = len(values)
    return aggregated


def _primary_metric_value_ok(val_metrics: dict, primary_metric: str) -> bool:
    if not val_metrics or not isinstance(val_metrics, dict):
        return False
    if primary_metric not in val_metrics:
        return False
    v = val_metrics[primary_metric]
    if v is None:
        return False
    try:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return False
    except (TypeError, ValueError):
        return False
    return True


def evaluate_fit(**kwargs):
    from types import SimpleNamespace
    from .hpo_step_timeout import run_train_with_exec_timeout, TrialProcessError

    kwargs["model_class"].validate_parameters(kwargs["params"], kwargs["model_args"])
    isolated = {**kwargs, "trial": SimpleNamespace(number=kwargs["trial"].number)}
    try:
        result = run_train_with_exec_timeout(
            train, isolated, 24 * 60 * 60, raise_on_failure=True
        )
        if result is None:
            raise RuntimeError("Evaluation process reached its deadline")
    except (TrialProcessError, RuntimeError) as error:
        kwargs["dataset"].last_evaluation_error = str(error)
        return None, None, None, None, None
    output, predictions = result
    kwargs["test_preds"].extend(predictions)
    if len(output) > 5 and output[0] is None:
        kwargs["dataset"].last_evaluation_error = output[5].get(
            "fit_error", "Evaluation failed"
        )
    return output[:5]
