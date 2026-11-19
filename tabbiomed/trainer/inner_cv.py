"""Prepare independent inner-fold preprocessing and score one HPO configuration."""

import gc
import time

import numpy as np

from ..cv_splits import make_inner_cv_plan
from ..tabbiomed_datasets import TabularDataset
from .cli_setup import _set_reproducibility
from .hpo_step_timeout import HPOStepDeadlineExceeded
from .train_eval import _primary_metric_value_ok, get_fold_model_args, train


def prepare_inner_cv(
    parent,
    outer_dataset,
    model_class,
    *,
    requested_folds,
    seed,
    outer_fold_id,
    mv_strategy,
    cat_encoding_strategy,
):
    """Use raw outer-pool rows; never slice an already imputed outer dataset."""
    plan = make_inner_cv_plan(
        parent.y,
        parent.task,
        outer_dataset.split_indices,
        groups=getattr(parent, "groups", None),
        requested_folds=requested_folds,
        seed=seed,
        outer_fold_id=outer_fold_id,
        stratification=parent.split_indices.get("validation_stratification"),
    )
    outer_dataset.hpo_validation = plan
    outer_dataset.hpo_inner_datasets = []
    if plan["actual_folds"] == 1:
        return plan
    pool = np.sort(
        np.r_[outer_dataset.split_indices["train"], outer_dataset.split_indices["val"]]
    )
    local_index = np.full(parent.num_samples, -1, dtype=int)
    local_index[pool] = np.arange(len(pool))
    raw_X, raw_y = parent.X[pool].copy(), parent.y[pool].copy()
    groups = getattr(parent, "groups", None)
    for split in plan["folds"]:
        inner = TabularDataset(
            name=parent.name,
            X=raw_X,
            y=raw_y,
            cat_idx=list(parent.cat_idx),
            task=parent.task,
            num_classes=parent.num_classes,
            cat_dims=list(parent.cat_dims) if parent.cat_dims is not None else None,
            split_indices={
                "train": local_index[split["train"]].tolist(),
                "val": local_index[split["val"]].tolist(),
                "test": [],
            },
            source=parent.source,
            description=parent.description,
            label=parent.biomed_label,
            force_scale=outer_dataset.force_scale,
            groups=None if groups is None else groups[pool],
        )
        inner.is_cv_fold = True
        inner.prepare_dataset(
            inpute_mv=model_class.needs_missing_value_imputation(),
            encode_cat=model_class.needs_cat_encoding(),
            scale_features=False,
            mv_strategy=mv_strategy,
            cat_encoding_strategy=cat_encoding_strategy,
        )
        outer_dataset.hpo_inner_datasets.append(inner)
    return plan


def train_inner_cv(model_class, params, model_args, trial, dataset, test_preds):
    """One attempted configuration comprises every inner fit, under one deadline.

    Returns the ordinary train() five fields plus inner-fold diagnostics. No
    outer-test array or prediction is available to these training calls.
    """
    folds = dataset.hpo_inner_datasets
    if len(folds) < 2:
        raise ValueError("Inner CV requires at least two prepared folds")
    primary = (getattr(model_args, "optimize_optuna_with", None) or {}).get(
        model_args.objective,
        {"binary": "roc_auc", "classification": "log_loss"}.get(
            model_args.objective, "mse"
        ),
    )
    diagnostics, metrics = [], []
    total_train, total_val = 0.0, 0.0

    def check_deadline():
        deadline = getattr(dataset, "hpo_step_deadline_monotonic", None)
        if deadline is not None and time.monotonic() >= deadline:
            raise HPOStepDeadlineExceeded(
                "Outer-fold HPO budget exhausted during inner CV"
            )

    for fold_id, inner in enumerate(folds):
        check_deadline()
        _set_reproducibility(model_args.seed)
        fold_args = get_fold_model_args(model_args, inner)
        val, _, elapsed, inference, _ = train(
            model_class,
            params,
            fold_args,
            trial,
            inner,
            [],
            evaluate_test=False,
            merge_train_val_for_fit=False,
        )
        check_deadline()
        total_train += float(elapsed or 0.0)
        total_val += float(inference or 0.0)
        ok = _primary_metric_value_ok(val, primary)
        diagnostics.append(
            dict(
                inner_fold_id=fold_id,
                train_samples=len(inner.y_train),
                validation_samples=len(inner.y_val),
                validation_metrics=val or {},
                train_wall_time_seconds=elapsed,
                val_inference_wall_time_seconds=inference,
                status="success" if ok else "failed",
            )
        )

        import delu

        delu.cuda.free_memory()
        gc.collect()
        if not ok:
            return (
                None,
                {},
                total_train,
                total_val,
                None,
                {
                    "inner_cv_metrics": diagnostics,
                    "inner_cv_error": getattr(
                        inner,
                        "last_evaluation_error",
                        f"Inner fold {fold_id} has no finite {primary}",
                    ),
                },
            )
        metrics.append(val)
    common = set.intersection(*(set(m) for m in metrics))
    mean = {
        key: float(np.mean([m[key] for m in metrics]))
        for key in sorted(common)
        if all(_primary_metric_value_ok(m, key) for m in metrics)
    }
    return (
        mean,
        {},
        total_train,
        total_val,
        None,
        {"inner_cv_metrics": diagnostics, "inner_cv_folds": len(folds)},
    )
