"""Evaluate models whose benchmark protocol has no hyperparameter search."""

from datetime import datetime
from ..models import str2model
from ..utils import dump_report
from .artifacts import save_predictions
from .cli_setup import _set_reproducibility
from .evaluation import (
    run_default_hyperparameter_baseline,
    _run_default_baseline_on_fixed_split,
)
from .hpo_trajectory import save_test_labels
from .selection import update_report, test_ok
from .train_eval import get_fold_model_args


def run_default_model(
    *,
    args,
    config,
    report_base,
    dataset,
    model_args,
    metadata,
    experiment_dir,
    optuna_handler,
):
    cls = str2model(args.model)
    metric = report_base["primary_metric"]
    report = dict(
        report_base, hpo_enabled=False, hpo_finished=True, actual_configurations=0
    )
    is_cv = metadata.get("split_strategy") == "cv"
    if is_cv:
        report.update(n_outer_folds=5, cv_outer_folds=5)
    folds = (
        ((i, ds) for i, (ds, _, _) in enumerate(dataset.get_cv_splits()))
        if is_cv
        else [(None, dataset)]
    )
    cat = (
        cls.get_cat_encoding_strategy()
        if hasattr(cls, "get_cat_encoding_strategy")
        else "onehot"
    )
    mv = (
        cls.get_mv_imputation_strategy()
        if hasattr(cls, "get_mv_imputation_strategy")
        else "knn"
    )
    for fold_id, ds in folds:
        if not is_cv:
            ds.prepare_dataset(
                inpute_mv=cls.needs_missing_value_imputation(),
                encode_cat=cls.needs_cat_encoding(),
                scale_features=False,
                mv_strategy=mv,
                cat_encoding_strategy=cat,
            )
        _set_reproducibility(args.seed)
        call = dict(
            model_class=cls,
            model_args=get_fold_model_args(model_args, ds),
            dataset=ds,
            experiment_start=datetime.now(),
            model_name=args.model,
            search_space_dir=config["search_space_dir"],
        )
        if is_cv:
            baseline, predictions = _run_default_baseline_on_fixed_split(
                fold_id=fold_id,
                cat_encoding_strategy=cat,
                mv_imputation_strategy=mv,
                **call,
            )
        else:
            baseline, predictions = run_default_hyperparameter_baseline(
                metadata=metadata, **call
            )
        prefix = "" if fold_id is None else f"fold_{fold_id}_"
        if test_ok(baseline, metric):
            save_predictions(experiment_dir, prefix + "default", predictions)
        save_test_labels(experiment_dir, ds, fold_id)
        if is_cv:
            report["cv_fold_metrics"][fold_id] = dict(
                fold_id=fold_id,
                default=baseline,
                best=None,
                actual_configurations=0,
            )
        else:
            report.update(default=baseline, best=None)
        update_report(report, is_cv)
        dump_report(report, experiment_dir)
