from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from ..budgets import completed_within_deadline
from .hpo_trajectory import _save_budget_report, select_budget_trial
from .train_eval import _primary_metric_value_ok

PROTOCOL = "validation_selection_v1"


def read_events(run):
    path = Path(run) / "hpo_trials.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return events


def load_trials(run, fold_id, events=None):
    events = read_events(run) if events is None else events
    searches = [
        e["search_id"]
        for e in events
        if e.get("event") == "search_started" and e.get("fold_id") == fold_id
    ]
    if not searches:
        return []
    (starts, trials) = ({}, {})
    for e in events:
        if e.get("search_id") != searches[-1] or "trial_id" not in e:
            continue
        tid = e["trial_id"]
        trials[tid] = {**trials.get(tid, {}), **e}
        if e["event"] == "trial_started":
            starts[tid] = e["hpo_elapsed_seconds"]
    for tid, trial in trials.items():
        trial["started_at_hpo_seconds"] = starts[tid]
        if trial["event"] != "trial_finished":
            trial.update(
                status="interrupted", error="Process stopped before trial completion"
            )
    return [trials[i] for i in sorted(trials)]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_budget_report(value, path)


def stage_directory(run_dir, fold_id):
    return (
        Path(run_dir)
        / "selection"
        / ("fixed" if fold_id is None else f"fold_{fold_id}")
    )


def test_ok(rep, metric):
    return bool(
        isinstance(rep, dict)
        and rep.get("fixed_split_test_evaluated")
        and (not rep.get("error"))
        and (not rep.get("post_hpo_test_error"))
        and _primary_metric_value_ok(rep.get("test_metrics"), metric)
    )


def select(trials, candidate, metric, cap, budget):
    """Failed attempts consume their original clock, ties favor the default."""
    cost = candidate["validation_seconds"]
    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError("Invalid default time charge")
    cutoff = max(0.0, budget - cost)
    sampled = select_budget_trial(trials, metric, cap, seconds=cutoff)
    source_finished = (
        candidate.get("source_completion_seconds")
        if "source_trial_id" in candidate
        else 0.0
    )
    default_ok = (
        candidate.get("status") == "success"
        and cost <= budget
        and completed_within_deadline(source_finished, cutoff)
        and _primary_metric_value_ok(candidate.get("validation_metrics"), metric)
    )
    use_default = False
    if default_ok:
        d = candidate["validation_metrics"][metric]
        maximize = metric.lower() in ("roc_auc", "auc", "accuracy", "r2", "f1")
        use_default = sampled is None or (
            d >= sampled["validation_metrics"][metric]
            if maximize
            else d <= sampled["validation_metrics"][metric]
        )
    selected = deepcopy(candidate if use_default else sampled)
    if selected is not None:
        selected["trial_id"] = -1 if use_default else selected["trial_id"]
        if use_default:
            selected.setdefault(
                "hpo_elapsed_seconds", candidate.get("source_completion_seconds") or 0.0
            )
        selected["test_metrics"] = {}
        selected.pop("error", None)
        selected["fixed_split_test_evaluated"] = False
    attempted = [
        t
        for t in trials
        if t["trial_id"] < cap
        and t.get("started_at_hpo_seconds", t["hpo_elapsed_seconds"]) < cutoff
    ]
    unsuccessful = [
        t["trial_id"]
        for t in attempted
        if t.get("status") != "success"
        or t["hpo_elapsed_seconds"] > cutoff
        or (not _primary_metric_value_ok(t.get("validation_metrics"), metric))
    ]
    return dict(
        protocol=PROTOCOL,
        default_selected=use_default,
        selected=selected,
        validation_seconds=cost,
        hpo_cutoff_seconds=cutoff,
        selection_budget_seconds=budget,
        attempted_configuration_cap=cap,
        retained_attempts=len(attempted),
        trial_completion=dict(
            n_completed_trials=len(attempted),
            n_successful_trials=len(attempted) - len(unsuccessful),
            unsuccessful_trial_ids=unsuccessful,
        ),
        default_candidate=deepcopy(candidate),
    )


def completed_evaluation(unit, report, selected):
    (metric, fold_id) = (unit["primary_metric"], unit["fold_id"])
    run = Path(unit["report_path"]).parent
    f = source_fold(report, fold_id)
    prefix = "" if fold_id is None else f"fold_{fold_id}_"
    tid = selected["trial_id"]
    if tid == -1:
        (rep, filename) = (f.get("default"), f"predictions_{prefix}default.pt")
        if test_ok(rep, metric) and (run / filename).is_file():
            return (deepcopy(rep), filename)
    else:
        rep = f.get("best")
        filename = (rep or {}).get("predictions") or f"predictions_{prefix}best.pt"
        if (
            test_ok(rep, metric)
            and rep.get("trial_id") == tid
            and (rep.get("parameters") == selected["parameters"])
            and (run / filename).is_file()
        ):
            return (deepcopy(rep), filename)
        suffix = "" if fold_id is None else f"_fold_{fold_id}"
        path = run / f"hpo_budget_results{suffix}.json"
        if path.exists():
            budgets = json.loads(path.read_text())
            if budgets.get("search_id") == unit["search_id"]:
                for entry in budgets.get("budgets", []):
                    if (
                        entry.get("status") == "success"
                        and entry.get("selected_trial_id") == tid
                        and (entry.get("parameters") == selected["parameters"])
                        and _primary_metric_value_ok(entry.get("test_metrics"), metric)
                        and (run / str(entry.get("predictions"))).is_file()
                    ):
                        rep = deepcopy(selected)
                        rep.update(
                            test_metrics=entry["test_metrics"],
                            fixed_split_test_evaluated=True,
                            post_hpo_train_wall_time_seconds=entry.get("refit_seconds"),
                            test_inference_wall_time_seconds=entry.get(
                                "test_inference_seconds"
                            ),
                            train_val_merged_for_fit=entry.get(
                                "train_val_merged_for_fit"
                            ),
                        )
                        return (rep, entry["predictions"])
    return (None, None)


def score_default(unit, runtime):
    from .cli_setup import _set_reproducibility
    from .hpo_step_timeout import run_train_with_exec_timeout
    from .inner_cv import train_inner_cv
    from .train_eval import train

    (cls, ds, args) = runtime
    budget = unit["budget_seconds"]
    params = unit["default_parameters"]
    cls.validate_parameters(params, args)
    _set_reproducibility(args.seed)
    inner = bool(getattr(ds, "hpo_inner_datasets", None))
    fn = train_inner_cv if inner else train
    kwargs = dict(
        model_class=cls,
        params=params,
        model_args=args,
        trial=SimpleNamespace(number=-1),
        dataset=ds,
        test_preds=[],
    )
    if not inner:
        kwargs.update(evaluate_test=False, merge_train_val_for_fit=False)
    started = time.monotonic()
    ds.hpo_step_deadline_monotonic = started + budget
    (output, error) = (None, None)
    try:
        result = run_train_with_exec_timeout(fn, kwargs, budget, raise_on_failure=True)
        output = result[0] if result is not None else None
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    import delu
    import gc

    delu.cuda.free_memory()
    gc.collect()
    elapsed = time.monotonic() - started
    metrics = output[0] if output else None
    if output and len(output) > 5 and (metrics is None):
        error = error or output[5].get("fit_error") or output[5].get("inner_cv_error")
    ok = elapsed <= budget and _primary_metric_value_ok(metrics, unit["primary_metric"])
    return dict(
        status="success" if ok else "timeout" if elapsed >= budget else "failed",
        parameters=params,
        validation_metrics=metrics if ok else {},
        error=error,
        validation_seconds=elapsed,
        inner_cv_metrics=output[5].get("inner_cv_metrics")
        if output and len(output) == 6
        else None,
        train_wall_time_seconds=output[2] if output else None,
        val_inference_wall_time_seconds=output[3] if output else None,
    )


def aggregate_collected_metrics(folds, name, field):
    from .train_eval import aggregate_cv_metrics

    metrics = []
    for fold in folds:
        block = fold.get(name) or {}
        values = block.get(field) or {}
        metrics.append(values if "error" not in block else {})
    if not metrics:
        return {}
    common = set.intersection(*(set(values) for values in metrics))
    complete = [
        key
        for key in metrics[0]
        if key in common
        and all(
            (
                isinstance(values[key], (int, float)) and math.isfinite(values[key])
                for values in metrics
            )
        )
    ]
    return aggregate_cv_metrics(
        [{key: values[key] for key in complete} for values in metrics]
    )


def update_report(report, is_cv):
    """Recompute official metrics and counts, never publish a partial CV test mean."""
    from .evaluation import _set_test_completion_status
    from .train_eval import (
        aggregate_cv_report_timing_fields,
        CV_TIMING_FIELDS_DEFAULT,
        CV_TIMING_FIELDS_BEST,
    )

    metric = report["primary_metric"]
    folds = report.get("cv_fold_metrics", []) if is_cv else [report]
    if is_cv:
        report["cv_aggregated_metrics"] = {
            name: {
                key: aggregate_collected_metrics(folds, name, field)
                for (key, field) in [
                    ("validation", "validation_metrics"),
                    ("test", "test_metrics"),
                ]
            }
            for name in ("default", "best")
        }
        for name in ("default", "best"):
            if len(folds) != 5 or not all(
                (test_ok(f.get(name), metric) for f in folds)
            ):
                report["cv_aggregated_metrics"][name]["test"] = {}
        report["cv_aggregated_timing"] = {
            "default": aggregate_cv_report_timing_fields(
                folds, "default", CV_TIMING_FIELDS_DEFAULT
            ),
            "best": aggregate_cv_report_timing_fields(
                folds, "best", CV_TIMING_FIELDS_BEST
            ),
        }
        report["cv_best_params_per_fold"] = [
            dict(
                fold_id=f["fold_id"],
                best_params=(f.get("best") or {}).get("parameters"),
            )
            for f in folds
        ]
        report["train_best_time"] = report["cv_aggregated_timing"]["best"].get(
            "train_wall_time_seconds"
        )
        report["reach_best_trial_seconds"] = report["cv_aggregated_timing"]["best"].get(
            "elapsed_seconds"
        )
        report["actual_configurations"] = sum(
            (f.get("actual_configurations", 0) for f in folds)
        )
        completion = dict(
            n_completed_trials=sum(
                (
                    (f.get("trial_completion") or {}).get("n_completed_trials", 0)
                    for f in folds
                )
            ),
            n_successful_trials=sum(
                (
                    (f.get("trial_completion") or {}).get("n_successful_trials", 0)
                    for f in folds
                )
            ),
        )
        report["trial_completion"] = completion
        report.update(completion)
    else:
        best = report.get("best") or {}
        report["train_best_time"] = best.get(
            "post_hpo_train_wall_time_seconds", best.get("train_wall_time_seconds")
        )
        report["reach_best_trial_seconds"] = best.get("elapsed_seconds")
    _set_test_completion_status(report, metric, 5 if is_cv else None)


def run_experiment(
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
    """Run validation selection and test evaluation under the benchmark budgets."""
    import datetime
    import functools
    from ..models import str2model
    from ..utils import dump_report
    from .artifacts import save_predictions, selected_study
    from .cli_setup import _set_reproducibility
    from .evaluation import (
        run_default_hyperparameter_baseline,
        _run_default_baseline_on_fixed_split,
        _run_sweep_with_hpo_step_timeout,
    )
    from .hpo_step_timeout import HPOStepDeadlineExceeded
    from .hpo_trajectory import (
        evaluate_hpo_budgets,
        save_test_labels,
        budget_results_complete,
    )
    from .inner_cv import prepare_inner_cv, train_inner_cv
    from .sweep import OptunaSweep
    from .train_eval import (
        train,
        get_fold_model_args,
        apply_fixed_split_test_eval_for_best_trial,
    )
    from .utils.parameter_config import merge_default_params_with_search_space

    cls = str2model(args.model)
    metric = config["optimize_optuna_with"][args.objective]
    is_cv = metadata.get("split_strategy") == "cv"
    report = deepcopy(report_base)
    report.update(
        hpo_enabled=True,
        primary_metric=metric,
        selection=dict(
            protocol=PROTOCOL,
            execution="validation",
            default_counts_toward_attempt_cap=False,
        ),
    )
    if is_cv:
        report.update(
            cv_hpo_protocol="nested_inner_cv_v1",
            cv_inner_folds_requested=3,
            cv_outer_folds=5,
            n_outer_folds=5,
            cv_time_limit_per_fold=config["cv_time_limit_per_fold"],
        )
    saved = (
        {f["fold_id"]: f for f in report.get("cv_fold_metrics", [])}
        if is_cv
        else {None: report}
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
    defaults = cls.resolve_parameters(
        merge_default_params_with_search_space(
            cls.get_default_params(), args.model, config["search_space_dir"]
        ),
        model_args,
    )

    def persist(f, fold_id):
        if is_cv:
            saved[fold_id] = f
            report["cv_fold_metrics"] = [saved[i] for i in sorted(saved)]
        update_report(report, is_cv)
        dump_report(report, experiment_dir)

    outer = (
        ((i, ds) for i, (ds, _, _) in enumerate(dataset.get_cv_splits()))
        if is_cv
        else [(None, dataset)]
    )
    for fold_id, ds in outer:
        if fold_id is not None and fold_id >= 5:
            break
        f = saved.get(fold_id, {})
        if (
            test_ok(f.get("best"), metric)
            and f.get("selection")
            and test_ok(f.get("default"), metric)
            and budget_results_complete(
                experiment_dir, config.get("hpo_evaluation_budgets", []), fold_id
            )
        ):
            continue
        if is_cv:
            plan = prepare_inner_cv(
                dataset,
                ds,
                cls,
                requested_folds=3,
                seed=args.seed,
                outer_fold_id=fold_id,
                mv_strategy=mv,
                cat_encoding_strategy=cat,
            )
            if f.get("hpo_validation") and f["hpo_validation"] != plan:
                raise ValueError("Inner-validation plan differs from the recorded plan")
            f.update(fold_id=fold_id, hpo_validation=plan)
        baseline = f.get("default")
        if not is_cv or test_ok(baseline, metric):
            ds.prepare_dataset(
                inpute_mv=cls.needs_missing_value_imputation(),
                encode_cat=cls.needs_cat_encoding(),
                scale_features=False,
                mv_strategy=mv,
                cat_encoding_strategy=cat,
            )
        fold_args = get_fold_model_args(model_args, ds)
        prefix = "" if fold_id is None else f"fold_{fold_id}_"
        if not test_ok(baseline, metric):
            _set_reproducibility(args.seed)
            kw = dict(
                model_class=cls,
                model_args=fold_args,
                dataset=ds,
                experiment_start=datetime.datetime.now(),
                model_name=args.model,
                search_space_dir=config["search_space_dir"],
            )
            if is_cv:
                (baseline, preds) = _run_default_baseline_on_fixed_split(
                    fold_id=fold_id,
                    cat_encoding_strategy=cat,
                    mv_imputation_strategy=mv,
                    **kw,
                )
            else:
                (baseline, preds) = run_default_hyperparameter_baseline(
                    metadata=metadata, **kw
                )
            if test_ok(baseline, metric):
                save_predictions(experiment_dir, prefix + "default", preds)
                save_test_labels(experiment_dir, ds, fold_id)
        f["default"] = baseline
        fold_args = get_fold_model_args(model_args, ds)
        persist(f, fold_id)
        budget = config["cv_time_limit_per_fold"] if is_cv else config["time_limit"]
        stage = stage_directory(experiment_dir, fold_id)
        stage.mkdir(parents=True, exist_ok=True)
        uid = hashlib.sha256(
            json.dumps(
                dict(identity=report_base["input_identity"], fold=fold_id),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cp = stage / "reference.json"
        c = json.loads(cp.read_text()) if cp.exists() else None
        if c and c.get("unit_identity") != uid:
            raise ValueError("Default evaluation belongs to a different run identity")
        unit = dict(
            model=args.model,
            dataset=args.dataset,
            fold_id=fold_id,
            primary_metric=metric,
            budget_seconds=budget,
            default_parameters=defaults,
            report_path=str(Path(experiment_dir) / "report.json"),
        )
        runtime = (cls, ds, fold_args)
        if c is None and args.model not in ("HyperFast", "iLTM"):
            if (
                not is_cv
                and baseline
                and _primary_metric_value_ok(baseline.get("validation_metrics"), metric)
            ):
                c = dict(
                    status="success",
                    parameters=defaults,
                    validation_metrics=baseline["validation_metrics"],
                    validation_seconds=baseline["elapsed_seconds"]
                    - baseline["test_inference_wall_time_seconds"],
                )
            else:
                c = score_default(unit, runtime)
            c["unit_identity"] = uid
            save(cp, c)
        remaining = max(0, math.floor(budget - (c["validation_seconds"] if c else 0)))
        if not f.get("hpo_finished"):
            sweep_config = deepcopy(config)
            sweep_config["time_limit"] = remaining
            sweep_config["selection_budget_seconds"] = budget
            sweep_config["validation_charge_seconds"] = (
                c["validation_seconds"] if c else 0.0
            )
            fn = (
                train_inner_cv
                if getattr(ds, "hpo_inner_datasets", None)
                else functools.partial(train, evaluate_test=False)
            )

            def progress(study, records, *unused):
                from .hpo_trajectory import official_hpo_result

                (view, records) = official_hpo_result(
                    study, records, config["n_trials"]
                )
                f["actual_configurations"] = len(view.trials)
                target_trials = config.get("trajectory_n_trials", config["n_trials"])
                f["hpo_finished"] = (
                    isinstance(target_trials, int)
                    and len(study.trials) >= target_trials
                )
                persist(f, fold_id)

            _set_reproducibility(args.seed)
            sweep = OptunaSweep(
                cls,
                ds,
                fold_args,
                metric,
                sweep_config,
                fn,
                args.model,
                seed=args.seed,
                experiment_dir=experiment_dir,
                report_base={**report_base, "cv_fold_id": fold_id},
                save_callback=progress,
                optuna_handler=optuna_handler,
            )
            sweep.validation_candidate = c
            sweep.validation_parameters = defaults
            try:
                _run_sweep_with_hpo_step_timeout(sweep, remaining)
            except HPOStepDeadlineExceeded:
                pass
            f["hpo_finished"] = True
            persist(f, fold_id)
        ts = load_trials(experiment_dir, fold_id)
        if c is None:
            t = next((t for t in ts if t["trial_id"] == 0), {})
            effective = {**defaults, **t.get("parameters", {})}
            expected = dict(defaults)
            if args.model == "HyperFast":
                for p in (effective, expected):
                    if p.get("finetuning_batch_size") is None:
                        p["finetuning_batch_size"] = p["batch_size"]
            if effective != expected:
                raise ValueError("The first HPO trial no longer matches the default")
            c = dict(
                status=t.get("status", "failed"),
                parameters=defaults,
                validation_metrics=t.get("validation_metrics", {}),
                inner_cv_metrics=t.get("inner_cv_metrics"),
                validation_seconds=0.0,
                source_trial_id=0,
                source_completion_seconds=t.get("hpo_elapsed_seconds"),
                unit_identity=uid,
            )
            save(cp, c)
        result = select(ts, c, metric, config["n_trials"], budget)
        selected = result["selected"]
        unit.update(identity=uid, search_id=ts[0]["search_id"] if ts else None)
        if selected is not None:
            (cached, filename) = completed_evaluation(unit, report, selected)
            if cached is None:
                _set_reproducibility(args.seed)
                preds = []
                apply_fixed_split_test_eval_for_best_trial(
                    selected_study(selected), [selected], ds, cls, fold_args, preds
                )
                filename = None
                if test_ok(selected, metric):
                    save_predictions(stage, "selected", preds)
                    save_test_labels(experiment_dir, ds, fold_id)
                    filename = str(
                        (stage / "predictions_selected.pt").relative_to(experiment_dir)
                    )
                evaluation = selected
            else:
                evaluation = {
                    **selected,
                    **{
                        k: v
                        for (k, v) in cached.items()
                        if k
                        not in (
                            "validation_metrics",
                            "inner_cv_metrics",
                            "trial_id",
                            "parameters",
                        )
                    },
                }
            evaluation["predictions"] = filename
            evaluation["elapsed_seconds"] = c["validation_seconds"] + (
                selected.get("hpo_elapsed_seconds", 0)
                if not result["default_selected"]
                else c.get("source_completion_seconds") or 0
            )
        else:
            evaluation = None
        f.update(
            best=evaluation,
            selection=dict(
                protocol=PROTOCOL,
                execution="validation",
                default_selected=result["default_selected"],
                validation_seconds=c["validation_seconds"],
                hpo_cutoff_seconds=result["hpo_cutoff_seconds"],
            ),
            actual_configurations=result["retained_attempts"],
            trial_completion=result["trial_completion"],
        )
        f.update(result["trial_completion"])
        result.update(evaluation=evaluation, unit_identity=uid)
        save(stage / "result.json", result)
        persist(f, fold_id)
        ds.selection_context = dict(candidate=c, budget=budget, baseline=baseline)
        evaluate_hpo_budgets(
            experiment_dir,
            config.get("hpo_evaluation_budgets", []),
            metric,
            cls,
            fold_args,
            ds,
            best_rep=evaluation,
            fold_id=fold_id,
        )
    update_report(report, is_cv)
    dump_report(report, experiment_dir)


def source_fold(report, fold_id):
    if fold_id is None:
        return report
    return next((f for f in report["cv_fold_metrics"] if f["fold_id"] == fold_id))
