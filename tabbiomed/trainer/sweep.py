from pathlib import Path
import datetime
import gc
import logging
import time
from copy import deepcopy
import delu
import optuna
from loguru import logger
from tabbiomed.models import get_recommended_model_configs
from tabbiomed.trainer.hpo_step_timeout import (
    HPOStepDeadlineExceeded,
    TrialProcessError,
    hpo_step_seconds_remaining,
    set_hpo_step_deadline,
    run_train_with_exec_timeout,
    trial_ref_for_subprocess,
)
from tabbiomed.trainer.utils.parameter_config import (
    load_model_search_space,
    define_trial_parameters_from_config,
    merge_default_params_with_search_space,
    encode_weighted_categorical_parameters,
)
from tabbiomed.trainer.train_eval import _primary_metric_value_ok
from tabbiomed.trainer.hpo_trajectory import HPOHistory, official_hpo_result


def _is_lightgbm_model(model_name: str) -> bool:
    normalized = model_name.replace("-", "").replace("_", "").lower()
    return normalized in ("lightgbm", "lgbm")


def _hyperfast_invalid_optimize_pair(params: dict) -> bool:
    """True if optimization='optimize' with n_ensemble != 1 (invalid for HyperFast)."""
    if params.get("optimization") != "optimize":
        return False
    n_ens = params.get("n_ensemble")
    try:
        return int(n_ens) != 1
    except (TypeError, ValueError):
        return True


class OptunaSweep:
    def __init__(
        self,
        model_class,
        dataset,
        model_args,
        metric,
        config,
        train_func,
        model_name,
        seed=42,
        experiment_dir=None,
        report_base=None,
        save_callback=None,
        optuna_handler=None,
        study_start_time: datetime.datetime | None = None,
        initial_test_preds: list | None = None,
    ):
        self.model_class = model_class
        self.dataset = dataset
        self.model_args = model_args
        self.metric = metric
        self.config = config
        self.recommended_configs = get_recommended_model_configs(model_name)
        self.seed = seed
        self.model_name = model_name
        self.trial_reports = []
        self.test_preds = list(initial_test_preds) if initial_test_preds else []
        self.train_func = train_func
        self.start_time = study_start_time or datetime.datetime.now()
        self.experiment_dir = experiment_dir
        self.report_base = report_base
        self.save_callback = save_callback
        self.study = None
        self.optuna_handler = optuna_handler
        self.hpo_step_deadline_monotonic: float | None = None
        self.history = None
        self._active_parameters = {}

    def _end_trial_cleanup(self) -> None:
        """Release GPU memory after each trial; force GC for LightGBM (native heap pressure)."""
        delu.cuda.free_memory()
        if _is_lightgbm_model(self.model_name):
            gc.collect()

    def _stop_study_for_hpo_deadline(self) -> None:
        if self.study is None:
            return
        try:
            self.study.stop()
        except Exception as exc:
            logger.warning(f"Failed to stop Optuna study after HPO deadline: {exc}")

    def _run_trial_train(self, trial, params):
        remaining = hpo_step_seconds_remaining(self)
        time_limit = self.config.get("time_limit")
        if remaining is None and time_limit is not None:
            if float(time_limit) <= 0:
                self._stop_study_for_hpo_deadline()
                raise HPOStepDeadlineExceeded("Selection budget exhausted before trial fit")
            set_hpo_step_deadline(self, time_limit)
            remaining = hpo_step_seconds_remaining(self)
        if remaining is not None and remaining <= 0:
            self._stop_study_for_hpo_deadline()
            raise HPOStepDeadlineExceeded("Selection budget exhausted before trial fit")
        if getattr(self.dataset, "hpo_inner_datasets", None):
            self.dataset.hpo_step_deadline_monotonic = self.hpo_step_deadline_monotonic
        kwargs = dict(
            model_class=self.model_class,
            params=params,
            model_args=self.model_args,
            trial=trial_ref_for_subprocess(trial),
            dataset=self.dataset,
            test_preds=[],
        )
        self._end_trial_cleanup()
        result = run_train_with_exec_timeout(
            self.train_func,
            kwargs,
            remaining,
            raise_on_failure=True,
            propagate_rng_state=True,
        )
        if result is None:
            return None
        output, predictions = result
        self.test_preds.extend(predictions)
        return output

    def objective(self, trial: optuna.trial.Trial):
        self._active_parameters = {}
        started = time.monotonic()
        if self.history is not None:
            self.history.write(
                "trial_started", trial_id=trial.number, attempt=trial.number + 1
            )
        (status, error) = ("success", None)
        try:
            return self._objective(trial)
        except optuna.TrialPruned as exc:
            (status, error) = ("pruned", str(exc) or "Trial pruned")
            raise
        except HPOStepDeadlineExceeded as exc:
            (status, error) = ("timeout", str(exc))
            raise
        except BaseException as exc:
            status = "failed" if isinstance(exc, Exception) else "interrupted"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self.history is not None:
                rep = next(
                    (
                        r
                        for r in reversed(self.trial_reports)
                        if r.get("trial_id") == trial.number
                    ),
                    {},
                )
                if status == "success" and (
                    not _primary_metric_value_ok(
                        rep.get("validation_metrics"), self.metric
                    )
                ):
                    status = "failed"
                self.history.write(
                    "trial_finished",
                    trial_id=trial.number,
                    attempt=trial.number + 1,
                    parameters=self._active_parameters,
                    status=status,
                    error=rep.get("error", error),
                    validation_metrics=rep.get("validation_metrics", {}),
                    inner_cv_metrics=rep.get("inner_cv_metrics"),
                    trial_wall_time_seconds=time.monotonic() - started,
                    train_wall_time_seconds=rep.get("train_wall_time_seconds"),
                    val_inference_wall_time_seconds=rep.get(
                        "val_inference_wall_time_seconds"
                    ),
                )

    def _objective(self, trial: optuna.trial.Trial):
        if self.optuna_handler is not None:
            optuna_logger = logging.getLogger("optuna")
            if self.optuna_handler not in optuna_logger.handlers:
                optuna_logger.addHandler(self.optuna_handler)
                optuna_logger.setLevel(logging.INFO)
                optuna_logger.propagate = False
        dataset_metadata = {
            "objective": self.model_args.objective,
            "use_gpu": self.model_args.use_gpu,
        }
        if self.model_name.lower() == "knn":
            inner = getattr(self.dataset, "hpo_inner_datasets", None)
            dataset_metadata["n_train_samples"] = (
                min((len(f.X_train) for f in inner))
                if inner
                else self.dataset.X_train.shape[0]
            )
        search_space_dir = self.config.get(
            "search_space_dir",
            str(Path(__file__).resolve().parents[1] / "configs/model_search_space"),
        )
        expected_space = self.config.get("expected_search_space")
        if (
            expected_space is not None
            and load_model_search_space(self.model_name, config_dir=search_space_dir)
            != expected_space
        ):
            raise RuntimeError(
                "Pinned HPO search space changed during the selection job"
            )
        if self.recommended_configs is not None:
            params = deepcopy(self.recommended_configs[trial.number])
            params.pop("device", None)
            trial.set_user_attr("recommended_config_index", trial.number)
            trial.set_user_attr("sampled_parameters", deepcopy(params))
        else:
            params = define_trial_parameters_from_config(
                trial,
                self.model_name,
                config_dir=search_space_dir,
                dataset_metadata=dataset_metadata,
            )
        params = self.model_class.resolve_parameters(params, self.model_args)
        self._active_parameters = deepcopy(params)
        if self.history is not None:
            self.history.write(
                "trial_parameters", trial_id=trial.number, parameters=params
            )
        logger.info(f"Trial {trial.number} - Parameters: {params}")
        if self.model_name.replace("-", "").replace("_", "").lower() == "hyperfast":
            if _hyperfast_invalid_optimize_pair(params):
                logger.info(
                    f"Trial {trial.number} - Pruned: HyperFast requires n_ensemble=1 when optimization='optimize' (got n_ensemble={params.get('n_ensemble')})."
                )
                self.trial_reports.append(
                    {
                        "error": "Pruned: invalid HyperFast (optimization='optimize' needs n_ensemble=1)",
                        "trial_id": trial.number,
                        "parameters": params,
                    }
                )
                raise optuna.TrialPruned()
        remaining_before = hpo_step_seconds_remaining(self)
        if remaining_before is not None and remaining_before <= 0:
            self._stop_study_for_hpo_deadline()
            raise HPOStepDeadlineExceeded(
                "HPO step wall clock exhausted before starting this trial."
            )
        try:
            train_out = self._run_trial_train(trial, params)
        except TrialProcessError as exc:
            self.trial_reports.append(
                dict(error=str(exc), trial_id=trial.number, parameters=params)
            )
            self._end_trial_cleanup()
            raise optuna.TrialPruned(str(exc)) from exc
        if train_out is None:
            if (
                hpo_step_seconds_remaining(self) is not None
                and hpo_step_seconds_remaining(self) <= 0
            ):
                self.trial_reports.append(
                    {
                        "error": "Trial training exceeded remaining HPO time budget",
                        "trial_id": trial.number,
                        "parameters": params,
                    }
                )
                self._stop_study_for_hpo_deadline()
                raise HPOStepDeadlineExceeded(
                    "HPO step wall clock exhausted during trial training."
                )
            self.trial_reports.append(
                {
                    "error": "Trial training exceeded remaining HPO time budget or child failed",
                    "trial_id": trial.number,
                    "parameters": params,
                }
            )
            self._end_trial_cleanup()
            raise optuna.TrialPruned()
        details = train_out[5] if len(train_out) == 6 else {}
        (
            val_metrics,
            test_metrics,
            train_wall_time_seconds,
            val_inference_wall_time_seconds,
            test_inference_wall_time_seconds,
        ) = train_out[:5]
        if val_metrics is None:
            self.trial_reports.append(
                {
                    "error": details.get(
                        "inner_cv_error",
                        details.get("fit_error", "Error during training"),
                    ),
                    "trial_id": trial.number,
                    "parameters": params,
                    **details,
                }
            )
            self._end_trial_cleanup()
            raise optuna.TrialPruned()
        if not _primary_metric_value_ok(val_metrics, self.metric):
            available = list(val_metrics.keys()) if val_metrics else []
            msg = f"Pruned: primary metric '{self.metric}' missing or non-finite in validation metrics (available: {available})"
            logger.info(f"Trial {trial.number} - {msg}")
            self.trial_reports.append(
                {
                    "error": msg,
                    "trial_id": trial.number,
                    "parameters": params,
                    "validation_metrics": val_metrics,
                    **details,
                }
            )
            self._end_trial_cleanup()
            raise optuna.TrialPruned()
        metric = self.metric
        score = val_metrics[metric]
        logger.info(f"Trial {trial.number} - Validation {metric}: {score:.4f}")
        elapsed_seconds = (datetime.datetime.now() - self.start_time).total_seconds()
        report = {
            "parameters": params,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "trial_id": trial.number,
            "tuning_time": str(datetime.datetime.now() - self.start_time),
            "train_wall_time_seconds": train_wall_time_seconds,
            "val_inference_wall_time_seconds": val_inference_wall_time_seconds,
            "test_inference_wall_time_seconds": test_inference_wall_time_seconds,
            "elapsed_seconds": elapsed_seconds,
            **details,
        }
        self.trial_reports.append(report)
        self._end_trial_cleanup()
        return score

    def run(self):
        from ..models import NO_HPO_MODELS, model_name_map

        if model_name_map(self.model_name) in NO_HPO_MODELS:
            return (None, [], self.test_preds)
        official_cap = self.config["n_trials"]
        collection_cap = self.config.get("trajectory_n_trials", official_cap)
        if "trajectory_n_trials" in self.config and (
            type(official_cap) is not int
            or type(collection_cap) is not int
            or official_cap < 1
            or (collection_cap < official_cap)
        ):
            raise ValueError(
                "trajectory_n_trials must be an integer >= the positive official n_trials cap"
            )
        if self.experiment_dir is not None:
            self.history = HPOHistory(
                self.experiment_dir,
                self.model_name,
                getattr(
                    self.model_args, "dataset", getattr(self.dataset, "name", None)
                ),
                (self.report_base or {}).get("cv_fold_id"),
                self.seed,
                self.metric,
                origin=getattr(self, "hpo_step_started_monotonic", None),
            )
            self.history.write(
                "search_limits",
                official_attempt_cap=official_cap,
                collection_attempt_cap=collection_cap,
                time_limit=self.config["time_limit"],
                trial_process_isolation=True,
                validation_protocol=getattr(self.dataset, "hpo_validation", None),
            )
            if "validation_charge_seconds" in self.config:
                self.history.write(
                    "selection_budget",
                    validation_seconds=self.config["validation_charge_seconds"],
                    selection_budget_seconds=self.config["selection_budget_seconds"],
                )
        direction = metric_to_direction(self.metric)
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.RandomSampler(seed=self.seed),
            pruner=None,
        )
        if (
            self.model_name.replace("-", "").replace("_", "").lower() == "hyperfast"
            and (collection_cap is None or collection_cap > 0)
            and (
                self.config.get("time_limit") is None
                or self.config["time_limit"] > 0
            )
        ):
            defaults = merge_default_params_with_search_space(
                self.model_class.get_default_params(),
                self.model_name,
                config_dir=self.config.get(
                    "search_space_dir",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "configs/model_search_space"
                    ),
                ),
            )
            if defaults.get("finetuning_batch_size") is None:
                defaults["finetuning_batch_size"] = defaults["batch_size"]
            space = load_model_search_space(
                self.model_name,
                config_dir=self.config.get(
                    "search_space_dir",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "configs/model_search_space"
                    ),
                ),
            )
            self.study.enqueue_trial(
                encode_weighted_categorical_parameters(defaults, space)
            )
        n_trials = collection_cap if isinstance(collection_cap, int) else None
        if self.recommended_configs is not None:
            count = len(self.recommended_configs)
            n_trials = count if n_trials is None else min(n_trials, count)
            logger.info(
                "Using {} recommended model configurations in supplied order, trial cap: {}",
                count,
                n_trials,
            )

        def _after_trial_callback(study, frozen_trial):
            """Run after Optuna finishes a trial and updates best_value / best_trial."""
            boundary = (
                isinstance(official_cap, int)
                and len(study.trials) == official_cap
                and isinstance(collection_cap, int)
                and (collection_cap > official_cap)
            )
            if boundary and self.history is not None:
                (official_study, records) = official_hpo_result(
                    study, self.trial_reports, official_cap
                )
                try:
                    tid = official_study.best_trial.number
                except ValueError:
                    tid = None
                selected = next((r for r in records if r["trial_id"] == tid), None)
                self.history.write(
                    "official_selection_frozen",
                    official_attempt_cap=official_cap,
                    selected_trial_id=tid,
                    selection=selected,
                )
                if "validation_charge_seconds" in self.config:
                    from .selection import load_trials, select

                    saved = load_trials(
                        self.experiment_dir, (self.report_base or {}).get("cv_fold_id")
                    )
                    candidate = self.validation_candidate
                    if candidate is None:
                        first = next((r for r in saved if r["trial_id"] == 0), {})
                        candidate = dict(
                            status=first.get("status", "failed"),
                            parameters=self.validation_parameters,
                            validation_metrics=first.get("validation_metrics", {}),
                            validation_seconds=0.0,
                            source_trial_id=0,
                            source_completion_seconds=first.get("hpo_elapsed_seconds"),
                        )
                    frozen = select(
                        saved,
                        candidate,
                        self.metric,
                        official_cap,
                        self.config["selection_budget_seconds"],
                    )
                    self.history.write(
                        "official_selection_frozen",
                        official_attempt_cap=official_cap,
                        selection=frozen,
                    )
            if self.save_callback is None:
                return
            try:
                self.save_callback(
                    study, self.trial_reports, self.experiment_dir, self.report_base
                )
            except Exception as e:
                if boundary:
                    raise
                logger.warning(
                    f"Failed to save incremental report after trial {frozen_trial.number}: {e}"
                )

        try:
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=self.config["time_limit"],
                callbacks=[_after_trial_callback],
            )
        except HPOStepDeadlineExceeded:
            logger.warning(
                f"Optuna study stopped: HPO step wall clock reached ({self.config.get('time_limit')}s budget)."
            )
        finally:
            if self.history is not None:
                count = len(self.study.trials)
                remaining = hpo_step_seconds_remaining(self)
                if remaining is not None and remaining <= 0:
                    reason = "time_budget"
                elif self.recommended_configs is not None and count >= len(
                    self.recommended_configs
                ):
                    reason = "search_space_exhausted"
                elif n_trials is not None and count >= n_trials:
                    reason = "configuration_cap"
                else:
                    reason = (
                        "time_budget"
                        if time.monotonic() - self.history.origin
                        >= self.config["time_limit"]
                        else "interrupted"
                    )
                self.history.write(
                    "search_finished",
                    actual_attempts=count,
                    stop_reason=reason,
                    official_attempt_cap=official_cap,
                    collection_attempt_cap=collection_cap,
                    time_limit=self.config["time_limit"],
                )
        return (self.study, self.trial_reports, self.test_preds)


def metric_to_direction(metric: str) -> str:
    if metric.lower() in ["accuracy", "auc", "roc_auc", "f1", "r2"]:
        return "maximize"
    if metric.lower() in ["log_loss", "rmse", "mae", "mse"]:
        return "minimize"
    raise ValueError(
        f"Unknown metric '{metric}' for determining optimization direction."
    )
