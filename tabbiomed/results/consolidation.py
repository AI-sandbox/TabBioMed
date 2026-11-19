"""Manifest-driven publication of benchmark evaluations."""

from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import shutil
import tempfile

from .io import copy_verified, digest, local_file, read_json, write_json
from .predictions import validate_predictions
from .schema import (
    CONFIG_KEYS,
    UNIT_KEYS,
    clean,
    compact_failure,
    pick,
    public_block,
    summarize,
)
from .timing import (
    evaluation_timing,
    fitted_rows,
    load_history,
    seconds,
    selection_timing,
    sum_known,
    units,
)

from .sources import (
    Evidence,
    safe_name,
    registered_name,
    source_identity,
    same_inputs,
    dataset_splits,
    validate_labels,
)
from .selection import (
    test_available,
    failure_record,
    inferred_failure,
    read_selection,
    verify_selection,
    aggregate,
)


REPORT_KEYS = (
    "model",
    "dataset",
    "seed",
    "gpu",
    "primary_metric",
    "hpo_enabled",
    "cv_hpo_protocol",
    "cv_inner_folds_requested",
)
ENTRY_KEYS = {
    "report",
    "model",
    "dataset",
    "scope",
    "data_root",
    "failures",
    "fold_failures",
    "search_failure",
    "hardware",
}


@dataclass
class Evaluation:
    destination: Path
    report: dict
    timing: dict
    artifacts: dict
    evidence: Evidence
    identity: dict
    metadata: dict
    source_path: Path | None = None


def prepare_evaluation(entry, base):
    if set(entry) - ENTRY_KEYS:
        raise ValueError(
            f"Unknown evaluation fields: {sorted(set(entry) - ENTRY_KEYS)}"
        )
    evidence = Evidence()
    report_path = (base / entry["report"]).resolve()
    source = evidence.read(report_path)
    identity = source_identity(source)
    model, dataset = registered_name(source["model"]), safe_name(source["dataset"])
    if entry.get("model", model) != model or entry.get("dataset", dataset) != dataset:
        raise ValueError("Manifest model/dataset differs from its source report")
    if source.get("status") not in ("complete", "incomplete", "failed"):
        raise ValueError("Only settled reports can be consolidated")
    if type(source.get("hpo_enabled")) is not bool or not source.get("primary_metric"):
        raise ValueError("Source must record the HPO policy and primary metric")
    hpo, metric, config = (
        source["hpo_enabled"],
        source["primary_metric"],
        source.get("config") or {},
    )
    if "search_space" not in identity:
        raise ValueError("Source must record the search space actually used")
    if hpo and (
        type(config.get("n_trials")) is not int
        or config["n_trials"] <= 0
        or seconds(config.get("time_limit")) is None
    ):
        raise ValueError(
            "Tunable models must record their attempted-configuration cap and budget"
        )
    variants = ["default", "best"] if hpo else ["default"]
    scope = safe_name(entry.get("scope", "main"))
    destination = Path(model) / dataset
    if scope != "main":
        destination /= scope
    data_root = (base / entry.get("data_root", identity.get("data_root", ""))).resolve()
    splits, cv, split_sha256 = dataset_splits(identity, data_root, evidence)
    metadata = evidence.read(data_root / "metadata.json")
    su = units(source, len(splits))
    expected = set(range(len(splits))) if cv else {None}
    if set(su) != expected:
        raise ValueError("Source folds differ from the stored split layout")
    if set(entry.get("fold_failures", {})) - {str(fold) for fold in expected}:
        raise ValueError("Failure annotation refers to an unknown outer fold")
    if cv and (type(config.get("cv_time_limit_per_fold")) not in (int, float) and hpo):
        raise ValueError("CV selection needs its recorded per-fold budget")
    public = pick(source, REPORT_KEYS)
    public.update(
        model=model,
        search_space=clean(identity["search_space"]),
        config={**pick(config, CONFIG_KEYS), "default_only": not hpo},
        status="complete",
        completion_errors=[],
    )
    if entry.get("hardware"):
        public["hardware"] = pick(entry["hardware"], ("name", "total_memory"))
    history = report_path.parent / "hpo_trials.jsonl"
    if hpo and evidence.optional(history):
        events, history_sha, _ = load_history(
            local_file(report_path.parent, history.name)
        )
        evidence.pin(history, history_sha)
        for event in events:
            for key in ("model", "dataset", "seed", "primary_metric"):
                if key in event and event[key] != source[key]:
                    raise ValueError(f"HPO history belongs to a different {key}")
    else:
        events = []
    artifacts, exported, timing_units = {}, [], []
    for fold, source_unit in su.items():
        counts = {
            name: len(splits[fold or 0][key])
            for name, key in (
                ("train", "train"),
                ("validation", "val"),
                ("test", "test"),
            )
        }
        value = {"fold_id": fold} if cv else {}
        timing_unit = dict(
            fold_id=fold,
            split_samples=counts,
            outer_training_pool_samples=counts["train"] + counts["validation"],
        )
        failures = {
            **entry.get("failures", {}),
            **entry.get("fold_failures", {}).get(str(fold), {}),
        }
        if set(failures) - set(variants):
            raise ValueError("Failure annotation refers to an excluded variant")
        for variant in variants:
            raw = source_unit.get(variant)
            block = public_block(raw)
            available = test_available(raw, metric)
            if available:
                if variant in failures:
                    raise ValueError(
                        "Cannot annotate a successful test evaluation as failed"
                    )
                prefix = "" if fold is None else f"fold_{fold}_"
                filename = f"predictions_{prefix}{variant}.pt"
                pointer = raw.get("predictions") or filename
                if (
                    variant == "best"
                    and not raw.get("predictions")
                    and raw.get("trial_id") == -1
                ):
                    pointer = f"predictions_{prefix}default.pt"
                run = report_path.parent
                artifacts[filename] = local_file(run, pointer)
                evidence.pin(artifacts[filename])
                block["predictions"] = filename
            else:
                block = block or {}
                block.update(
                    test_metrics={},
                    predictions=None,
                    fixed_split_test_evaluated=False,
                    failure=failure_record(failures[variant])
                    if variant in failures
                    else inferred_failure(raw),
                )
                prefix = "" if fold is None else f"Fold {fold}: "
                public["completion_errors"].append(
                    f"{prefix}{variant} has no successful test evaluation for {metric}"
                )
            value[variant] = block
            clock = evaluation_timing(raw, metric, tuned=(variant == "best"))
            clock["evaluation_status"] = "success" if available else "failed"
            clock["final_fit_samples"] = fitted_rows(raw, counts)
            if variant == "default":
                clock["training_cost_seconds"] = clock[
                    "final_fit_and_validation_seconds"
                ]
            timing_unit["default" if variant == "default" else "tuned"] = clock
        label = f"test_labels_{'fixed' if fold is None else 'fold_' + str(fold)}.npy"
        label_path = evidence.optional(report_path.parent / label)
        if label_path is not None:
            label_path = local_file(report_path.parent, label)
            evidence.pin(label_path)
            validate_labels(
                label_path,
                data_root / "y.npy",
                splits[fold or 0]["test"],
                target_classes=identity.get("target_classes"),
            )
            artifacts[label] = label_path
            for variant in variants:
                block = value[variant]
                if test_available(block, metric):
                    validate_predictions(
                        artifacts[block["predictions"]],
                        label_path,
                        metadata,
                        block["test_metrics"],
                        metric,
                    )
        elif any(test_available(value.get(name), metric) for name in variants):
            raise ValueError("A successful test evaluation is missing test labels")
        if hpo:
            candidate, selection = read_selection(report_path.parent, fold, evidence)
            clock, completion = verify_selection(
                source_unit, candidate, selection, events, fold, metric, config
            )
            if not clock:
                clock, _ = selection_timing(
                    events, fold, {}, config, selection, candidate
                )
            if selection:
                value["selection"] = selection
            value.update(pick(source_unit, UNIT_KEYS))
            value.update(completion)
            if entry.get("search_failure"):
                value["tuning"] = {"failure": failure_record(entry["search_failure"])}
            elif (source_unit.get("tuning") or {}).get("failure"):
                value["tuning"] = {
                    "failure": failure_record(
                        compact_failure(source_unit["tuning"]["failure"])
                    )
                }
            timing_unit["hpo"] = clock
            if "best" in variants:
                timing_unit["tuned"]["training_cost_seconds"] = sum_known(
                    clock["selection_wall_seconds"],
                    timing_unit["tuned"]["final_fit_and_validation_seconds"],
                )
        exported.append(value)
        timing_units.append(timing_unit)
    if cv:
        public.update(n_outer_folds=len(splits), cv_fold_metrics=exported)
        public["cv_aggregated_metrics"] = {
            variant: {
                name: aggregate(exported, variant, field)
                if name != "test"
                or all(test_available(unit.get(variant), metric) for unit in exported)
                else {}
                for name, field in (
                    ("test", "test_metrics"),
                    ("validation", "validation_metrics"),
                )
            }
            for variant in variants
        }
        public["cv_best_params_per_fold"] = (
            [
                dict(
                    fold_id=unit["fold_id"],
                    best_params=(unit.get("best") or {}).get("parameters"),
                )
                for unit in exported
            ]
            if hpo
            else []
        )
    else:
        public.update(exported[0])
    if public["completion_errors"]:
        public["status"] = "incomplete"
    public = summarize(public)
    timing = dict(
        artifact_type="evaluation_timing_summary",
        schema_version=1,
        model=model,
        dataset=dataset,
        scope=scope,
        hpo_enabled=hpo,
        dataset_split_sha256=split_sha256,
        units=timing_units,
    )
    return Evaluation(
        destination=destination,
        report=public,
        timing=clean(timing),
        artifacts=artifacts,
        evidence=evidence,
        identity=identity,
        metadata=metadata,
        source_path=report_path,
    )


def prepare_manifest(manifest, locks):
    manifest = Path(manifest).resolve()
    document, sha = read_json(manifest)
    if (
        set(document) != {"schema_version", "evaluations"}
        or document["schema_version"] != 1
        or not isinstance(document["evaluations"], list)
        or not document["evaluations"]
    ):
        raise ValueError(
            "Expected a version 1 manifest with a nonempty evaluations list"
        )
    runs = sorted(
        {
            (manifest.parent / entry["report"]).resolve().parent
            for entry in document["evaluations"]
        }
    )
    for run in runs:
        path = run / ".run.lock"
        if not path.is_file():
            raise ValueError(f"Expected a benchmark run lock: {run}")
        lock = locks.enter_context(path.open())
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Evaluation is still active: {run}") from error
    results = [
        prepare_evaluation(entry, manifest.parent) for entry in document["evaluations"]
    ]
    destinations = [result.destination for result in results]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate evaluation destinations in manifest")
    for result in results:
        result.evidence.pin(manifest, sha)
    return sorted(
        results,
        key=lambda result: (len(result.destination.parts), str(result.destination)),
    )


def check_destinations(results, output):
    for result in results:
        destination = output / result.destination
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Evaluation destination already exists: {destination}"
            )
        for parent in destination.parents:
            if parent == output.parent:
                break
            if parent.is_symlink():
                raise ValueError("Output ancestors must not be symbolic links")
            if parent.exists() and not parent.is_dir():
                raise ValueError("An output ancestor is not a directory")


def check_scopes(results, output):
    planned = {result.destination: result for result in results}
    for result in results:
        if len(result.destination.parts) == 2:
            continue
        parent = result.destination.parent
        if parent in planned:
            baseline = planned[parent]
            same_inputs(result.identity, baseline.identity)
            if (
                result.timing["dataset_split_sha256"]
                != baseline.timing["dataset_split_sha256"]
            ):
                raise ValueError(
                    "Nested evaluation has different stored splits from Main"
                )
            report = baseline.report
        else:
            path = output / parent / "report.json"
            if not path.is_file():
                raise ValueError(
                    "A nested evaluation needs an existing or planned Main evaluation"
                )
            report = result.evidence.read(path)
            timing = result.evidence.read(output / parent / "timing.json")
            expected_splits = {
                result.timing["dataset_split_sha256"],
                result.identity["data_files"]["split_indices.npy"]["sha256"],
            }
            if timing.get("dataset_split_sha256") not in expected_splits:
                raise ValueError(
                    "Nested evaluation has different stored splits from Main"
                )
        for key in ("model", "dataset", "seed", "primary_metric"):
            if report.get(key) != result.report[key]:
                raise ValueError(f"Nested evaluation differs from Main on {key}")


def install_directory(source, destination):
    destination.mkdir(mode=0o700)
    try:
        os.rename(source, destination)
    except BaseException:
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


def consolidate(manifest, output, log_dir=None, *, execute=False):
    """Validate explicit runs, then optionally install new self-contained evaluations."""
    manifest, output = map(lambda path: Path(path).absolute(), (manifest, output))
    log_dir = Path(log_dir).absolute() if log_dir is not None else None
    if log_dir is not None and (log_dir / "installed.json").exists():
        raise FileExistsError("Use a new log directory for each consolidation batch")
    with ExitStack() as locks:
        results = prepare_manifest(manifest, locks)
        return _consolidate_evaluations(
            manifest, output, log_dir, results, execute=execute
        )


def _consolidate_evaluations(manifest, output, log_dir, results, *, execute):
    if log_dir is not None:
        destinations = [(output / result.destination).resolve() for result in results]
        inputs = {manifest.resolve()} | {
            path for result in results for path in result.evidence.files
        }
        if any(log_dir.resolve().is_relative_to(dest) for dest in destinations):
            raise ValueError("Consolidation logs must not overlap evaluation folders")
        for name in ("preflight.json", "prepared.json", "installed.json"):
            log_file = (log_dir / name).resolve()
            if log_file in inputs or any(
                dest.is_relative_to(log_file) for dest in destinations
            ):
                raise ValueError(
                    "Consolidation logs must not overwrite inputs or evaluations"
                )
    for result in results:
        if any(path.is_relative_to(output.resolve()) for path in result.evidence.files):
            raise ValueError("Raw inputs must be outside the public destination")
    check_destinations(results, output)
    check_scopes(results, output)
    def build_receipt(status):
        return dict(
            schema_version=1,
            status=status,
            manifest=str(manifest),
            output=str(output),
            evaluations=[
                dict(
                    destination=str(result.destination),
                    status=result.report["status"],
                    sources=result.evidence.receipt(),
                )
                for result in results
            ],
        )

    if not execute:
        if any(result.report["hpo_enabled"] for result in results):
            from .hpo import prepare_history

            with tempfile.TemporaryDirectory(prefix="tabbiomed_preflight_") as tmp:
                tmp_root = Path(tmp)
                for index, result in enumerate(results):
                    if not result.report["hpo_enabled"]:
                        continue
                    target = tmp_root / str(index)
                    target.mkdir()
                    for filename, source in result.artifacts.items():
                        copy_verified(source, target / filename)
                    write_json(target / "report.json", result.report)
                    result.evidence.verify()
                    history = prepare_history(
                        target / "report.json",
                        result.source_path,
                        split_sha256=result.timing["dataset_split_sha256"],
                        metadata=result.metadata,
                        expected_source_sha256=result.evidence.files[result.source_path],
                    )
                    for path, sha in history.evidence.files.items():
                        if not path.is_relative_to(tmp_root):
                            result.evidence.pin(path, sha)
        for result in results:
            result.evidence.verify()
        check_destinations(results, output)
        receipt = build_receipt("validated")
        if log_dir is not None:
            write_json(log_dir / "preflight.json", receipt)
        return receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    installed, created_parents = [], []
    with tempfile.TemporaryDirectory(
        prefix=".results_", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        staged = []
        for index, result in enumerate(results):
            target = staging / str(index)
            target.mkdir()
            for filename, source in result.artifacts.items():
                copy_verified(source, target / filename)
            report_sha = write_json(target / "report.json", result.report)
            write_json(
                target / "timing.json", dict(result.timing, report_sha256=report_sha)
            )
            if result.report["hpo_enabled"]:
                from .hpo import prepare_history, stage_history

                result.evidence.verify()
                history = prepare_history(
                    target / "report.json",
                    result.source_path,
                    split_sha256=result.timing["dataset_split_sha256"],
                    metadata=result.metadata,
                    expected_source_sha256=result.evidence.files[result.source_path],
                )
                stage_history(history, target / "hpo")
                for path, sha in history.evidence.files.items():
                    if not path.is_relative_to(staging):
                        result.evidence.pin(path, sha)
            staged.append(target)
        for result in results:
            result.evidence.verify()
        check_destinations(results, output)
        receipt = build_receipt("prepared")
        if log_dir is not None:
            write_json(log_dir / "prepared.json", receipt)
        try:
            for result, source in zip(results, staged):
                destination = output / result.destination
                missing = []
                parent = destination.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created_parents.append(parent)
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(destination)
                install_directory(source, destination)
                installed.append(destination)
            receipt["status"] = "installed"
            if log_dir is not None:
                for record, destination in zip(receipt["evaluations"], installed):
                    record["files"] = {
                        str(path.relative_to(destination)): digest(path)
                        for path in sorted(destination.rglob("*"))
                        if path.is_file()
                    }
                write_json(log_dir / "installed.json", receipt)
        except BaseException:
            for destination in reversed(installed):
                shutil.rmtree(destination)
            for parent in reversed(created_parents):
                try:
                    parent.rmdir()
                except OSError:
                    pass
            raise
    return receipt
