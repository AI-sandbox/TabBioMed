"""Prepare and submit complete 24-hour model/dataset evaluations."""

import argparse
import importlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .models import all_models, get_model_spec, model_name_map
from .profiles import (
    PROFILES,
    PCA_MODELS,
    PCA_MIN_FEATURES,
    default_output,
    validate_profile,
)


MAX_ARRAY_SIZE = 1000


def prepare(
    directory,
    *,
    datasets=None,
    models=None,
    data_root="data/processed",
    output=None,
    partition=None,
    gpu_partition=None,
    gpu_type=None,
    model_module=None,
    python=None,
    config=None,
    throttle=None,
    profile="main",
):
    if model_module:
        importlib.import_module(model_module)
    scope = json.loads((Path(__file__).parent / "configs/datasets.json").read_text())[
        "datasets"
    ]
    if profile not in PROFILES:
        raise ValueError(f"Unknown experiment profile: {profile}")
    eligible = [
        name
        for name, entry in scope.items()
        if profile == "main" or entry["metadata"]["num_features"] > PCA_MIN_FEATURES
    ]
    datasets = list(eligible if datasets is None else datasets)
    default_models = (
        PCA_MODELS
        if profile == "pca500"
        else [m for m in all_models if m != "TabPFN-Wide"]
    )
    models = [model_name_map(m) for m in (default_models if models is None else models)]
    unknown = set(datasets) - set(scope)
    if unknown:
        raise ValueError(f"Datasets outside the benchmark scope: {sorted(unknown)}")
    if len(set(models)) != len(models) or len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate model or dataset selection")
    if throttle is not None and throttle < 1:
        raise ValueError("Array throttle must be positive")
    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists():
        raise FileExistsError("Use a new job directory for a new plan")
    tasks = []
    for model in models:
        supported = get_model_spec(model).tasks
        for dataset in datasets:
            validate_profile(profile, model, scope[dataset]["metadata"])
            if scope[dataset]["metadata"]["task"] in supported:
                tasks.append(
                    dict(model=model, dataset=dataset, gpu=get_model_spec(model).gpu)
                )
    if not tasks:
        raise ValueError("No supported model/dataset pairs")
    directory.mkdir(parents=True, exist_ok=True)
    batches = {}
    script_names = []
    for gpu, label in ((False, "cpu"), (True, "gpu")):
        indices = [i for i, t in enumerate(tasks) if t["gpu"] == gpu]
        if not indices:
            continue
        for chunk_idx in range(0, len(indices), MAX_ARRAY_SIZE):
            chunk = indices[chunk_idx : chunk_idx + MAX_ARRAY_SIZE]
            batch_no = chunk_idx // MAX_ARRAY_SIZE + 1
            script_name = (
                f"{label}.sbatch" if batch_no == 1 else f"{label}_{batch_no}.sbatch"
            )
            batches[script_name] = chunk
            script_names.append(script_name)
    manifest = dict(
        schema_version=1,
        tasks=tasks,
        batches=batches,
        scripts=script_names,
        data_root=str(Path(data_root).expanduser().resolve()),
        output=str(
            Path(output if output is not None else default_output(profile))
            .expanduser()
            .resolve()
        ),
        python=python or sys.executable,
        model_module=model_module,
        config=str(Path(config).expanduser().resolve()) if config else None,
        partition=partition,
        gpu_partition=gpu_partition,
        gpu_type=gpu_type,
        throttle=throttle,
        profile=profile,
    )
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    scripts = []
    escaped_dir = str(directory).replace("%", "%%")
    for script_name in script_names:
        chunk = batches[script_name]
        gpu = tasks[chunk[0]]["gpu"]
        label = script_name[: -len(".sbatch")]
        spec = "0" if len(chunk) == 1 else f"0-{len(chunk) - 1}"
        array = spec + (f"%{throttle}" if throttle else "")
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=tabbiomed-" + label,
            "#SBATCH --time=24:00:00",
            "#SBATCH --cpus-per-task=10",
            "#SBATCH --mem=128G",
            "#SBATCH --array=" + array,
            f'#SBATCH --output="{escaped_dir}/%A_%a.out"',
        ]
        selected_partition = gpu_partition if gpu else partition
        if selected_partition:
            lines.append("#SBATCH --partition=" + selected_partition)
        if gpu:
            lines.append(
                "#SBATCH --gres=gpu:" + (gpu_type + ":" if gpu_type else "") + "1"
            )
        lines += [
            "set -euo pipefail",
            "export OMP_NUM_THREADS=10",
            "export MKL_NUM_THREADS=10",
            "export OPENBLAS_NUM_THREADS=10",
            shlex.join(
                [
                    manifest["python"],
                    "-m",
                    "tabbiomed.slurm",
                    "run",
                    "--manifest",
                    str(path),
                    "--batch",
                    script_name,
                ]
            )
            + ' --index "$SLURM_ARRAY_TASK_ID"',
            "",
        ]
        script = directory / script_name
        script.write_text("\n".join(lines))
        scripts.append(script)
    return scripts


def submit(directory, *, execute=False):
    directory = Path(directory).expanduser().resolve()
    receipt = directory / "submission.json"
    results = json.loads(receipt.read_text()) if receipt.exists() else []
    submitted = {r["script"] for r in results}
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        script_names = json.loads(manifest_path.read_text()).get(
            "scripts", ["cpu.sbatch", "gpu.sbatch"]
        )
    else:
        script_names = ["cpu.sbatch", "gpu.sbatch"]
    scripts = [
        directory / name
        for name in script_names
        if (directory / name).is_file() and name not in submitted
    ]
    if not scripts and not results:
        raise FileNotFoundError("No prepared job scripts")
    for script in scripts:
        command = ["sbatch", "--parsable", str(script)]
        print(shlex.join(command), flush=True)
        if execute:
            job = subprocess.check_output(command, text=True).strip()
            results.append(dict(script=script.name, job_id=job))
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(results, indent=2) + "\n")
            temporary.replace(receipt)
    return results


def run_task(manifest_path, index, *, batch=None):
    document = json.loads(Path(manifest_path).read_text())
    tasks = document["tasks"]
    if batch is not None:
        batches = document.get("batches") or {}
        if batch not in batches:
            raise ValueError(f"Unknown job batch: {batch!r}")
        chunk = batches[batch]
        if type(index) is not int or not 0 <= index < len(chunk):
            raise ValueError(
                f"Task index must be an integer in [0, {len(chunk)})"
            )
        task_index = chunk[index]
    else:
        task_index = index
    if type(task_index) is not int or not 0 <= task_index < len(tasks):
        raise ValueError(f"Task index must be an integer in [0, {len(tasks)})")
    task = tasks[task_index]
    command = [
        document["python"],
        "-m",
        "tabbiomed.trainer",
        "--model",
        task["model"],
        "--dataset",
        task["dataset"],
        "--data-root",
        document["data_root"],
        "--output",
        document["output"],
        "--profile",
        document.get("profile", "main"),
    ]
    if task["gpu"]:
        command.append("--gpu")
    if document.get("model_module"):
        command += ["--model-module", document["model_module"]]
    if document.get("config"):
        command += ["--config", document["config"]]
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("prepare")
    plan.add_argument("--directory", type=Path, required=True)
    plan.add_argument("--datasets", nargs="+")
    plan.add_argument("--models", nargs="+")
    plan.add_argument("--data-root", default="data/processed")
    plan.add_argument("--output")
    plan.add_argument("--profile", choices=PROFILES, default="main")
    plan.add_argument("--partition")
    plan.add_argument("--gpu-partition")
    plan.add_argument("--gpu-type")
    plan.add_argument("--model-module")
    plan.add_argument("--python")
    plan.add_argument("--config")
    plan.add_argument("--throttle", type=int)
    launch = commands.add_parser("submit")
    launch.add_argument("--directory", type=Path, required=True)
    launch.add_argument("--execute", action="store_true")
    worker = commands.add_parser("run")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--batch")
    worker.add_argument("--index", type=int, required=True)
    args = vars(parser.parse_args(argv))
    action = args.pop("command")
    if action == "prepare":
        for path in prepare(**args):
            print(path)
    elif action == "submit":
        submit(**args)
    else:
        run_task(args["manifest"], args["index"], batch=args.get("batch"))


if __name__ == "__main__":
    main()
