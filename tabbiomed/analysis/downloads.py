import hashlib
import json
import os
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download
from tqdm.contrib.concurrent import thread_map

from .paths import CONFIG, DATA_CACHE, RESULTS
from tabbiomed.data.hub import hub_request


def ensure_results(root=None):
    env_results = os.environ.get("TBM_RESULTS")
    if root is None:
        target = (
            Path(env_results).expanduser().resolve() if env_results else RESULTS
        )
    else:
        target = Path(root).expanduser().resolve()
    if env_results or target != RESULTS:
        if not target.is_dir():
            raise FileNotFoundError(
                f"Local results directory does not exist: {target}"
            )
        return target
    return download_results(target)


def download_results(root=None):
    if root is None:
        env_results = os.environ.get("TBM_RESULTS")
        root = Path(env_results).expanduser().resolve() if env_results else RESULTS
    else:
        root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = json.loads((CONFIG / "results_source.json").read_text())
    key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    cache = DATA_CACHE / "results"
    cache.mkdir(parents=True, exist_ok=True)
    inventory = cache / f"{key}.json"
    with FileLock(str(inventory) + ".lock"):
        if not inventory.is_file():
            files = hub_request(HfApi().list_repo_files, **source, repo_type="dataset")
            for name in files:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Invalid remote result path: {name}")
            temporary = inventory.with_suffix(".tmp")
            temporary.write_text(json.dumps(files) + "\n")
            os.replace(temporary, inventory)
        files = json.loads(inventory.read_text())
    destination_key = hashlib.sha256(str(root).encode()).hexdigest()
    with FileLock(str(cache / f"{destination_key}.lock")):
        missing = [name for name in files if not (root / name).is_file()]
        if missing:
            print(
                f"Downloading {len(missing)} missing result files to {root}", flush=True
            )

            def download(name):
                hub_request(
                    hf_hub_download,
                    **source,
                    repo_type="dataset",
                    local_dir=root,
                    filename=name,
                )
                if not (root / name).is_file():
                    raise FileNotFoundError(f"Result download incomplete: {name}")

            thread_map(download, missing, max_workers=2, desc="Results", unit="file")
    return root
