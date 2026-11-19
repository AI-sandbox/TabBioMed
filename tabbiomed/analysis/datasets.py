"""Analysis scopes and access to the shared benchmark dataset loader."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from tabbiomed.data import catalog
from tabbiomed.data import storage
from .paths import PLAN, PARQUET, PROCESSED


def _numpy_root(root):
    if PROCESSED is not None and PARQUET is not None:
        raise ValueError("Set either TBM_PROCESSED or TBM_PARQUET")
    return Path(root).expanduser().resolve() if root is not None else PROCESSED


def load_metadata(name, root=None):
    return storage.load_metadata(name, _numpy_root(root))


def dataset_directory(name, root=None, *, arrays=False):
    return storage.dataset_directory(name, _numpy_root(root), arrays=arrays)


def load_splits(name, root=None):
    return storage.load_splits(name, _numpy_root(root))


def load_arrays(name, root=None):
    return storage.load_arrays(name, _numpy_root(root))


def data_identity(name, root=None):
    return storage.data_identity(name, _numpy_root(root))


def outer_fold_count(metadata):
    return storage.outer_fold_count(metadata)


def load_benchmark(path=None, root=None):
    plan = json.loads(Path(path or PLAN).read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata_values = list(
            pool.map(lambda name: load_metadata(name, root), plan["datasets"])
        )
    for (name, reference), metadata in zip(plan["datasets"].items(), metadata_values):
        identity = {
            key: value
            for key, value in catalog()[name].items()
            if key.endswith("_sha256")
        }
        plan["datasets"][name] = dict(
            identity,
            **reference,
            task=metadata["task"],
            n_folds=outer_fold_count(metadata),
        )
    return plan
