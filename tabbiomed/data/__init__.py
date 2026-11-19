import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from filelock import FileLock
from .catalog import catalog, dataset_info, dataset_names


def verify_dataset(name, root="data/processed"):
    from .storage import load_metadata, load_splits, validate_arrays

    dataset_info(name)
    root = Path(root).expanduser()
    metadata = load_metadata(name, root)
    validate_arrays(name, root / name)
    load_splits(name, root)
    return metadata


def ensure_dataset(name, root="data/processed"):
    from . import storage

    dataset_info(name)
    root = Path(root).expanduser().resolve()
    destination = root / name
    lock_id = hashlib.sha256(str(destination).encode()).hexdigest()
    locks = storage.DATA_CACHE / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / (lock_id + ".lock"))):
        if destination.exists():
            verify_dataset(name, root)
            return destination
        root.mkdir(parents=True, exist_ok=True)
        source = storage.dataset_directory(name, arrays=True)
        temporary = Path(tempfile.mkdtemp(prefix="." + name + "-", dir=root))
        try:
            for filename in ("X.npy", "y.npy", "split_indices.npy", "metadata.json"):
                shutil.copyfile(source / filename, temporary / filename)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    verify_dataset(name, root)
    return destination


__all__ = [
    "catalog",
    "dataset_names",
    "dataset_info",
    "ensure_dataset",
    "verify_dataset",
]
