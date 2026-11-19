"""Dataset access for the benchmark Parquet release and local NumPy inputs."""

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from .identity import (
    METADATA_FIELDS,
    array_digest,
    json_digest,
    split_digest,
    text_columns,
)
from .paths import CONFIG, DATA_CACHE, PARQUET
from .hub import hub_request


@lru_cache(maxsize=1)
def benchmark_datasets():
    return json.loads((CONFIG / "datasets.json").read_text())["datasets"]


@lru_cache(maxsize=1)
def public_source():
    return json.loads((CONFIG / "data_source.json").read_text())


def _info(name):
    if name not in benchmark_datasets():
        raise ValueError(f"Dataset is outside the benchmark: {name}")
    return benchmark_datasets()[name]


def _numpy_root(root):
    return Path(root).expanduser().resolve() if root is not None else None


@lru_cache(maxsize=4096)
def _download(relative):
    source = public_source()
    return Path(
        hub_request(
            hf_hub_download,
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            filename=relative,
            cache_dir=DATA_CACHE / "hub",
        )
    )


def _source_file(name, relative):
    if PARQUET is None:
        return _download(f"{name}/{relative}")
    path = PARQUET / name / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_metadata(name, metadata):
    if metadata.get("name") != name:
        raise ValueError(f"Dataset name differs from the requested dataset: {name}")
    if metadata.get("READY") is not True:
        raise ValueError(f"Dataset is not marked ready: {name}")
    actual = json_digest({key: metadata.get(key) for key in METADATA_FIELDS})
    if actual != _info(name)["metadata_sha256"]:
        raise ValueError(f"Dataset metadata differs from the benchmark release: {name}")
    reference = _info(name)["metadata"]
    for key, default in (("force_scale", False), ("cat_dims", None)):
        if metadata.get(key, default) != reference.get(key, default):
            raise ValueError(f"Dataset preprocessing metadata differs: {name}: {key}")
    return metadata


def _metadata_path(name, root=None):
    local = _numpy_root(root)
    if local is not None:
        return local / name / "metadata.json"
    return _source_file(name, "metadata.json")


@lru_cache(maxsize=256)
def _read_metadata(path, size, modified):
    return json.loads(Path(path).read_text())


def load_metadata(name, root=None):
    path = _metadata_path(name, root)
    stat = path.stat()
    return _validate_metadata(
        name, _read_metadata(str(path), stat.st_size, stat.st_mtime_ns)
    )


def dataset_directory(name, root=None, *, arrays=False):
    load_metadata(name, root)
    path = _metadata_path(name, root)
    if arrays:
        if _numpy_root(root) is not None:
            validate_arrays(name, path.parent)
        else:
            return _converted_directory(name, path)
    return path.parent


def outer_fold_count(metadata):
    return 5 if metadata.get("split_strategy") == "cv" else 0


def _parts(name):
    count = outer_fold_count(load_metadata(name))
    for fold in range(count or 1):
        base = f"data/fold_{fold}" if count else "data"
        for part in ("train", "val", "test"):
            yield (fold, part, f"{base}/{part}.parquet")


def _columns(name, path):
    metadata = load_metadata(name)
    columns = pq.ParquetFile(path).schema_arrow.names
    numbered = []
    for col in columns:
        match = re.fullmatch("f_(\\d+)(?:_.*)?", col)
        if match:
            numbered.append((int(match.group(1)), col))
    numbered.sort()
    if [index for index, _ in numbered] != list(range(metadata["num_features"])):
        raise ValueError(f"Invalid feature columns: {path}")
    if "target" in columns:
        targets = ["target"]
    else:
        targets = [col for col in columns if re.fullmatch("target_\\d+", col)]
        targets.sort(key=lambda col: int(col.removeprefix("target_")))
        if not targets or targets != [f"target_{i}" for i in range(len(targets))]:
            raise ValueError(f"Invalid target columns: {path}")
    features = [col for _, col in numbered]
    if set(columns) != {"row_idx", *features, *targets} or len(columns) != len(
        set(columns)
    ):
        raise ValueError(f"Unexpected Parquet schema: {path}")
    return (features, targets)


def _matrix_dtype(schemas, columns, *, has_nulls=False):
    types = [schema.field(col).type for schema in schemas for col in columns]
    if any(
        (
            pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_null(t)
            for t in types
        )
    ):
        return np.dtype(object)
    if not all(
        (
            pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)
            for t in types
        )
    ):
        raise ValueError("Unsupported Parquet value type")
    if has_nulls and any(
        pa.types.is_integer(t) or pa.types.is_boolean(t) for t in types
    ):
        return np.dtype(np.float64)
    return np.result_type(*(t.to_pandas_dtype() for t in types))


def _row_indices(values, name):
    ids = np.asarray(values)
    n = load_metadata(name)["num_samples"]
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= n)
    ):
        raise ValueError(f"Invalid original row IDs: {name}")
    return ids.astype(np.int64)


@lru_cache(maxsize=1)
def _public_files():
    source = public_source()
    directory = DATA_CACHE / "files"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{json_digest(source)}.json"
    with FileLock(str(path) + ".lock"):
        if not path.is_file():
            files = hub_request(
                HfApi().list_repo_files,
                source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
            )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(files) + "\n")
            os.replace(temporary, path)
    return frozenset(json.loads(path.read_text()))


def _has_groups(name):
    relative = f"{name}/split_groups.json"
    return (
        relative in _public_files()
        if PARQUET is None
        else (PARQUET / relative).is_file()
    )


def _attach_groups(name, splits):
    if _has_groups(name):
        groups = json.loads(_source_file(name, "split_groups.json").read_text())[
            "groups"
        ]
        splits["groups"] = groups
    splits.update(_info(name).get("split_metadata", {}))
    return splits


def validate_splits(name, splits, root=None):
    info = _info(name)
    metadata = load_metadata(name, root)
    policy = info.get("split_metadata", {})
    policy_fields = {
        key
        for dataset in benchmark_datasets().values()
        for key in dataset.get("split_metadata", {})
    }
    for key in sorted(policy_fields):
        if splits.get(key) != policy.get(key):
            raise ValueError(
                f"Stored split policy differs from the benchmark: {name}: {key}"
            )
    n_folds = outer_fold_count(metadata)
    if n_folds == 0 and "cv_folds" in splits:
        raise ValueError(f"Unexpected cv_folds in fixed-split dataset: {name}")
    actual = split_digest(splits, metadata["num_samples"], n_folds)
    if actual != info["splits_sha256"]:
        raise ValueError(f"Stored row memberships, row order or groups differ: {name}")
    return splits


def load_splits(name, root=None):
    load_metadata(name, root)
    local = _numpy_root(root)
    if local is not None:
        splits = np.load(local / name / "split_indices.npy", allow_pickle=True).item()
        return validate_splits(name, splits, root)
    directory, _ = _cache_directory(name)
    cached = directory / "split_indices.npy"
    if cached.is_file():
        return validate_splits(name, np.load(cached, allow_pickle=True).item())
    count = outer_fold_count(load_metadata(name))
    folds = [{} for _ in range(count or 1)]
    for fold, part, relative in _parts(name):
        path = _source_file(name, relative)
        _columns(name, path)
        ids = pq.read_table(path, columns=["row_idx"]).column("row_idx").to_numpy()
        folds[fold][part] = _row_indices(ids, name).tolist()
    splits = {"cv_folds": folds} if count else folds[0]
    return validate_splits(name, _attach_groups(name, splits))


def _load_array(path):
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


@lru_cache(maxsize=256)
def _array_digest(path, size, modified):
    array = _load_array(Path(path))
    return array_digest(array, text_columns(array))


def validate_arrays(name, directory):
    info = _info(name)
    for filename, key in (("X.npy", "features"), ("y.npy", "targets")):
        path = Path(directory) / filename
        stat = path.stat()
        actual = _array_digest(str(path), stat.st_size, stat.st_mtime_ns)
        if actual != info[f"{key}_sha256"]:
            raise ValueError(
                f"Dataset values or row/column order differ: {name}/{filename}"
            )


def load_arrays(name, root=None):
    directory = dataset_directory(name, root, arrays=True)
    return (_load_array(directory / "X.npy"), _load_array(directory / "y.npy"))


def data_identity(name, root=None):
    directory = dataset_directory(name, root, arrays=True)
    info = _info(name)
    validate_splits(
        name, np.load(directory / "split_indices.npy", allow_pickle=True).item(), root
    )
    metadata = load_metadata(name, root)
    return dict(
        metadata={key: metadata.get(key) for key in METADATA_FIELDS},
        features=info["features_sha256"],
        targets=info["targets_sha256"],
        splits=info["splits_sha256"],
    )


def _source_signature(name):
    if PARQUET is None:
        return public_source()
    paths = ["metadata.json", *(relative for _, _, relative in _parts(name))]
    if _has_groups(name):
        paths.append("split_groups.json")
    return {
        "root": str(PARQUET),
        "files": {
            relative: {
                "size": _source_file(name, relative).stat().st_size,
                "mtime_ns": _source_file(name, relative).stat().st_mtime_ns,
            }
            for relative in paths
        },
    }


def _file_states(directory):
    return {
        name: {
            "size": (directory / name).stat().st_size,
            "mtime_ns": (directory / name).stat().st_mtime_ns,
        }
        for name in ("X.npy", "y.npy", "split_indices.npy", "metadata.json")
    }


def _cache_directory(name):
    signature = _source_signature(name)
    content = {
        key: _info(name)[key]
        for key in ("features_sha256", "targets_sha256", "splits_sha256")
    }
    cache_key = json_digest(dict(source=signature, identity=content))[:24]
    return (DATA_CACHE / "numpy" / cache_key / name, content)


def _converted_directory(name, metadata_path):
    directory, content = _cache_directory(name)
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(parent / (name + ".lock"))):
        if directory.exists():
            receipt = json.loads((directory / "identity.json").read_text())
            if receipt["identity"] != content:
                raise ValueError(f"Converted dataset identity differs: {directory}")
            if receipt["files"] != _file_states(directory):
                _validate_metadata(
                    name, json.loads((directory / "metadata.json").read_text())
                )
                validate_arrays(name, directory)
                validate_splits(
                    name,
                    np.load(directory / "split_indices.npy", allow_pickle=True).item(),
                )
                _write_receipt(directory, content)
            return directory
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=parent))
        try:
            _convert(name, temporary)
            shutil.copyfile(metadata_path, temporary / "metadata.json")
            validate_arrays(name, temporary)
            validate_splits(
                name, np.load(temporary / "split_indices.npy", allow_pickle=True).item()
            )
            _write_receipt(temporary, content)
            os.replace(temporary, directory)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return directory


def _write_receipt(directory, content):
    path = directory / "identity.json"
    temporary = directory / ".identity.json"
    temporary.write_text(
        json.dumps(dict(identity=content, files=_file_states(directory)), indent=2)
        + "\n"
    )
    os.replace(temporary, path)


def _values_equal(first, second):
    if first.shape != second.shape:
        return False
    if first.dtype.kind != "O" and second.dtype.kind != "O":
        return np.array_equal(first, second, equal_nan=True)
    left, right = (pd.isna(first), pd.isna(second))
    return np.array_equal(left, right) and np.array_equal(first[~left], second[~right])


def _assign_checked(column, rows, values, fresh, path):
    previous = ~fresh
    if previous.any() and (not _values_equal(values[previous], column[rows[previous]])):
        raise ValueError(f"Inconsistent repeated rows: {path}")
    column[rows[fresh]] = values[fresh]
    if values.dtype != column.dtype and (
        not _values_equal(values[fresh], column[rows[fresh]])
    ):
        raise ValueError(f"Lossy conversion: {path}")


def _parquet_has_nulls(parquet_files, columns):
    col_set = set(columns)
    for pf in parquet_files:
        meta = pf.metadata
        if meta is None:
            continue
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col_meta = rg.column(col_idx)
                if col_meta.path_in_schema in col_set:
                    stats = col_meta.statistics
                    if stats is None or (
                        stats.has_null_count and stats.null_count > 0
                    ):
                        return True
    return False


def _convert(name, output):
    metadata = load_metadata(name)
    n_folds = outer_fold_count(metadata)
    part_entries = list(_parts(name))
    parquet_files = []
    schemas = []
    first_schema = None
    for _, _, relative in part_entries:
        path = _source_file(name, relative)
        features, targets = _columns(name, path)
        col_pair = (features, targets)
        if first_schema is not None and col_pair != first_schema:
            raise ValueError(
                f"Feature names or target columns differ across partitions: {name}"
            )
        first_schema = col_pair
        pf = pq.ParquetFile(path)
        parquet_files.append((path, pf))
        schemas.append(pf.schema_arrow)
    features, targets = first_schema
    n = metadata["num_samples"]
    x_shape = (n, metadata["num_features"])
    y_shape = (n,) if targets == ["target"] else (n, len(targets))
    x_dtype = _matrix_dtype(
        schemas, features, has_nulls=_parquet_has_nulls([pf for _, pf in parquet_files], features)
    )
    y_dtype = _matrix_dtype(
        schemas, targets, has_nulls=_parquet_has_nulls([pf for _, pf in parquet_files], targets)
    )
    x = (
        np.empty(x_shape, dtype=object)
        if x_dtype.hasobject
        else np.lib.format.open_memmap(
            output / "X.npy", mode="w+", dtype=x_dtype, shape=x_shape
        )
    )
    y = (
        np.empty(y_shape, dtype=object)
        if y_dtype.hasobject
        else np.lib.format.open_memmap(
            output / "y.npy", mode="w+", dtype=y_dtype, shape=y_shape
        )
    )
    folds = [{} for _ in range(n_folds or 1)]
    seen = np.zeros(x_shape[0], dtype=bool)
    for (fold, part, _), (path, parquet) in zip(part_entries, parquet_files):
        labels = parquet.read(columns=["row_idx", *targets], use_threads=False)
        rows = _row_indices(labels.column("row_idx").to_numpy(), name)
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"Duplicate Parquet row IDs: {path}")
        fresh = ~seen[rows]
        for target, field in enumerate(targets):
            values = labels.column(field).to_numpy()
            column = y if len(y_shape) == 1 else y[:, target]
            _assign_checked(column, rows, values, fresh, path)
        for start in range(0, len(features), 512):
            names = features[start : start + 512]
            offset = 0
            for batch in parquet.iter_batches(
                batch_size=8192, columns=["row_idx", *names], use_threads=False
            ):
                count = batch.num_rows
                ids = _row_indices(batch.column(0).to_numpy(zero_copy_only=False), name)
                expected = rows[offset : offset + count]
                if not np.array_equal(ids, expected):
                    raise ValueError(
                        f"Parquet row identities differ between column blocks: {path}"
                    )
                new = fresh[offset : offset + count]
                columns = [
                    batch.column(index + 1).to_numpy(zero_copy_only=False)
                    for index in range(len(names))
                ]
                if x_dtype.hasobject:
                    values = np.empty((count, len(names)), dtype=object)
                    for index, column in enumerate(columns):
                        values[:, index] = column
                else:
                    values = np.column_stack(columns)
                _assign_checked(
                    x[:, start : start + len(names)], ids, values, new, path
                )
                offset += count
            if offset != len(rows):
                raise ValueError(f"Missing rows in a Parquet column block: {path}")
        seen[rows] = True
        folds[fold][part] = rows.tolist()
    if not seen.all():
        raise ValueError(f"Missing original rows in Parquet dataset: {name}")
    if x_dtype.hasobject:
        np.save(output / "X.npy", x, allow_pickle=True)
    else:
        x.flush()
    if y_dtype.hasobject:
        np.save(output / "y.npy", y, allow_pickle=True)
    else:
        y.flush()
    splits = {"cv_folds": folds} if n_folds else folds[0]
    np.save(
        output / "split_indices.npy", _attach_groups(name, splits), allow_pickle=True
    )
