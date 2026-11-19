"""Pinned raw inputs and dataset identity checks."""

import json
from pathlib import Path
import re

from .io import digest, digest_bytes


def safe_name(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", value)
        or value in (".", "..")
    ):
        raise ValueError(f"Expected a single safe directory name: {value!r}")
    return value


def registered_name(value):
    from tabbiomed.models import model_name_map

    safe_name(value)
    return model_name_map(value)


class Evidence:
    def __init__(self):
        self.files = {}
        self.absent = set()
        self.stats = {}

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        observed = digest(path)
        if expected is not None and observed != expected:
            raise ValueError(f"Source checksum changed: {path}")
        if path in self.files and self.files[path] != observed:
            raise ValueError(f"Source changed during preparation: {path}")
        self.files[path] = observed
        return observed

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        self.pin(path, digest_bytes(raw))
        return json.loads(raw)

    def optional(self, path):
        path = Path(path).resolve()
        if path.is_file():
            return path
        self.absent.add(path)
        return None

    def verify(self):
        for path, expected in self.files.items():
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"Source changed before publication: {path}")
        if any(path.exists() for path in self.absent):
            raise ValueError("Source artifacts appeared during preparation")
        for path, expected in self.stats.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != expected:
                raise ValueError(f"Dataset changed before publication: {path}")

    def receipt(self):
        return {str(path): value for path, value in sorted(self.files.items())}


def source_identity(report):
    identity = report.get("input_identity") or {}
    for key in ("model", "dataset", "seed"):
        if key not in report or identity.get(key) != report[key]:
            raise ValueError(f"Report and saved input identity disagree on {key}")
    files = identity.get("data_files") or {}
    for name in ("X.npy", "y.npy", "metadata.json", "split_indices.npy"):
        if not isinstance(files.get(name), dict) or not files[name].get("size"):
            raise ValueError(f"Missing saved dataset identity: {name}")
        if name in ("metadata.json", "split_indices.npy") and not files[name].get(
            "sha256"
        ):
            raise ValueError(f"Missing dataset SHA-256: {name}")
    return identity


def same_inputs(left, right):
    for key in ("model", "dataset", "seed", "cv_validation", "target_classes"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Evaluation input identities differ on {key}")
    for name, expected in left["data_files"].items():
        if name == "split_indices.npy":
            continue
        observed = right["data_files"].get(name, {})
        keys = ("size", "sha256") if expected.get("sha256") else ("size", "mtime_ns")
        if any(expected.get(key) != observed.get(key) for key in keys):
            raise ValueError(f"Evaluation dataset identities differ: {name}")


def dataset_splits(identity, data_root, evidence):
    import numpy as np
    from tabbiomed.data.identity import split_digest

    for name in ("metadata.json", "split_indices.npy"):
        evidence.pin(data_root / name, identity["data_files"][name]["sha256"])
    for name in ("X.npy", "y.npy"):
        path, expected = data_root / name, identity["data_files"][name]
        stat = path.stat()
        if stat.st_size != expected["size"]:
            raise ValueError(f"Dataset file size differs from the recorded run: {name}")
        if expected.get("sha256"):
            evidence.pin(path, expected["sha256"])
        elif stat.st_mtime_ns != expected.get("mtime_ns"):
            raise ValueError(
                f"Dataset file identity differs from the recorded run: {name}"
            )
        evidence.stats[path] = stat.st_size, stat.st_mtime_ns
    path = data_root / "split_indices.npy"
    stored = np.load(path, allow_pickle=True).item()
    folds = stored.get("cv_folds") or [stored]
    for fold in folds:
        if any(name not in fold for name in ("train", "val", "test")):
            raise ValueError("Stored split lacks train, validation or test indices")
    cv = bool(stored.get("cv_folds"))
    metadata = evidence.read(data_root / "metadata.json")
    split_sha256 = split_digest(
        stored, metadata["num_samples"], len(folds) if cv else 0
    )
    return folds, cv, split_sha256


def validate_labels(path, target_path, test_indices, *, target_classes=None):
    import numpy as np

    labels = np.load(path, allow_pickle=False)
    targets = np.load(target_path, allow_pickle=True)
    if target_classes is not None:
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder().fit(targets)
        if (
            not isinstance(target_classes, list)
            or len(target_classes) < 2
            or not np.array_equal(np.asarray(target_classes), encoder.classes_)
        ):
            raise ValueError("Recorded target classes differ from the stored targets")
        targets = encoder.transform(targets)
    expected = targets[test_indices]
    if labels.ndim not in (1, 2) or labels.shape != expected.shape:
        raise ValueError(
            f"Test labels do not match the stored test split size: {path} "
            f"({labels.shape} versus {expected.shape})"
        )
    try:
        np.testing.assert_array_equal(labels, expected)
    except AssertionError as error:
        raise ValueError(
            "Test label values differ from the stored outer test targets"
        ) from error
