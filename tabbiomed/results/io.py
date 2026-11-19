"""Checksummed artifact I/O for benchmark results."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), digest_bytes(raw)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".result_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest_bytes(raw)


def local_file(run, name):
    if not isinstance(name, str) or not name:
        raise ValueError("Artifact reference must be a nonempty relative path")
    reference = Path(name)
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("Artifact reference escapes its run directory")
    path = (Path(run) / reference).resolve()
    if not path.is_relative_to(Path(run).resolve()):
        raise ValueError("Artifact reference escapes its run directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_verified(source, target):
    source, target = Path(source), Path(target)
    before = digest(source)
    shutil.copyfile(source, target)
    os.chmod(target, 0o644)
    if digest(target) != before or digest(source) != before:
        raise ValueError("Exported file differs from its source")
