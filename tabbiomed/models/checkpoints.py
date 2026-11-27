"""Resolve author-distributed model weights or explicitly supplied local files."""

import hashlib
import json
import os
from pathlib import Path

from filelock import FileLock


def resolve_checkpoint(family, filename, directory=None):
    if directory is not None:
        path = Path(directory).expanduser() / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    sources = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/model_checkpoints.json"
        ).read_text()
    )
    source = sources[family]
    cache = Path(
        os.environ.get(
            "TABBIOMED_CHECKPOINTS", Path.home() / ".cache/tabbiomed/checkpoints"
        )
    )
    folder = cache / family
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    with FileLock(str(path) + ".lock"):
        if not path.is_file():
            if "repo_id" in source:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    source["repo_id"],
                    filename,
                    revision=source["revision"],
                    local_dir=folder,
                )
            elif "urls" in source:
                from torch.hub import download_url_to_file

                download_url_to_file(source["urls"][filename], str(path), progress=True)
            else:
                raise ValueError(f"Unknown checkpoint family: {family}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != source["sha256"][filename]:
            path.unlink()
            raise ValueError(f"Checkpoint identity differs: {filename}")
    return path
