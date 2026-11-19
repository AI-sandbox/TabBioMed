"""Build an explicit consolidation manifest from unambiguous settled runs."""

import fcntl
import json
from pathlib import Path

from .sources import registered_name, safe_name


def create_manifest(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    entries = []
    pairs = set()
    for path in sorted(raw.rglob("report.json")):
        lock = path.parent / ".run.lock"
        if not lock.is_file():
            raise ValueError(f"Expected a benchmark run lock: {path.parent}")
        with lock.open() as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    f"Evaluation is still active: {path.parent}"
                ) from error
            report = json.loads(path.read_text())
        if report.get("status") not in ("complete", "incomplete", "failed"):
            raise ValueError(f"Evaluation has no settled outcome: {path}")
        scope = safe_name(report.get("scope", "main"))
        pair = registered_name(report["model"]), report["dataset"], scope
        if pair in pairs:
            raise ValueError(
                f"Multiple runs for {pair}, provide an explicit manifest choosing one"
            )
        pairs.add(pair)
        entry = dict(report=str(path), data_root=report["input_identity"]["data_root"])
        if scope != "main":
            entry["scope"] = scope
        entries.append(entry)
    if not entries:
        raise ValueError("No raw benchmark reports found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(dict(schema_version=1, evaluations=entries), stream, indent=2)
        stream.write("\n")
    return output
