from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace


@contextmanager
def run_lock(run_dir):
    with (Path(run_dir) / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another process is already using this run directory"
            ) from exc
        yield


def selected_study(best):
    """Only the selected trial reference is needed by the existing refit helper."""
    if not best:
        return None
    return SimpleNamespace(best_trial=SimpleNamespace(number=best["trial_id"]))


def save_predictions(run_dir, label, predictions):
    if not predictions:
        return
    import torch

    path = Path(run_dir) / f"predictions_{label}.pt"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        os.chmod(tmp, (path.stat().st_mode & 0o777) if path.exists() else 0o664)
        torch.save(predictions[-1], tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
