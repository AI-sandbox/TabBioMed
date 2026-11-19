from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def temporary_runtime_environment():
    """Keep library caches in temporary runtime storage."""
    with tempfile.TemporaryDirectory(
        prefix="tabbiomed_", dir=os.environ.get("SLURM_TMPDIR") or None
    ) as tmp:
        values = {
            name: str(Path(tmp) / child)
            for name, child in (
                ("MPLCONFIGDIR", "matplotlib"),
                ("NUMBA_CACHE_DIR", "numba"),
                ("ILTM_CACHE_DIR", "iltm"),
            )
        }
        for path in values.values():
            Path(path).mkdir()
        previous = {name: os.environ.get(name) for name in values}
        try:
            os.environ.update(values)
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
