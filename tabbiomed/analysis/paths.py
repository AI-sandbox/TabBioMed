import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path.cwd()
CONFIG = PACKAGE / "configs/analysis"
TEMPLATES = PACKAGE / "resources/analysis/templates"
PLAN = (
    Path(os.environ.get("TBM_ANALYSIS_PLAN", CONFIG / "benchmark.json"))
    .expanduser()
    .resolve()
)
MODEL_FAMILIES = (
    Path(os.environ.get("TBM_MODEL_FAMILIES", CONFIG / "model_families.json"))
    .expanduser()
    .resolve()
)
RESULTS = (
    Path(os.environ.get("TBM_RESULTS", ROOT / "data/results")).expanduser().resolve()
)


def _optional_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


PROCESSED = _optional_path("TBM_PROCESSED")
PARQUET = _optional_path("TBM_PARQUET")
DATA_CACHE = (
    Path(os.environ.get("TBM_DATA_CACHE", Path.home() / ".cache/tabbiomed"))
    .expanduser()
    .resolve()
)
OUTPUT = Path(os.environ.get("TBM_OUTPUT", ROOT / "outputs")).expanduser().resolve()
