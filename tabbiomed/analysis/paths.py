import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path.cwd()
CONFIG = PACKAGE / "configs/analysis"
TEMPLATES = PACKAGE / "resources/analysis/templates"
DEFAULT_PLAN = "living.json" if os.environ.get("TBM_RESULTS") else "benchmark.json"
PLAN = Path(os.environ.get("TBM_ANALYSIS_PLAN", CONFIG / DEFAULT_PLAN)).expanduser().resolve()
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


def read_plan(path):
    path = Path(path)
    plan = json.loads(path.read_text())
    if "extends" in plan:
        reference = json.loads((path.parent / plan["extends"]).read_text())
        models = reference["models"]
        if plan.get("include_registered_results"):
            from tabbiomed.models import all_models, get_model_spec

            for name in all_models:
                if name not in models and (RESULTS / name).is_dir():
                    models[name] = {"hpo_enabled": get_model_spec(name).hpo_enabled}
        models.update(plan.get("models", {}))
        return reference
    return plan
