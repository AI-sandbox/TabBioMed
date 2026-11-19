from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def catalog():
    path = Path(__file__).resolve().parents[1] / "configs/datasets.json"
    return json.loads(path.read_text())["datasets"]


def dataset_names():
    return tuple(catalog())


def dataset_info(name):
    if name not in catalog():
        raise ValueError(f"Dataset is outside the benchmark: {name}")
    return catalog()[name]
