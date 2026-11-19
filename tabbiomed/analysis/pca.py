from tabbiomed.analysis.datasets import load_benchmark
from .common import analysis_dataset_metadata
from .paths import PROCESSED


def dataset_inventory():
    plan = load_benchmark()
    return {
        name: analysis_dataset_metadata(name, info, PROCESSED)
        for name, info in plan["datasets"].items()
    }


def eligible_datasets(inventory):
    return [name for name, info in inventory.items() if info["num_features"] > 2000]
