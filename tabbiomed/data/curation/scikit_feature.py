from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import json
import shutil
from scipy.io import loadmat

SCIKIT_FEATURE_RAW_DIR = Path("data/raw/scikit-feature")
UNIFIED_DIR = Path("data/unified_format")


def get_task_type(y):
    unique_values = np.unique(y)
    n_unique = len(unique_values)
    if n_unique == 2:
        return "binary"
    elif n_unique > 2:
        return "classification"
    else:
        return "regression"


def generate_metadata(name, X, y, source_file):
    task = get_task_type(y)
    n_classes = len(np.unique(y)) if task in ["binary", "classification"] else None
    metadata = {
        "name": name,
        "task": task,
        "cat_idx": [],
        "source": "scikit-feature",
        "description": f"Gene expression dataset from scikit-feature collection (source: {Path(source_file).name})",
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
    }
    if n_classes is not None:
        metadata["num_classes"] = n_classes
    return metadata


def save_dataset(X, y, metadata, split_indices=None):
    dataset_info(metadata["name"])
    output_dir = UNIFIED_DIR / metadata["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Task: {metadata['task']}, Classes: {metadata.get('num_classes', 'N/A')}")
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if split_indices is not None:
        np.save(output_dir / "split_indices.npy", split_indices)


def process_mat_file(filepath):
    filename = filepath.stem
    dataset_name = filename
    print(f"\nProcessing: {filepath.name}")
    data = loadmat(str(filepath))
    X = data["X"]
    y = data["Y"].flatten()
    metadata = generate_metadata(dataset_name, X, y, filepath.name)
    splits = None
    if dataset_name == "TOX_171":
        from .tox171_datasets import grouped_splits, source_metadata

        splits = grouped_splits(X, y)
        metadata.update(source_metadata())
    save_dataset(X, y, metadata, split_indices=splits)
    if dataset_name == "TOX_171":
        from tabbiomed.data.curation.tox171_datasets import MAPPING

        shutil.copy2(MAPPING, UNIFIED_DIR / dataset_name / "sample_metadata.csv")


def main():
    for name in (
        "SMK_CAN_187",
        "Prostate_GE",
        "GLIOMA",
        "GLI_85",
        "ALLAML",
        "CLL_SUB_111",
        "lung",
        "TOX_171",
    ):
        process_mat_file(SCIKIT_FEATURE_RAW_DIR / (name + ".mat"))


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="Curate scikit-feature benchmark data"
    ).parse_args()
    main()
