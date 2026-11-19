from pathlib import Path
from tabbiomed.data.catalog import dataset_names
import argparse
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TASK_MAP = {
    "binclass": "binary",
    "multiclass": "classification",
    "regression": "regression",
}
SPLIT_RANDOM_STATE = 42


def _make_talent_split_indices(y: np.ndarray, task: str) -> dict[str, list[int]]:
    n_total = len(y)
    idx = np.arange(n_total)
    stratify = None
    if task in ("binary", "classification"):
        stratify = y
    try:
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.3, random_state=SPLIT_RANDOM_STATE, stratify=stratify
        )
        y_temp = y[temp_idx] if stratify is not None else None
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=SPLIT_RANDOM_STATE, stratify=y_temp
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            idx, test_size=0.3, random_state=SPLIT_RANDOM_STATE
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=SPLIT_RANDOM_STATE
        )
    return {
        "train": np.sort(train_idx).tolist(),
        "val": np.sort(val_idx).tolist(),
        "test": np.sort(test_idx).tolist(),
    }


def sanitize_dataset_name(folder_name: str) -> str:
    return re.sub("[^\\w\\-]", "_", folder_name).strip("_") or "Talent_dataset"


def load_info(folder: Path) -> dict | None:
    info_path = folder / "info.json"
    if not info_path.exists():
        return None
    try:
        with open(info_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def ingest_one_talent_folder(
    folder: Path,
    output_base: Path,
    force: bool = False,
    *,
    errors: list[str] | None = None,
) -> bool:
    folder = Path(folder)
    info = load_info(folder)
    if not info:
        print(f"  Skip {folder.name}: no info.json")
        return False
    dataset_name = sanitize_dataset_name(info.get("name") or folder.name)
    if dataset_name not in dataset_names():
        print(f"  Skip {dataset_name}: outside the benchmark")
        return False
    task_type = info.get("task_type", "").strip().lower()
    task = TASK_MAP.get(task_type)
    if not task:
        print(f"  Skip {folder.name}: unknown task_type '{task_type}'")
        if errors is not None:
            errors.append(folder.name)
        return False
    n_num = int(info.get("n_num_features", 0))
    n_cat = int(info.get("n_cat_features", 0))
    N_train = load_npy(folder / "N_train.npy")
    N_val = load_npy(folder / "N_val.npy")
    N_test = load_npy(folder / "N_test.npy")
    C_train = load_npy(folder / "C_train.npy")
    C_val = load_npy(folder / "C_val.npy")
    C_test = load_npy(folder / "C_test.npy")
    y_train = load_npy(folder / "y_train.npy")
    y_val = load_npy(folder / "y_val.npy")
    y_test = load_npy(folder / "y_test.npy")
    if y_train is None or y_val is None or y_test is None:
        print(f"  Skip {folder.name}: missing y_train/y_val/y_test")
        if errors is not None:
            errors.append(folder.name)
        return False

    def flatten_y(a):
        if a is None:
            return None
        a = np.asarray(a)
        if a.ndim > 1:
            a = a.ravel()
        return a

    y_train = flatten_y(y_train)
    y_val = flatten_y(y_val)
    y_test = flatten_y(y_test)
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    n_total = n_train + n_val + n_test
    parts = []
    if (
        n_num > 0
        and N_train is not None
        and (N_val is not None)
        and (N_test is not None)
    ):
        N_train = np.atleast_2d(N_train) if N_train.ndim == 1 else N_train
        N_val = np.atleast_2d(N_val) if N_val.ndim == 1 else N_val
        N_test = np.atleast_2d(N_test) if N_test.ndim == 1 else N_test
        N_full = np.vstack([N_train, N_val, N_test])
        parts.append(N_full)
    if (
        n_cat > 0
        and C_train is not None
        and (C_val is not None)
        and (C_test is not None)
    ):
        C_train = np.atleast_2d(C_train) if C_train.ndim == 1 else C_train
        C_val = np.atleast_2d(C_val) if C_val.ndim == 1 else C_val
        C_test = np.atleast_2d(C_test) if C_test.ndim == 1 else C_test
        C_full = np.vstack([C_train, C_val, C_test])
        parts.append(C_full)
    if not parts:
        print(f"  Skip {folder.name}: no N_* or C_* arrays found")
        if errors is not None:
            errors.append(folder.name)
        return False
    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    y = np.concatenate([y_train, y_val, y_test])
    if X.shape[0] != n_total or y.shape[0] != n_total:
        print(
            f"  Skip {folder.name}: shape mismatch X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}, expected {n_total}"
        )
        if errors is not None:
            errors.append(folder.name)
        return False
    if dataset_name == "QSAR_biodegradation":
        from tabbiomed.data.curation.qsar_datasets import clean_rows

        X, y, _ = clean_rows(X, y)
        n_total = len(y)
    if dataset_name == "Cardiovascular-Disease-dataset":
        from tabbiomed.data.curation.cardiovascular_datasets import clean_rows

        X, y, _, _ = clean_rows(X, y)
        n_total = len(y)
    if dataset_name in {
        "dna",
        "CDC_Diabetes_Health_Indicators",
        "maternal_health_risk",
    }:
        from tabbiomed.data.curation.openml_datasets import (
            deduplicate_dropping_conflicts,
        )

        X_clean_df, y = deduplicate_dropping_conflicts(pd.DataFrame(X), y)
        X = X_clean_df.to_numpy(dtype=X.dtype)
        n_total = len(y)
    num_features = X.shape[1]
    if n_num > 0 and n_cat > 0:
        cat_idx = list(range(n_num, num_features))
    elif n_cat > 0:
        cat_idx = list(range(num_features))
    else:
        cat_idx = []
    split_indices = _make_talent_split_indices(y, task)
    num_classes = None
    if task in ("binary", "classification"):
        num_classes = int(info.get("num_classes", len(np.unique(y))))
    display_name = folder.name
    metadata = {
        "name": dataset_name,
        "task": task,
        "cat_idx": cat_idx,
        "num_samples": int(n_total),
        "num_features": int(num_features),
        "source": f"Talent Benchmark Dataset - {display_name}",
        "description": f"Talent Benchmark dataset: {display_name} (task={task}). Rows from official train/val/test files are concatenated; TabBioMed then applies a 70/15/15 train/val/test split (stratified for classification, random_state={SPLIT_RANDOM_STATE}), not the original Talent split boundaries.",
    }
    if num_classes is not None:
        metadata["num_classes"] = num_classes
    if dataset_name in {"thyroid", "National_Health_and_Nutrition_Health_Survey"}:
        from tabbiomed.data.curation.clinical_coding import (
            repair_values,
            metadata_fields,
            FEATURES,
        )

        X = repair_values(dataset_name, X, y)
        metadata.update(metadata_fields(dataset_name))
        cat_idx = metadata["cat_idx"]
    if dataset_name == "Cardiovascular-Disease-dataset":
        from tabbiomed.data.curation.cardiovascular_datasets import DESCRIPTION

        metadata["description"] = DESCRIPTION
    if dataset_name == "QSAR_biodegradation":
        from tabbiomed.data.curation.qsar_datasets import DESCRIPTION, PROTOCOL

        metadata.update(
            description=DESCRIPTION, curation_protocol=PROTOCOL, label="drug discovery"
        )
    output_dir = output_base / dataset_name
    if output_dir.exists() and (not force):
        print(f"  Skip (exists): {dataset_name}")
        return True
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X, allow_pickle=True)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "split_indices.npy", split_indices)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if dataset_name in {"thyroid", "National_Health_and_Nutrition_Health_Survey"}:
        (output_dir / "feature_names.json").write_text(
            json.dumps(FEATURES[dataset_name], indent=2) + "\n"
        )
    print(f"  Saved: {dataset_name} (n={n_total}, task={task}, cat_idx={len(cat_idx)})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Talent Benchmark datasets into unified format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/Talent_Datasets",
        help="Path to Talent_Datasets folder (contains one subfolder per dataset)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/unified_format",
        help="Output base directory for unified datasets",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing unified datasets"
    )
    args = parser.parse_args()
    input_dir = Path(args.input)
    output_base = Path(args.output)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"No subfolders in {input_dir}")
        raise SystemExit(1)
    print(f"Ingesting Talent datasets from {input_dir} into {output_base}")
    success = 0
    errors: list[str] = []
    for folder in sorted(subdirs):
        if ingest_one_talent_folder(
            folder, output_base, force=args.force, errors=errors
        ):
            success += 1
    print(
        f"Done: {success}/{len(subdirs)} datasets written. Run general pipeline to produce processed format."
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
