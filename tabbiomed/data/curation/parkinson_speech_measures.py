from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = str(Path.cwd())
UNIFIED_DIR = os.path.join(BASE_DIR, "data/unified_format")
RAW_DIR = os.path.join(BASE_DIR, "data/raw/parkinson_voice")
RANDOM_SEED = 42
SOURCE = "Parkinson's Speech with Multiple Types of Sound Recordings dataset in UC Irvine ML repository, at https://archive.ics.uci.edu/dataset/301/c+speech+dataset+with+multiple+types+of+sound+recordings"
LABEL = "clinical & healthcare"
SUB_LABELS = ["Speech Features"]
DESCRIPTION = "The PD database consists of training and test files. The training data belongs to 20 PWP (6 female, 14 male) and 20 healthy individuals (10 female, 10 male) who appealed at the Department of Neurology in Cerrahpasa Faculty of Medicine, Istanbul University. From all subjects, multiple types of sound recordings (26 voice samples including sustained vowels, numbers, words and short sentences) are taken. A group of 26 linear and time-frequency based features are extracted from each voice sample. This dataset uses only the train_data.txt file as the test data contains only positive Parkinson patients and is more like a 'clinical' test set."
COLUMNS = [
    "subject_id",
    "jitter_local",
    "jitter_local_absolute",
    "jitter_rap",
    "jitter_ppq5",
    "jitter_ddp",
    "shimmer_local",
    "shimmer_local_db",
    "shimmer_apq3",
    "shimmer_apq5",
    "shimmer_apq11",
    "shimmer_dda",
    "ac",
    "nth",
    "htn",
    "median_pitch",
    "mean_pitch",
    "std_pitch",
    "min_pitch",
    "max_pitch",
    "number_of_pulses",
    "number_of_periods",
    "mean_period",
    "std_period",
    "fraction_locally_unvoiced_frames",
    "number_of_voice_breaks",
    "degree_of_voice_breaks",
    "updrs",
    "class_information",
]


def save_dataset(
    name: str, X: np.ndarray, y: np.ndarray, metadata: dict, split_indices: dict = None
):
    dataset_info(metadata["name"])
    out_dir = os.path.join(UNIFIED_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    if split_indices is not None:
        np.save(os.path.join(out_dir, "split_indices.npy"), split_indices)
        si_info = f"  split: train={len(split_indices['train'])} val={len(split_indices['val'])} test={len(split_indices['test'])}"
    else:
        si_info = "  split: delegated"
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    feature_names = [
        c for c in COLUMNS if c not in ["subject_id", "updrs", "class_information"]
    ]
    with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=4)
    print(f"  Saved → {out_dir}  X={X.shape}  y={y.shape}{si_info}")


def build_metadata(
    name: str,
    description: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    label: str,
    sub_labels: list,
    cat_idx: list = None,
) -> dict:
    meta = {
        "name": name,
        "source": SOURCE,
        "description": description,
        "label": label,
        "sub_labels": sub_labels,
        "task": task,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "cat_idx": cat_idx if cat_idx is not None else [],
    }
    if task in ("binary", "classification"):
        meta["num_classes"] = int(len(np.unique(y)))
    return meta


def build_parkinson_speech() -> list:
    print("\n[Parkinson Speech Measures]")
    raw_path = os.path.join(RAW_DIR, "train_data.txt")
    if not os.path.exists(raw_path):
        print(f"    Raw file not found at {raw_path}")
        return []
    df = pd.read_csv(raw_path, header=None, names=COLUMNS)
    print(f"    Loaded {len(df)} samples.")
    subject_df = (
        df.groupby("subject_id").agg({"class_information": "first"}).reset_index()
    )
    train_subj, temp_subj = train_test_split(
        subject_df,
        test_size=0.3,
        stratify=subject_df["class_information"],
        random_state=RANDOM_SEED,
    )
    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=0.5,
        stratify=temp_subj["class_information"],
        random_state=RANDOM_SEED,
    )
    train_ids = set(train_subj["subject_id"])
    val_ids = set(val_subj["subject_id"])
    test_ids = set(test_subj["subject_id"])
    indices = np.arange(len(df))
    split_indices = {
        "train": indices[df["subject_id"].isin(train_ids)].tolist(),
        "val": indices[df["subject_id"].isin(val_ids)].tolist(),
        "test": indices[df["subject_id"].isin(test_ids)].tolist(),
    }
    X_cols = [
        c for c in COLUMNS if c not in ["subject_id", "updrs", "class_information"]
    ]
    X = df[X_cols].values.astype(np.float32)
    y = df["class_information"].values.astype(np.int64)
    ds_name = "Parkinson_Speech"
    return [(ds_name, X, y, split_indices, [], DESCRIPTION, "binary")]


def main():
    print("=" * 60)
    print("Parkinson Speech Measures dataset builder")
    print("=" * 60)
    all_datasets = build_parkinson_speech()
    if not all_datasets:
        raise SystemExit(1)
    for name, X, y, split, cat_idx, desc, task_type in all_datasets:
        meta = build_metadata(
            name=name,
            description=desc,
            X=X,
            y=y,
            task=task_type,
            label=LABEL,
            sub_labels=SUB_LABELS,
            cat_idx=cat_idx,
        )
        save_dataset(name, X, y, meta, split_indices=split)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Parkinson Speech Measures dataset for the TabBioMed benchmark."
    )
    args = parser.parse_args()
    main()
