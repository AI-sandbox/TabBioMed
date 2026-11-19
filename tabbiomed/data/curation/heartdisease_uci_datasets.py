from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import pandas as pd
import json
import os
from .split_utils import load_split_config, create_group_stratified_split

HEARTDISEASE_RAW_DIR = Path("data/raw/EHR_Datasets/HeartDisease_UCI/HeartDisease_Data")
HEARTDISEASE_FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]
HEARTDISEASE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def preprocess_heartdisease(df):
    df = df.copy()
    for col in ("chol", "trestbps"):
        df.loc[pd.to_numeric(df[col], errors="coerce") == 0, col] = np.nan
    if "num" in df.columns:
        df["num"] = pd.to_numeric(df["num"], errors="coerce").fillna(0).astype(int)
        df["num"] = (df["num"] > 0).astype(int)
    return df


def generate_metadata(num_samples, num_features, cat_idx, num_classes):
    task = "binary"
    description = "Heart Disease UCI dataset for heart disease prediction (binary classification: 0=no disease, 1=disease present)"
    metadata = {
        "name": "HeartDisease_UCI",
        "task": task,
        "cat_idx": cat_idx,
        "source": "UCI Heart Disease",
        "description": description,
        "num_samples": num_samples,
        "num_features": num_features,
        "num_classes": num_classes,
    }
    from tabbiomed.data.curation.clinical_coding import metadata_fields

    metadata.update(metadata_fields("HeartDisease_UCI"))
    return metadata


def save_dataset(X, y, metadata, split_indices, output_dir, feature_names=None):
    dataset_info(metadata["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving processed dataset to {output_dir}")
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    if split_indices is not None:
        np.save(output_dir / "split_indices.npy", split_indices)
        print(f"Saved split_indices.npy")
    else:
        print(f"No split_indices - general pipeline will create random splits")
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if feature_names is not None:
        with open(output_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
    return


def process_heartdisease_dataset(data_dir=None):
    if data_dir is None:
        data_dir = HEARTDISEASE_RAW_DIR
    else:
        data_dir = Path(data_dir)
    missing_files = []
    for filename in HEARTDISEASE_FILES:
        filepath = data_dir / filename
        if not os.path.exists(filepath):
            missing_files.append(filename)
    if missing_files:
        raise FileNotFoundError(
            f"Data files not found: {missing_files}\nExpected directory: {data_dir}"
        )
    print(f"Loading data from {data_dir}")
    dataframes = []
    for filename in HEARTDISEASE_FILES:
        filepath = data_dir / filename
        print(f"  Reading {filename}...")
        df = pd.read_csv(filepath, header=None, na_values="?")
        df.columns = HEARTDISEASE_COLUMNS
        dataframes.append(df)
        print(f"    Shape: {df.shape}")
    print("\nCombining all datasets...")
    df_all = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset shape: {df_all.shape}")
    print("\nMissing values per column:")
    missing_counts = df_all.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count / len(df_all) * 100:.1f}%)")
    print("\nPreprocessing dataset...")
    df_clean = preprocess_heartdisease(df_all)
    print(f"Processed dataset shape: {df_clean.shape}")
    dataset_name = "HeartDisease_UCI"
    split_config = load_split_config(dataset_name)
    split_indices = None
    if split_config:
        print(f"\nFound split configuration for {dataset_name}")
        print(f"  Group by: {split_config.get('group_by', None)}")
        print(f"  Stratify by: {split_config.get('stratify_by', None)}")
        print(
            f"  Split sizes: train={split_config.get('train_size', 0.7)}, val={split_config.get('val_size', 0.15)}, test={split_config.get('test_size', 0.15)}"
        )
        print("\nCreating group/stratified splits...")
        split_indices = create_group_stratified_split(df_clean, split_config)
    else:
        print(f"\nNo split configuration found for {dataset_name}")
        print("Skipping splits - general pipeline will create random splits")
    drop_cols = ["num"]
    X_df = df_clean.drop(columns=drop_cols)
    binary_cols = ["sex"]
    categorical_cols = ["cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    all_cat_cols = binary_cols + categorical_cols
    cat_cols = [col for col in all_cat_cols if col in X_df.columns]
    if len(cat_cols) > 0:
        print(f"\nIdentified {len(cat_cols)} categorical columns: {cat_cols}")
        print(f"  Binary columns: {[col for col in cat_cols if col in binary_cols]}")
        print(
            f"  Non-binary categorical columns: {[col for col in cat_cols if col in categorical_cols]}"
        )
    if len(cat_cols) < len(all_cat_cols):
        missing_cols = set(all_cat_cols) - set(cat_cols)
        print(
            f"Warning: Some categorical columns not found in dataframe: {missing_cols}"
        )
    print("\nConverting non-binary categorical columns to strings...")
    for col in cat_cols:
        if col in X_df.columns and col in categorical_cols:
            mask = X_df[col].notna()
            X_df.loc[mask, col] = X_df.loc[mask, col].astype(str)
            print(f"  Converted {col} to string")
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df_clean["num"].values
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    cat_idx = (
        [X_df.columns.get_loc(col) for col in cat_cols] if len(cat_cols) > 0 else []
    )
    print(f"Categorical column indices: {cat_idx}")
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    metadata = generate_metadata(X.shape[0], X.shape[1], cat_idx, num_classes)
    unified_dir = Path("data/unified_format")
    output_dir = unified_dir / "HeartDisease_UCI"
    save_dataset(X, y, metadata, split_indices, output_dir, feature_names=feature_names)
    print(f"\nDataset saved successfully to {output_dir}")
    return (X, y, metadata, split_indices)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="Curate heart-disease benchmark data"
    ).parse_args()
    process_heartdisease_dataset()
