from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import pandas as pd
import json
import os
from .split_utils import load_split_config, create_group_stratified_split

PATIENTCARE_RAW_DIR = Path("data/raw/EHR_Datasets/PatientCare")
PATIENTCARE_FILE = "data-ori.csv"


def preprocess_patientcare(df):
    if "SOURCE" in df.columns:
        df["SOURCE"] = df["SOURCE"].map({"out": 0, "in": 1})
        if df["SOURCE"].isna().any():
            raise ValueError(
                f"Found unexpected values in SOURCE column. Expected only 'in' and 'out'."
            )
    if "SEX" in df.columns:
        mask = df["SEX"].notna()
        df.loc[mask, "SEX"] = df.loc[mask, "SEX"].astype(str).str.strip()
        if df["SEX"].isna().any():
            raise ValueError(f"Found missing values in SEX column.")
        invalid = ~df["SEX"].isin(["F", "M"])
        if invalid.any():
            raise ValueError(
                f"Found unexpected values in SEX column. Expected only 'F' and 'M'. Got: {df.loc[invalid, 'SEX'].unique().tolist()}"
            )
    return df


def generate_metadata(num_samples, num_features, cat_idx, num_classes):
    task = "binary"
    description = "PatientCare dataset for binary classification (0=out, 1=in)"
    metadata = {
        "name": "PatientCare",
        "task": task,
        "cat_idx": cat_idx,
        "source": "EHR Dataset for Patient Treatment Classification, Mujiono Sadikin (2020), version 1. Mendeley Data, DOI: 10.17632/7kv3rctx7m.1.",
        "description": description,
        "num_samples": num_samples,
        "num_features": num_features,
        "num_classes": num_classes,
    }
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


def process_patientcare_dataset(data_dir=None):
    if data_dir is None:
        data_dir = PATIENTCARE_RAW_DIR
    else:
        data_dir = Path(data_dir)
    filepath = data_dir / PATIENTCARE_FILE
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\nExpected directory: {data_dir}"
        )
    print(f"Loading data from {filepath}")
    print(f"  Reading {PATIENTCARE_FILE}...")
    df = pd.read_csv(filepath)
    print(f"    Shape: {df.shape}")
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count / len(df) * 100:.1f}%)")
    dataset_name = "PatientCare"
    split_config = load_split_config(dataset_name)
    split_indices = None
    if split_config:
        print(f"\nFound split configuration for {dataset_name}")
        print(f"  Group by: {split_config.get('group_by', None)}")
        print(f"  Stratify by: {split_config.get('stratify_by', None)}")
        print(
            f"  Split sizes: train={split_config.get('train_size', 0.7)}, val={split_config.get('val_size', 0.15)}, test={split_config.get('test_size', 0.15)}"
        )
        print("\nCreating stratified splits...")
        split_indices = create_group_stratified_split(df.copy(), split_config)
    else:
        print(f"\nNo split configuration found for {dataset_name}")
        print("Skipping splits - general pipeline will create random splits")
    print("\nPreprocessing dataset...")
    df_clean = preprocess_patientcare(df.copy())
    print(f"Processed dataset shape: {df_clean.shape}")
    drop_cols = ["SOURCE"]
    X_df = df_clean.drop(columns=drop_cols)
    binary_cols = []
    categorical_cols = []
    for col in X_df.columns:
        if col not in binary_cols:
            if X_df[col].dtype == "object" or X_df[col].dtype.name == "category":
                categorical_cols.append(col)
    all_cat_cols = binary_cols + categorical_cols
    cat_cols = [col for col in all_cat_cols if col in X_df.columns]
    if len(cat_cols) > 0:
        print(f"\nIdentified {len(cat_cols)} categorical columns: {cat_cols}")
        print(f"  Binary columns: {[col for col in cat_cols if col in binary_cols]}")
        print(
            f"  Non-binary categorical columns: {[col for col in cat_cols if col in categorical_cols]}"
        )
    print("\nConverting non-binary categorical columns to strings...")
    for col in cat_cols:
        if col in X_df.columns and col in categorical_cols:
            mask = X_df[col].notna()
            X_df.loc[mask, col] = X_df.loc[mask, col].astype(str)
            print(f"  Converted {col} to string")
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df_clean["SOURCE"].values
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
    if num_classes != 2:
        raise ValueError(
            f"Expected 2 classes for binary classification, but found {num_classes}"
        )
    metadata = generate_metadata(X.shape[0], X.shape[1], cat_idx, num_classes)
    unified_dir = Path("data/unified_format")
    output_dir = unified_dir / "PatientCare"
    save_dataset(X, y, metadata, split_indices, output_dir, feature_names=feature_names)
    print(f"\nDataset saved successfully to {output_dir}")
    return (X, y, metadata, split_indices)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(
        description="Curate patientcare benchmark data"
    ).parse_args()
    process_patientcare_dataset()
