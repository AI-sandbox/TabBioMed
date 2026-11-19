from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import pandas as pd
import json
import os
from .split_utils import load_split_config, create_group_stratified_split

IACCD_RAW_DIR = Path("data/raw/EHR_Datasets/MIMIC_2")
IACCD_FILE = "full_cohort_data.csv"
SOURCE_URL = "https://physionet.org/content/mimic2-iaccd/1.0/"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
DESCRIPTION = "Indwelling Arterial Catheter Clinical Dataset (IACCD) for 28-day mortality prediction. TabBioMed uses the separately released PhysioNet mimic2-iaccd dataset, available without credentialed access under the Open Data Commons Attribution License v1.0; it does not directly access the credentialed MIMIC-II database. The whole-stay blood-gas test count (abg_count) is excluded from predictors."


def source_metadata():
    return dict(
        name="IACCD",
        source="PhysioNet mimic2-iaccd v1.0",
        source_url=SOURCE_URL,
        description=DESCRIPTION,
        license="ODC-By-1.0",
        license_url=LICENSE_URL,
        access_policy="Open access; no credentials required",
        curation_protocol="iaccd_no_abg_v1",
    )


def preprocess_iaccd(df):
    drop_cols = [
        "hosp_exp_flg",
        "icu_exp_flg",
        "mort_day_censored",
        "censor_flg",
        "icu_los_day",
        "hospital_los_day",
        "abg_count",
        "hour_icu_intime",
        "service_num",
        "day_icu_intime_num",
        "sepsis_flg",
    ]
    drop_cols = [col for col in drop_cols if col in df.columns]
    df_clean = df.drop(columns=drop_cols)
    if "gender_num" in df_clean.columns:
        gender_mode = df_clean["gender_num"].mode()
        if len(gender_mode) > 0:
            print(f"Found missing values in gender_num; left them as NaN")
        else:
            print("Warning: No mode found for gender_num")
    return df_clean


def generate_metadata(num_samples, num_features, cat_idx, num_classes):
    task = "binary"
    metadata = {
        "task": task,
        "cat_idx": cat_idx,
        **source_metadata(),
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


def process_iaccd_dataset(data_dir=None):
    if data_dir is None:
        data_dir = IACCD_RAW_DIR
    else:
        data_dir = Path(data_dir)
    filepath = data_dir / IACCD_FILE
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\nExpected directory: {data_dir}"
        )
    print(f"Loading data from {filepath}")
    print(f"  Reading {IACCD_FILE}...")
    df = pd.read_csv(filepath)
    print(f"    Shape: {df.shape}")
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count / len(df) * 100:.1f}%)")
    dataset_name = "IACCD"
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
    df_clean = preprocess_iaccd(df.copy())
    print(f"Processed dataset shape: {df_clean.shape}")
    drop_cols = ["day_28_flg"]
    X_df = df_clean.drop(columns=drop_cols)
    binary_cols = [
        "aline_flg",
        "gender_num",
        "chf_flg",
        "afib_flg",
        "renal_flg",
        "liver_flg",
        "copd_flg",
        "cad_flg",
        "stroke_flg",
        "mal_flg",
        "resp_flg",
    ]
    categorical_cols = ["service_unit", "day_icu_intime"]
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
    y = df_clean["day_28_flg"].values
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
    output_dir = unified_dir / "IACCD"
    save_dataset(X, y, metadata, split_indices, output_dir, feature_names=feature_names)
    print(f"\nDataset saved successfully to {output_dir}")
    return (X, y, metadata, split_indices)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Curate iaccd benchmark data").parse_args()
    process_iaccd_dataset()
