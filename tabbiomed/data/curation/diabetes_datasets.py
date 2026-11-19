from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import pandas as pd
import json
import os
from .split_utils import load_split_config, create_group_stratified_split
from .diabetes_uci_provenance import (
    NOMINAL_ID_COLUMNS,
    digest,
    metadata_fields,
    registry,
    sample_metadata,
    verify_values,
    verify_prepared,
)

DIABETES_UCI_RAW_DIR = Path("data/raw/EHR_Datasets/Diabetes_UCI")
DIABETES_UCI_FILE = "diabetic_data.csv"


def icd9_group(code):
    if pd.isna(code) or code == "?":
        return "Missing"
    code = str(code)
    if code.startswith("E") or code.startswith("V"):
        return "Other"
    try:
        code_num = int(code.split(".")[0])
    except ValueError:
        return "Other"
    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    elif code_num == 250:
        return "Diabetes"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    elif code_num in [780, 781, 784] or 790 <= code_num <= 799:
        return "Other"
    elif 240 <= code_num <= 279 and code_num != 250:
        return "Other"
    elif 680 <= code_num <= 709 or code_num == 782:
        return "Other"
    elif 1 <= code_num <= 139:
        return "Other"
    elif 290 <= code_num <= 319:
        return "Other"
    elif 280 <= code_num <= 289:
        return "Other"
    elif 320 <= code_num <= 359:
        return "Other"
    elif 630 <= code_num <= 679:
        return "Other"
    elif 360 <= code_num <= 389:
        return "Other"
    elif 740 <= code_num <= 759:
        return "Other"
    else:
        return "Other"


def preprocess_diabetes_uci(df):
    df_sorted = df.sort_values(by=["patient_nbr", "encounter_id"])
    df_first_encounter = df_sorted.drop_duplicates(
        subset="patient_nbr", keep="first"
    ).reset_index(drop=True)
    hospice_death_codes = [11, 13, 14, 19, 20, 21]
    df_first_encounter = df_first_encounter[
        ~df_first_encounter["discharge_disposition_id"].isin(hospice_death_codes)
    ].copy()
    df_first_encounter = df_first_encounter.drop(columns=["weight", "payer_code"])
    df_first_encounter["readmitted_binary"] = (
        df_first_encounter["readmitted"] == "<30"
    ).astype(int)
    df_first_encounter = df_first_encounter.drop(columns=["readmitted"])
    df_first_encounter = df_first_encounter.rename(
        columns={"readmitted_binary": "readmitted_30"}
    )
    df_first_encounter = df_first_encounter.drop(columns=["diag_2", "diag_3"])
    df_first_encounter["race"] = df_first_encounter["race"].replace("?", "Missing")
    df_first_encounter["medical_specialty"] = df_first_encounter[
        "medical_specialty"
    ].replace("?", "Missing")
    df_first_encounter["diag_1_group"] = df_first_encounter["diag_1"].apply(icd9_group)
    df_first_encounter = df_first_encounter.drop(columns=["diag_1"])
    df_first_encounter["max_glu_serum"] = df_first_encounter["max_glu_serum"].replace(
        {None: "Not Measured", np.nan: "Not Measured"}
    )
    df_first_encounter["A1Cresult"] = df_first_encounter["A1Cresult"].replace(
        {None: "Not Measured", np.nan: "Not Measured"}
    )
    age_map = {
        "[0-10)": "[0-30)",
        "[10-20)": "[0-30)",
        "[20-30)": "[0-30)",
        "[30-40)": "[30-60)",
        "[40-50)": "[30-60)",
        "[50-60)": "[30-60)",
        "[60-70)": "[60-100)",
        "[70-80)": "[60-100)",
        "[80-90)": "[60-100)",
        "[90-100)": "[60-100)",
    }
    df_first_encounter["age_group"] = df_first_encounter["age"].map(age_map)
    df_first_encounter = df_first_encounter.drop(columns=["age"])
    df_first_encounter["diabetesMed"] = df_first_encounter["diabetesMed"].map(
        {"No": 0, "Yes": 1}
    )
    df_first_encounter["change"] = df_first_encounter["change"].map({"No": 0, "Ch": 1})
    df_first_encounter = df_first_encounter.drop(
        columns=["encounter_id", "patient_nbr"]
    )
    return df_first_encounter


def generate_metadata(num_samples, num_features, cat_idx, num_classes):
    task = "binary"
    description = (
        "Diabetes UCI dataset for 30-day readmission prediction (binary classification)"
    )
    metadata = {
        "name": "Diabetes_UCI",
        "task": task,
        "cat_idx": cat_idx,
        "source": "UCI Diabetes",
        "description": description,
        "num_samples": num_samples,
        "num_features": num_features,
        "num_classes": num_classes,
    }
    metadata.update(metadata_fields())
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


def process_diabetes_uci_dataset(data_dir=None, force=False, output_root=None):
    if data_dir is None:
        data_dir = DIABETES_UCI_RAW_DIR
    else:
        data_dir = Path(data_dir)
    filepath = data_dir / DIABETES_UCI_FILE
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Data file not found: {filepath}\nExpected directory: {data_dir}"
        )
    output_dir = Path(output_root or "data/unified_format") / "Diabetes_UCI"
    if output_dir.exists() and (not force):
        print(f"Unified dataset already exists at {output_dir}")
        print("Use --force to overwrite")
        return
    print(f"Loading data from {filepath}")
    print(f"  Reading {DIABETES_UCI_FILE}...")
    if digest(filepath) != registry()["source_sha256"]:
        raise ValueError(
            "Diabetes_UCI raw source differs from the verified UCI release"
        )
    df = pd.read_csv(filepath)
    print(f"    Shape: {df.shape}")
    print("\nMissing values per column:")
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} ({count / len(df) * 100:.1f}%)")
    print("\nPreprocessing dataset...")
    df_clean = preprocess_diabetes_uci(df.copy())
    print(f"Processed dataset shape: {df_clean.shape}")
    dataset_name = "Diabetes_UCI"
    split_config = load_split_config(dataset_name)
    split_indices = None
    if split_config:
        print(f"\nFound split configuration for {dataset_name}")
        print(f"  Group by: {split_config.get('group_by', None)}")
        print(f"  Stratify by: {split_config.get('stratify_by', None)}")
        print(
            f"  Split sizes: train={split_config.get('train_size', 0.7)}, val={split_config.get('val_size', 0.15)}, test={split_config.get('test_size', 0.15)}"
        )
        print("\nCreating stratified splits on preprocessed data...")
        split_indices = create_group_stratified_split(df_clean.copy(), split_config)
    else:
        print(f"\nNo split configuration found for {dataset_name}")
        print("Skipping splits - general pipeline will create random splits")
    id_cols_to_check = ["patient_nbr", "encounter_id"]
    remaining_id_cols = [col for col in id_cols_to_check if col in df_clean.columns]
    if remaining_id_cols:
        print(
            f"\nWarning: ID columns still present, dropping them now: {remaining_id_cols}"
        )
        df_clean = df_clean.drop(columns=remaining_id_cols)
    else:
        print("\nVerified: ID columns (patient_nbr, encounter_id) have been dropped")
    drop_cols = ["readmitted_30"]
    X_df = df_clean.drop(columns=drop_cols)
    binary_cols = ["diabetesMed", "change"]
    categorical_cols = []
    for col in X_df.columns:
        if (
            col in NOMINAL_ID_COLUMNS
            or pd.api.types.is_string_dtype(X_df[col].dtype)
            or X_df[col].dtype.name == "category"
        ):
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
            X_df[col] = X_df[col].astype(object)
            X_df.loc[mask, col] = X_df.loc[mask, col].astype(str)
            print(f"  Converted {col} to string")
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df_clean["readmitted_30"].values
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
    verify_values(X, y, feature_names)
    save_dataset(X, y, metadata, split_indices, output_dir, feature_names=feature_names)
    sample_metadata(df).to_csv(output_dir / "sample_metadata.csv", index=False)
    verify_prepared(output_dir, X, y)
    print(f"\nDataset saved successfully to {output_dir}")
    return (X, y, metadata, split_indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process Diabetes UCI dataset")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing unified dataset"
    )
    parser.add_argument(
        "--output-root", type=Path, help="Optional isolated unified-format destination"
    )
    args = parser.parse_args()
    process_diabetes_uci_dataset(force=args.force, output_root=args.output_root)
