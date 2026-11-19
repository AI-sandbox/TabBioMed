from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import pandas as pd
import json
import os
from .split_utils import load_split_config, create_group_stratified_split

VITALDB_RAW_DIR = Path("data/raw/EHR_Datasets/VitalDB")
VITALDB_CSV_FILE = "clinical_data.csv"


def categorize_cancer(dx):
    dx = dx.lower()
    if "gastric" in dx or "stomach" in dx:
        return "Gastric_Cancer"
    elif "lung" in dx:
        return "Lung_Cancer"
    elif "breast" in dx:
        return "Breast_Cancer"
    elif "thyroid" in dx:
        return "Thyroid_Cancer"
    elif "hepatocellular" in dx or ("liver" in dx and "cancer" in dx):
        return "Liver_Cancer"
    elif "pancreas" in dx or "pancreatic" in dx:
        return "Pancreatic_Cancer"
    else:
        return "Other_Cancer"


def categorize_non_cancer(dx):
    dx = dx.lower()
    if "stone" in dx or "gallbladder" in dx or "cholecystitis" in dx or ("polyp" in dx):
        return "Gallbladder_Disease"
    elif "hernia" in dx:
        return "Hernia"
    elif "renal" in dx or "kidney" in dx:
        return "Kidney_Disease"
    elif "thyroid" in dx:
        return "Thyroid_Benign"
    elif "breast" in dx:
        return "Breast_Benign"
    elif "nodule" in dx or "pulmonary" in dx:
        return "Lung_Nodule"
    elif "ovarian" in dx or "cyst" in dx or "myoma" in dx or ("uteri" in dx):
        return "Gyne_Benign"
    elif "donor" in dx:
        return "Donor"
    else:
        return "Other_NonCancer"


def preprocess_vitaldb(df):
    cols_to_drop = [
        "preop_be",
        "preop_sao2",
        "preop_hco3",
        "preop_paco2",
        "preop_pao2",
        "preop_ph",
        "icu_days",
    ]
    df_clean = df.drop(columns=cols_to_drop)
    df_clean = df_clean.drop(columns=["cormack"])
    df_clean = df_clean.drop(columns=["iv1"])
    df_clean["cline1_present"] = df_clean["cline1"].notnull().astype(int)
    df_clean["cline2_present"] = df_clean["cline2"].notnull().astype(int)
    df_clean = df_clean.drop(columns=["cline1", "cline2"])
    df_clean["aline1_present"] = df_clean["aline1"].notnull().astype(int)
    df_clean["aline2_present"] = df_clean["aline2"].notnull().astype(int)
    df_clean = df_clean.drop(columns=["aline1", "aline2"])
    df_clean["iv2_present"] = df_clean["iv2"].notnull().astype(int)
    df_clean = df_clean.drop(columns=["iv2"])
    df_clean["ETT_present"] = df_clean["tubesize"].notnull().astype(int)
    df_clean["LMA_present"] = df_clean["lmasize"].notnull().astype(int)
    df_clean["DLT_present"] = df_clean["dltubesize"].notnull().astype(int)
    df_clean = df_clean.drop(columns=["tubesize", "lmasize", "dltubesize"])
    df_clean["no_airway"] = (
        df_clean["airway"].isnull()
        & (df_clean["ETT_present"] == 0)
        & (df_clean["LMA_present"] == 0)
        & (df_clean["DLT_present"] == 0)
    ).astype(int)
    df_clean["airway_non_oral"] = (
        df_clean["airway"].isin(["Nasal", "Tracheostomy"]).astype(int)
    )
    df_clean = df_clean.drop(columns=["airway"])
    df_clean["intraop_uo_missing"] = df_clean["intraop_uo"].isnull().astype(int)
    df_clean["intraop_ebl_missing"] = df_clean["intraop_ebl"].isnull().astype(int)
    df_clean["position_supine"] = (df_clean["position"] == "Supine").astype(int)
    df_clean = df_clean.drop(columns=["position"])
    df_clean["preop_ecg_abnormal"] = (
        df_clean["preop_ecg"] != "Normal Sinus Rhythm"
    ).astype(int)
    df_clean = df_clean.drop(columns=["preop_ecg"])
    df_clean["preop_pft_abnormal"] = (df_clean["preop_pft"] != "Normal").astype(int)
    df_clean = df_clean.drop(columns=["preop_pft"])
    df_clean["ane_general"] = (df_clean["ane_type"] == "General").astype(int)
    df_clean = df_clean.drop(columns=["ane_type"])
    df_clean["is_cancer"] = df_clean["dx"].str.contains("cancer", case=False, na=False)
    df_clean.loc[df_clean["is_cancer"], "dx_group"] = df_clean.loc[
        df_clean["is_cancer"], "dx"
    ].apply(categorize_cancer)
    mask = ~df_clean["is_cancer"]
    df_clean.loc[mask, "dx_group"] = df_clean.loc[mask, "dx"].apply(
        categorize_non_cancer
    )
    df_clean.loc[
        (df_clean["dx_group"] == "Other_NonCancer")
        & df_clean["dx"].str.contains("Carcinoma", case=False, na=False),
        "dx_group",
    ] = "Other_Cancer"
    df_clean = df_clean.drop(columns=["is_cancer", "opname"])
    df_clean = df_clean.drop(columns=["dx"])
    df_clean["sex_binary"] = df_clean["sex"].map({"M": 1, "F": 0})
    df_clean = df_clean.drop(columns=["sex"])
    df_clean["age"] = df_clean["age"].replace(">89", 90)
    df_clean["age"] = df_clean["age"].astype(float)
    df_clean["case_duration"] = df_clean["caseend"]
    df_clean["op_duration"] = df_clean["opend"] - df_clean["opstart"]
    df_clean["ane_duration"] = df_clean["aneend"] - df_clean["anestart"]
    max_ane_duration_sec = 24 * 3600
    absurd_ane_ts = df_clean["aneend"].abs() > 100000000.0
    absurd_ane_dur = (df_clean["ane_duration"] < 0) | (
        df_clean["ane_duration"] > max_ane_duration_sec
    )
    n_sanitized = int((absurd_ane_ts | absurd_ane_dur).sum())
    if n_sanitized:
        print(
            f"Sanitized ane_duration to NaN for {n_sanitized} case(s) (negative, >24h, or |aneend|>1e8)."
        )
    df_clean.loc[absurd_ane_ts | absurd_ane_dur, "ane_duration"] = np.nan
    cols_to_drop = [
        "bmi",
        "casestart",
        "caseend",
        "opstart",
        "opend",
        "anestart",
        "aneend",
        "adm",
        "dis",
    ]
    df_clean = df_clean.drop(columns=cols_to_drop)
    df_clean = df_clean.drop(columns=["op_duration"])
    return df_clean


def generate_metadata(num_samples, num_features, cat_idx):
    metadata = {
        "name": "VitalDB",
        "task": "binary",
        "cat_idx": cat_idx,
        "source": "VitalDB",
        "description": "VitalDB perioperative clinical dataset for in-hospital death prediction (features available by end of operative/anesthesia case; post-case stay outcomes excluded)",
        "num_samples": num_samples,
        "num_features": num_features,
    }
    from tabbiomed.data.curation.vitaldb_provenance import metadata_fields

    metadata.update(metadata_fields())
    return metadata


def save_dataset(
    X, y, metadata, split_indices, output_dir, feature_names=None, sample_metadata=None
):
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
    if sample_metadata is not None:
        sample_metadata.to_csv(output_dir / "sample_metadata.csv", index=False)
    return


def process_vitaldb_dataset(csv_path=None, output_root=None):
    if csv_path is None:
        csv_path = VITALDB_RAW_DIR / VITALDB_CSV_FILE
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    from tabbiomed.data.curation.vitaldb_provenance import (
        digest,
        registry,
        sample_metadata,
        verify_values,
        verify_prepared,
    )

    if digest(csv_path) != registry()["source_sha256"]:
        raise ValueError("VitalDB clinical source differs from the verified release")
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Initial dataset shape: {df.shape}")
    print("Preprocessing dataset...")
    df_clean = preprocess_vitaldb(df)
    print(f"Processed dataset shape: {df_clean.shape}")
    dataset_name = "VitalDB"
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
    drop_cols = ["subjectid", "caseid", "death_inhosp"]
    X_df = df_clean.drop(columns=drop_cols)
    binary_cols = [
        "cline1_present",
        "cline2_present",
        "aline1_present",
        "aline2_present",
        "iv2_present",
        "ETT_present",
        "LMA_present",
        "DLT_present",
        "no_airway",
        "airway_non_oral",
        "intraop_uo_missing",
        "intraop_ebl_missing",
        "position_supine",
        "preop_ecg_abnormal",
        "preop_pft_abnormal",
        "ane_general",
        "sex_binary",
    ]
    categorical_cols = ["approach", "dx_group", "optype", "department"]
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
    y = df_clean["death_inhosp"].values
    verify_values(X, y, feature_names)
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    cat_idx = (
        [X_df.columns.get_loc(col) for col in cat_cols] if len(cat_cols) > 0 else []
    )
    print(f"Categorical column indices: {cat_idx}")
    metadata = generate_metadata(X.shape[0], X.shape[1], cat_idx)
    unified_dir = (
        Path(output_root) if output_root is not None else Path("data/unified_format")
    )
    output_dir = unified_dir / "VitalDB"
    save_dataset(
        X,
        y,
        metadata,
        split_indices,
        output_dir,
        feature_names=feature_names,
        sample_metadata=sample_metadata(df),
    )
    verify_prepared(output_dir, X, y)
    print(f"\nDataset saved successfully to {output_dir}")
    return (X, y, metadata, split_indices)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Curate vitaldb benchmark data").parse_args()
    process_vitaldb_dataset()
