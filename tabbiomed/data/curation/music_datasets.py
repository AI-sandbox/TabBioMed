from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import json
import numpy as np
import pandas as pd
from .split_utils import create_group_stratified_split, load_split_config

REPO_ROOT = Path.cwd()
MUSIC_RAW_DIR = REPO_ROOT / "data/raw/EHR_Datasets/MUSIC"
MUSIC_CSV_FILE = "subject-info.csv"
NAME = "MUSIC"
PROTOCOL = "music_resolved_outcomes_v1"
EXIT_LOST = 1
EXIT_TRANSPLANT = 2
DESCRIPTION = "MUSIC cause-of-death classification. Eleven patients lost to follow-up (exit code 1) and twenty heart-transplant patients (exit code 2) are excluded because no cause of death was observed. Class 0 is alive at the end of follow-up. Cause codes 1, 3, and 6/7 map to classes 1, 2, and 3."
BINARY_COLUMNS = [
    "Gender (male=1)",
    "Diabetes (yes=1)",
    "History of dyslipemia (yes=1)",
    "Peripheral vascular disease (yes=1)",
    "History of hypertension (yes=1)",
    "Prior Myocardial Infarction (yes=1)",
    "Signs of pulmonary venous hypertension (yes=1)",
    "Right ventricle contractility (altered=1)",
    "Calcium channel blocker (yes=1)",
    "Diabetes medication (yes=1)",
    "Amiodarone (yes=1)",
    "Angiotensin-II receptor blocker (yes=1)",
    "Anticoagulants/antitrombotics (yes=1)",
    "Betablockers (yes=1)",
    "Digoxin (yes=1)",
    "Loop diuretics (yes=1)",
    "Spironolactone (yes=1)",
    "Statins (yes=1)",
    "Hidralazina (yes=1)",
    "ACE inhibitor (yes=1)",
    "Nitrovasodilator (yes=1)",
]
CATEGORICAL_COLUMNS = [
    "HF etiology - Diagnosis",
    "Prior implantable device",
    "Prior Revascularization",
    "Syncope",
    "Mitral valve insufficiency",
    "Mitral flow pattern",
]


def clean_column_names(df):
    column_mapping = {}
    for col in df.columns:
        cleaned = " ".join(col.strip().split())
        if cleaned != col:
            column_mapping[col] = cleaned
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Cleaned {len(column_mapping)} column names")
    return df


def convert_comma_decimal_to_float(df, col):
    temp_col = df[col].astype(str).str.replace(",", ".").str.strip()
    mask = temp_col.replace("nan", np.nan).notna()
    non_numeric = (
        temp_col[mask].str.replace("-", "", 1).str.replace(".", "", 1).str.isnumeric()
        == False
    )
    if non_numeric.any():
        print(f"Non-numeric values found in column '{col}':")
        print(temp_col[mask][non_numeric])
    df[col] = pd.to_numeric(temp_col, errors="coerce")


def exclude_unresolved_outcomes(df):
    exit_col = "Exit of the study"
    cause_col = "Cause of death"
    id_col = "Patient ID"
    follow_col = "Follow-up period from enrollment (days)"
    for col in (exit_col, cause_col, id_col, follow_col):
        if col not in df.columns:
            raise ValueError(f"MUSIC source is missing {col}")
    exit_code = pd.to_numeric(df[exit_col], errors="coerce")
    cause = pd.to_numeric(df[cause_col], errors="coerce")
    lost = exit_code == EXIT_LOST
    transplant = exit_code == EXIT_TRANSPLANT
    if int(lost.sum()) != 11 or int(transplant.sum()) != 20:
        raise ValueError(
            f"Expected 11 lost-to-follow-up and 20 heart-transplant rows, got {int(lost.sum())} and {int(transplant.sum())}"
        )
    excluded = lost | transplant
    if not bool((cause[excluded] == 0).all()):
        raise ValueError("Unresolved MUSIC patients must have cause of death 0")
    records = []
    for idx in df.index[excluded]:
        follow = df.at[idx, follow_col]
        code = int(exit_code.at[idx])
        records.append(
            {
                "patient_id": str(df.at[idx, id_col]),
                "source_row": int(idx),
                "exit_code": code,
                "exit_meaning": "lost_to_follow_up"
                if code == EXIT_LOST
                else "heart_transplant",
                "cause_of_death": int(cause.at[idx]),
                "follow_up_days": None if pd.isna(follow) else int(float(follow)),
            }
        )
    kept = df.loc[~excluded].reset_index(drop=True)
    return (kept, records)


def preprocess_music(df):
    df, excluded = exclude_unresolved_outcomes(df)
    ecg_holter_cols = [
        "Hig-resolution ECG available",
        "ECG rhythm",
        "Q-waves (necrosis, yes=1)",
        "PR interval (ms)",
        "QRS duration (ms)",
        "QRS > 120 ms",
        "QT interval (ms)",
        "QT corrected",
        "Average RR (ms)",
        "Left ventricular hypertrophy (yes=1)",
        "Intraventricular conduction disorder",
        "Holter available",
        "Holter onset (hh:mm:ss)",
        "Holter rhythm",
        "minimum RR (ms)",
        "maximum RR (ms)",
        "RR range (ms)",
        "Number of ventricular premature beats in 24h",
        "Extrasystole couplets",
        "Ventricular Extrasystole",
        "Ventricular Tachycardia",
        "Number of ventricular premature contractions per hour",
        "Non-sustained ventricular tachycardia (CH>10)",
        "Number of supraventricular premature beats in 24h",
        "Paroxysmal supraventricular tachyarrhythmia",
        "Longest RR pause (ms)",
        "Bradycardia",
        "SDNN (ms)",
        "SDANN (ms)",
        "RMSSD (ms)",
        "pNN50 (%)",
    ]
    cols_to_drop = [col for col in ecg_holter_cols if col in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    cols_drop_patient = [
        "Follow-up period from enrollment (days)",
        "days_4years",
        "Exit of the study",
        "Body Mass Index (Kg/m2)",
    ]
    cols_to_drop = [col for col in cols_drop_patient if col in df.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)
    cols_to_clean = [
        "Albumin (g/L)",
        "Normalized Troponin",
        "Total Cholesterol (mmol/L)",
        "Gamma-glutamil transpeptidase (IU/L)",
        "Glucose (mmol/L)",
        "Hemoglobin (g/L)",
        "HDL (mmol/L)",
        "Potassium (mEq/L)",
        "LDL (mmol/L)",
        "Protein (g/L)",
        "T3 (pg/dL)",
        "T4 (ng/L)",
        "Troponin (ng/mL)",
        "TSH (mIU/L)",
        "Urea (mg/dL)",
        "Cardiothoracic ratio",
    ]
    for col in cols_to_clean:
        if col in df_clean.columns:
            convert_comma_decimal_to_float(df_clean, col)
    if "Average RR (ms)" in df_clean.columns:
        avg_rr_cols = [col for col in df_clean.columns if "Average RR" in col]
        if len(avg_rr_cols) > 1:
            for col in avg_rr_cols[1:]:
                df_clean = df_clean.drop(columns=col)
    df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce").astype("Int64")
    if "Patient ID" in df_clean.columns:
        df_clean.loc[df_clean["Patient ID"] == "P0167", "Age"] = 90
    death_cause_mapping = {0: 0, 1: 1, 3: 2, 6: 3, 7: 3}
    df_clean["Cause of death"] = df_clean["Cause of death"].map(death_cause_mapping)
    if "Patient ID" in df_clean.columns:
        df_clean = df_clean.drop(columns="Patient ID")
    return (df_clean, excluded)


def generate_metadata(num_samples, num_features, cat_idx, num_classes):
    metadata = {
        "name": "MUSIC",
        "task": "classification",
        "cat_idx": cat_idx,
        "source": "MUSIC",
        "description": DESCRIPTION,
        "label": "clinical & healthcare",
        "curation_protocol": PROTOCOL,
        "note": "Excluded exit code 1 (lost to follow-up, n=11) and exit code 2 (heart transplant, n=20). Follow-up length and the exit code are not features.",
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


def process_music_dataset(csv_path=None, output_dir=None):
    if csv_path is None:
        csv_path = MUSIC_RAW_DIR / MUSIC_CSV_FILE
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = REPO_ROOT / "data/unified_format" / NAME
    output_dir = Path(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, sep=";")
    print(f"Initial dataset shape: {df.shape}")
    print("Cleaning column names...")
    df = clean_column_names(df)
    print("Preprocessing dataset...")
    df_clean, excluded = preprocess_music(df)
    print(f"Processed dataset shape: {df_clean.shape}")
    print(f"Excluded unresolved outcomes: {len(excluded)}")
    split_config = load_split_config(NAME)
    if not split_config:
        raise ValueError(f"No packaged split configuration for {NAME}")
    split_indices = create_group_stratified_split(df_clean, split_config)
    X_df = df_clean.drop(columns=["Cause of death"])
    missing_cols = [
        col for col in BINARY_COLUMNS + CATEGORICAL_COLUMNS if col not in X_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"MUSIC categorical columns missing after preprocessing: {missing_cols}"
        )
    cat_cols = BINARY_COLUMNS + CATEGORICAL_COLUMNS
    print("\nConverting non-binary categorical columns to strings...")
    for col in CATEGORICAL_COLUMNS:
        mask = X_df[col].notna()
        X_df.loc[mask, col] = X_df.loc[mask, col].astype(str)
    feature_names = list(X_df.columns)
    X = X_df.values
    y = df_clean["Cause of death"].values
    counts = np.bincount(y.astype(int))
    cat_idx = [X_df.columns.get_loc(col) for col in cat_cols]
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Target distribution: {counts.tolist()}")
    print(f"Categorical column indices: {cat_idx}")
    if (
        X.shape != (961, 65)
        or counts.tolist() != [695, 61, 94, 111]
        or len(cat_idx) != 27
    ):
        raise ValueError(
            f"MUSIC repair did not match the expected cohort: shape={X.shape}, counts={counts.tolist()}, categoricals={len(cat_idx)}"
        )
    metadata = generate_metadata(X.shape[0], X.shape[1], cat_idx, len(np.unique(y)))
    save_dataset(X, y, metadata, split_indices, output_dir, feature_names=feature_names)
    exclusion = {
        "protocol": PROTOCOL,
        "excluded_exit_codes": {"1": "lost_to_follow_up", "2": "heart_transplant"},
        "n_excluded": len(excluded),
        "patients": excluded,
    }
    (output_dir / "exclusion.json").write_text(json.dumps(exclusion, indent=2) + "\n")
    print(f"\nDataset saved successfully to {output_dir}")
    return (X, y, metadata, split_indices)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=MUSIC_RAW_DIR / MUSIC_CSV_FILE,
        help="PhysioNet MUSIC subject-info.csv (semicolon-separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/unified_format/MUSIC",
        help="Directory for the repaired X.npy, y.npy, metadata, split, and feature names.",
    )
    args = parser.parse_args(argv)
    process_music_dataset(csv_path=args.csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
