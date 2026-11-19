from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import os
import json
import argparse
import numpy as np
import pandas as pd

SOURCE = "CPTAC (Clinical Proteomic Tumor Analysis Consortium). Proteomics data from the UCEC (Uterine Corpus Endometrial Carcinoma) cohort, measured at University of Michigan (umich). Clinical labels from the MSSM (Icahn School of Medicine at Mount Sinai) annotation."
HISTOLOGY_COLUMN = "baseline/histologic_type"
HISTOLOGY_MAPPING = {"Endometrioid carcinoma": 0, "Serous carcinoma": 1}
DESCRIPTION_HISTOLOGY = "Binary classification of endometrioid (0) versus serous (1) carcinoma from CPTAC UCEC tumour proteomics (umich), aligned to MSSM baseline histological annotations. One tumour profile per patient; clear-cell, mixed and unrecorded histologies are excluded. The release contains 101 patients: 80 endometrioid and 21 serous. Normal-tissue profiles and clinical annotations are excluded from the features. Missing protein measurements remain NaN; the general pipeline removes constant/all-empty columns and fits imputation/scaling within training partitions. Five stratified outer folds use three-fold inner CV for hyperparameter selection."
DATASETS = {"cptac_histology": (HISTOLOGY_COLUMN, "binary", DESCRIPTION_HISTOLOGY)}
LABEL = "Proteomics"
SUB_LABELS = ["Clinical", "mass-spectrometry"]


def align_histology_samples(proteomics_df, clinical_df):
    if not proteomics_df.index.is_unique or not clinical_df.index.is_unique:
        raise ValueError(
            "CPTAC requires one unambiguous profile/annotation per patient"
        )
    ids = sorted(set(proteomics_df.index) & set(clinical_df.index))
    ids = [
        pid
        for pid in ids
        if not str(pid).endswith(".N")
        and clinical_df.loc[pid, HISTOLOGY_COLUMN] in HISTOLOGY_MAPPING
    ]
    if not ids:
        raise ValueError("No matched endometrioid/serous tumour patients")
    X = proteomics_df.loc[ids].to_numpy(dtype=np.float32)
    y = (
        clinical_df.loc[ids, HISTOLOGY_COLUMN]
        .map(HISTOLOGY_MAPPING)
        .to_numpy(dtype=np.int64)
    )
    return (X, y, ids)


def load_cached_ucec(data_dir):
    root = Path(data_dir)
    proteins = pd.read_csv(
        root / "umich-ucec/Report_abundance_groupby=protein_protNorm=MD_gu=2.tsv.gz",
        sep="\t",
    )
    proteins["Database_ID"] = proteins["Index"].map(lambda value: value.split("|")[0])
    proteins["Name"] = proteins["Index"].map(lambda value: value.split("|")[6])
    proteins = (
        proteins.set_index(["Name", "Database_ID"])
        .drop(columns=["Index", "MaxPepProb", "NumberPSM", "Gene"])
        .T
    )
    proteins = proteins.subtract(
        proteins.loc["ReferenceIntensity"], axis="columns"
    ).iloc[1:]
    proteins = proteins.drop([f"RefInt_pool{i:02d}" for i in range(1, 18)])
    mapping = pd.read_csv(
        root / "umich-ucec/aliquot_to_patient_ID.tsv.gz",
        sep="\t",
        index_col="aliquot_ID",
        usecols=["aliquot_ID", "patient_ID"],
    )["patient_ID"]
    proteins.index = [mapping.get(value, value) for value in proteins.index]
    proteins.index = [
        str(value) + ".N" if "NX" in str(value) else str(value)
        for value in proteins.index
    ]
    occurrence = pd.Series(proteins.index).groupby(pd.Series(proteins.index)).cumcount()
    proteins.index = [
        f"{value}.{count}" if count else value
        for value, count in zip(proteins.index, occurrence)
    ]
    proteins = proteins.drop("C3N-01825.1")
    proteins.columns = ["_".join(map(str, column)) for column in proteins.columns]
    clinical = pd.read_csv(
        root / "mssm-all_cancers/clinical_Pan-cancer.May2022.tsv.gz",
        sep="\t",
        low_memory=False,
    )
    clinical = clinical[
        (clinical.tumor_code == "UCEC") & (clinical.discovery_study != "No")
    ]
    return (proteins, clinical.set_index("case_id").sort_index())


def build_metadata(name, source, description, X, task):
    return {
        "name": name,
        "source": source,
        "description": description,
        "label": LABEL,
        "sub_labels": SUB_LABELS,
        "task": task,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "cat_idx": [],
    }


def save_dataset(X, y, metadata, out_dir):
    dataset_info(metadata["name"])
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X.npy", X)
    np.save(f"{out_dir}/y.npy", y)
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  Saved → {out_dir}\n    X={X.shape}  n_samples={X.shape[0]}")


def process_dataset(
    internal_name: str,
    clinical_col: str,
    task: str,
    description: str,
    proteomics_df,
    clinical_df,
    unified_dir: str,
):
    print(f"\n[{internal_name}]  target='{clinical_col}'  task={task}")
    X, y, patient_ids = align_histology_samples(proteomics_df, clinical_df)
    print(
        f"  Aligned samples: {X.shape[0]} (from {len(proteomics_df)} proteomics × {len(clinical_df)} clinical)"
    )
    meta = build_metadata(internal_name, SOURCE, description, X, task)
    if internal_name == "cptac_histology":
        meta.update(
            target_mapping={
                str(value): key for key, value in HISTOLOGY_MAPPING.items()
            },
            split_strategy="cv",
        )
    out_dir = os.path.join(unified_dir, internal_name)
    save_dataset(X, y, meta, out_dir)
    if internal_name == "cptac_histology":
        Path(out_dir, "sample_ids.json").write_text(
            json.dumps(patient_ids, indent=2) + "\n"
        )
        Path(out_dir, "feature_names.json").write_text(
            json.dumps(list(proteomics_df.columns), indent=2) + "\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CPTAC proteomics tasks, including binary tumour histology."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path.cwd() / "data/raw/cptac")
    parser.add_argument(
        "--output-dir", type=Path, default=Path.cwd() / "data/unified_format"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS), nargs="+", default=list(DATASETS)
    )
    args = parser.parse_args()
    unified_dir = args.output_dir
    if args.raw_dir.is_dir():
        proteomics_df, clinical_df = load_cached_ucec(args.raw_dir)
    else:
        import cptac

        print("Loading CPTAC UCEC dataset …")
        en = cptac.Ucec()
        print("Fetching proteomics (umich) …")
        proteomics_df = en.get_proteomics("umich")
        print("Fetching clinical (mssm) …")
        clinical_df = en.get_clinical("mssm")
    proteomics_df = proteomics_df.fillna(np.nan)
    if hasattr(proteomics_df.columns, "levels"):
        proteomics_df.columns = [
            "_".join(filter(None, map(str, col))).strip()
            for col in proteomics_df.columns.values
        ]
    print(f"  Proteomics: {proteomics_df.shape}  Clinical: {clinical_df.shape}")
    for internal_name, (col, task, _) in DATASETS.items():
        if col not in clinical_df.columns:
            print(
                f"  WARNING: column '{col}' not found in clinical dataframe (dataset '{internal_name}' will be skipped). Available columns: {list(clinical_df.columns)}"
            )
    for internal_name, (clinical_col, task, description) in DATASETS.items():
        if internal_name not in args.dataset:
            continue
        if clinical_col not in clinical_df.columns:
            continue
        process_dataset(
            internal_name=internal_name,
            clinical_col=clinical_col,
            task=task,
            description=description,
            proteomics_df=proteomics_df,
            clinical_df=clinical_df,
            unified_dir=unified_dir,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
