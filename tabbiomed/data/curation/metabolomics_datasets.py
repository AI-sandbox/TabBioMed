from tabbiomed.data.catalog import dataset_info
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .metabolomics_utils import load_dataXL
from . import st001047_datasets as st001047

RAW_PATH = "data/raw/metabolomics/"
UNIFIED_DIR = "data/unified_format/"
DESCRIPTIONS = {
    "MTBLS136": "A serum LC-MS dataset consisting of 949 named metabolites. We compare estrogen plus progestin (E+P; Class=1) vs. estrogen-only (E; Class=0).",
    "MTBLS161": "A serum 1H-NMR dataset consisting of 29 named metabolites. We compare ME/CFS (Class=1) vs. HC (Class=0) using the serum dataset.",
    "MTBLS404": "A urine LC-MS dataset consisting of 120 named metabolites. We compare males (Class=1) and females (Class=0).",
    "ST000369": "A serum GC-MS dataset consisting of 181 named metabolites. We compare lung cancer (adenocarcinoma and adenosquamous carcinoma; Class=1) vs. healthy controls (Class=0).",
    "ST000496": "A saliva GC-MS dataset consisting of 69 named metabolites. We compare after debridement (Class=1) and before debridement (Class=0). The 100 samples are paired within 50 patients; outer and inner cross-validation keep each patient's samples together. Patient identifiers are split metadata, not features.",
    "ST001000": "A stool LC-MS dataset consisting of measured metabolite features. We compare CD (Class=1) and UC (Class=0) using the C18neg dataset mode.",
    "ST001047": st001047.DESCRIPTION,
}


def build_metadata(name, description, X, task, sub_labels):
    if "MTBLS" in name:
        source = f"MetaboLights accession {name}, processed via CIMCB/MetabComparisonBinaryML."
    elif name in ["ST000369", "ST000496", "ST001000", "ST001047"]:
        source = f"Metabolomics Workbench study {name}, processed via CIMCB/MetabComparisonBinaryML."
    else:
        source = "Metabolights Directly"
    return {
        "name": name,
        "source": source,
        "description": description,
        "label": "Metabolomics",
        "sub_labels": sub_labels,
        "task": task,
        "num_classes": 2,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "cat_idx": [],
        "force_scale": True,
    }


def save_dataset(X, y, split_indices, metadata, out_dir):
    dataset_info(metadata["name"])
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    if split_indices is not None:
        np.save(os.path.join(out_dir, "split_indices.npy"), split_indices)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  Saved -> {out_dir}  X={X.shape} y={y.shape}")


def st000496_patient_splits(sample_ids, y, seed=42):
    sample_ids = pd.Series(sample_ids, dtype=str)
    parsed = sample_ids.str.extract("^Sa_(pre|pro)_(\\d+)$")
    if parsed.isna().any().any() or sample_ids.duplicated().any():
        raise ValueError(
            "ST000496 requires unique Sa_pre_<patient>/Sa_pro_<patient> sample IDs"
        )
    y = np.asarray(y)
    expected = (parsed[0] == "pro").to_numpy(dtype=int)
    if y.ndim != 1 or not np.array_equal(y, expected):
        raise ValueError("ST000496 labels must match pre=0 and pro=1 sample IDs")
    groups = parsed[1].to_numpy(dtype=int)
    patients, counts = np.unique(groups, return_counts=True)
    if len(patients) != 50 or not np.all(counts == 2):
        raise ValueError(
            "ST000496 requires exactly 50 patients with one sample per class"
        )
    if not all((set(y[groups == p]) == {0, 1} for p in patients)):
        raise ValueError("Each ST000496 patient must have one pre and one post sample")
    folds = []
    for fold_id, (pool, test) in enumerate(
        KFold(5, shuffle=True, random_state=seed).split(patients)
    ):
        shuffled = np.random.default_rng(seed + (fold_id + 1) * 10007).permutation(pool)
        val, train = (shuffled[:5], shuffled[5:])
        folds.append(
            {
                key: np.flatnonzero(np.isin(groups, patients[indices])).tolist()
                for key, indices in (("train", train), ("val", val), ("test", test))
            }
        )
    return {
        "cv_folds": folds,
        "groups": groups.tolist(),
        "group_source": "ST000496.xlsx:Data:SampleID patient suffix",
        "group_split_protocol": "st000496_patient_outer_v1",
    }


def _build_excel_datasets(names=None, *, raise_on_failure=False):
    print("Building CIMCB Datasets...")
    datasets_to_load = [
        ("MTBLS136", "LC-MS"),
        ("MTBLS161", "NMR"),
        ("MTBLS404", "LC-MS"),
        ("ST000369", "GC-MS"),
        ("ST000496", "GC-MS"),
        ("ST001000", "LC-MS"),
        ("ST001047", "NMR"),
    ]
    failures = []
    for name, m_type in datasets_to_load:
        if names is not None and name not in names:
            continue
        print(f"  Processing {name}...")
        try:
            filename = os.path.join(RAW_PATH, f"{name}.xlsx")
            dt, pt = load_dataXL(filename, "Data", "Peak")
            if name == "ST001047":
                st001047.verify_source(filename)
                X, y, meta, feature_names, samples = st001047.curate_tables(dt, pt)
                meta["name"] = name
                out_dir = os.path.join(UNIFIED_DIR, name)
                save_dataset(X, y, st001047.retained_splits(), meta, out_dir)
                with open(os.path.join(out_dir, "feature_names.json"), "w") as stream:
                    json.dump(feature_names, stream, indent=2)
                samples.to_csv(
                    os.path.join(out_dir, "sample_metadata.csv"), index=False
                )
                yield (name, X, y)
                continue
            valid_classes = {0: 0, 1: 1, "0": 0, "1": 1, 0.0: 0, 1.0: 1}
            dt = dt[dt["Class"].isin(valid_classes.keys())].copy()
            dt["Class"] = dt["Class"].map(valid_classes)
            y = dt["Class"].astype(int).values
            peak_list = pt["Name"].values
            X = dt[peak_list].values.astype(np.float32)
            X = np.log1p(X)
            sub_labels = ["NMR"] if m_type == "NMR" else ["mass spectrometry", m_type]
            meta = build_metadata(name, DESCRIPTIONS[name], X, "binary", sub_labels)
            split_indices = None
            if name == "ST000496":
                split_indices = st000496_patient_splits(dt["SampleID"].to_numpy(), y)
                meta.update(split_strategy="cv", split_group="patient")
            out_dir = os.path.join(UNIFIED_DIR, name)
            save_dataset(X, y, split_indices, meta, out_dir)
            yield (name, X, y)
        except Exception as e:
            failures.append(name)
            print(f"    Failed {name}: {e}")
            import traceback

            traceback.print_exc()
    if raise_on_failure and failures:
        raise SystemExit(1)


def main():
    return list(_build_excel_datasets(raise_on_failure=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    gen_datasets = main()
