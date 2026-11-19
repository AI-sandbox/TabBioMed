from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import os
import json
import argparse
import shutil
import numpy as np
import pandas as pd

from tabbiomed.cv_splits import ZERO_NONZERO, make_grouped_outer_cv
from tabbiomed.data.curation import mush21_datasets as mush21

RAW_PATH = "data/raw/TabPFN-VOCS/"
UNIFIED_DIR = "data/unified_format/"
SOURCE = "On the use of TabPFN on mass spectrometry analysis of volatile organic compounds. Scientific Reports (2025). GitHub: https://github.com/CIFASIS/TabPFN-VOCS"
LABEL = "Metabolomics"
SUB_LABELS = ["mass spectrometry", "volatile organic compounds"]
N_FOLDS = 5
RANDOM_SEED = 42
IN_SCOPE_VOCS = frozenset({"Gum1_s", "Noc_s", "PT-Mush21"})
MUSH21_TARGET_MAPPING = mush21.CULTURE_CODES
AREA_NORMALIZATION_NOTE = "Source PTR-ToF-MS peak values are area-normalized to a row sum of one and are retained without an additional logarithmic transformation."
GUM1_VALIDATION_NOTE = "Three inner folds and the final-fit holdout stratify source formulation groups by zero versus nonzero component-1 concentration."
_SPLIT_SCALE_SUFFIX = "Feature scaling is fitted within each training partition. Five grouped outer folds test every sample exactly once. All samples sharing a source ID stay together in outer and inner CV and in the final-fit validation holdout; source IDs are split metadata, not features."
DATASETS = [
    (
        "PT-Mush21",
        "data_classif_2.json",
        "Funghi21",
        "classification",
        mush21.DESCRIPTION,
    ),
    (
        "Gum1_s",
        "data_reg.json",
        "Gum1s",
        "regression",
        "PTR-ToF-MS analysis of chewing gum base material (267 samples, 167 peaks, 27 batches of technical replicates). Regression: predicted value is the percentage concentration of component 1 in the base material formulation. S/P=1.60.",
    ),
    (
        "Noc_s",
        "data_reg.json",
        "Noc_S",
        "regression",
        "PTR-ToF-MS analysis of hazelnut paste (Corylus avellana L.) from different geographical origins, roasted at varying times and temperatures (72 samples, 380 peaks, 24 batches of technical replicates). Regression: predicted value is the roasting time. S/P=0.19. Experimental details in Mazzucotelli et al.",
    ),
]


def load_raw_dataset(json_file, json_key):
    fpath = os.path.join(RAW_PATH, json_file)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Raw data file not found: {fpath}")
    with open(fpath) as f:
        data = json.load(f)
    if json_key not in data:
        raise KeyError(
            f"Key '{json_key}' not found in {json_file}. Available keys: {list(data.keys())}"
        )
    entry = data[json_key]
    X = np.array(pd.DataFrame(entry["X"]).values, dtype=np.float32)
    y = np.array(entry["y"])
    ids = entry["IDs"]
    if json_key in {"Gum1s", "Noc_S", "Funghi21"}:
        if (
            not np.isfinite(X).all()
            or (X < 0).any()
            or (not np.allclose(X.sum(axis=1), 1, rtol=0, atol=3e-06))
        ):
            raise ValueError(
                "In-scope VOCS features must preserve source area normalization"
            )
    if json_key == "Funghi21":
        observed = {
            str(int(c)): set((str(g).split("_")[0] for g in np.asarray(ids)[y == c]))
            for c in np.unique(y)
        }
        if observed != {c: {code} for c, code in MUSH21_TARGET_MAPPING.items()}:
            raise ValueError(
                "PT-Mush21 labels no longer match the verified fungal culture codes"
            )
    return (X, y, ids)


def batch_aware_cv_folds(
    ids, y, task, n_folds=N_FOLDS, seed=RANDOM_SEED, validation_stratification=None
):
    return make_grouped_outer_cv(
        ids,
        y,
        task,
        n_folds=n_folds,
        seed=seed,
        validation_stratification=validation_stratification,
    )


def _num_classes(y, task):
    if task in ("binary", "classification"):
        return int(len(np.unique(y)))
    return None


def build_metadata(name, description, X, y, task):
    normalization = " " + AREA_NORMALIZATION_NOTE if name in IN_SCOPE_VOCS else ""
    suffix = (
        "Feature scaling is fitted within each training partition."
        if name == "PT-Mush21"
        else _SPLIT_SCALE_SUFFIX
    )
    full_description = description.rstrip() + normalization + " " + suffix
    if name == "Gum1_s":
        full_description += " " + GUM1_VALIDATION_NOTE
    meta = {
        "name": name,
        "source": SOURCE,
        "description": full_description,
        "label": LABEL,
        "sub_labels": SUB_LABELS,
        "task": task,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "cat_idx": [],
        "force_scale": True,
        "split_strategy": "cv",
        "split_group": "source_ID",
    }
    nc = _num_classes(y, task)
    if nc is not None:
        meta["num_classes"] = nc
    if name in IN_SCOPE_VOCS:
        meta["feature_representation"] = "source_area_normalized"
    if name == "Gum1_s":
        meta["validation_stratification"] = ZERO_NONZERO
    if name == "PT-Mush21":
        meta.update(mush21.source_metadata())
        meta["description"] = full_description
    return meta


def save_dataset(name, X, y, split_indices, metadata):
    dataset_info(metadata["name"])
    out_dir = os.path.join(UNIFIED_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    np.save(os.path.join(out_dir, "split_indices.npy"), split_indices)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    if name == "PT-Mush21":
        shutil.copy2(mush21.MAPPING, os.path.join(out_dir, "sample_metadata.csv"))
    n_folds = len(split_indices.get("cv_folds", []))
    fold0 = split_indices["cv_folds"][0]
    tr, va, te = (len(fold0["train"]), len(fold0["val"]), len(fold0["test"]))
    print(
        f"  Saved → {out_dir}  X={X.shape}  {n_folds}-fold CV  fold0: train={tr} val={va} test={te}"
    )


def _build_one(tabbiomed_name, json_file, json_key, task, description):
    print(f"  Loading  {tabbiomed_name}  ({json_file}['{json_key}']) ...")
    X, y, ids = load_raw_dataset(json_file, json_key)
    split_indices = (
        mush21.grouped_splits(X, y, ids)
        if tabbiomed_name == "PT-Mush21"
        else batch_aware_cv_folds(
            ids,
            y,
            task,
            validation_stratification=ZERO_NONZERO
            if tabbiomed_name == "Gum1_s"
            else None,
        )
    )
    metadata = build_metadata(tabbiomed_name, description, X, y, task)
    n_batches = len(set(ids))
    fold0 = split_indices["cv_folds"][0]
    print(
        f"    n={X.shape[0]}  feats={X.shape[1]}  batches={n_batches}  task={task}  fold0: tr={len(fold0['train'])} va={len(fold0['val'])} te={len(fold0['test'])}"
    )
    return (X, y, split_indices, metadata)


def _collect_all(*, raise_on_failure=False):
    failures = []
    for entry in DATASETS:
        tabbiomed_name, json_file, json_key, task, description = entry
        try:
            X, y, split_indices, metadata = _build_one(
                tabbiomed_name, json_file, json_key, task, description
            )
            yield (tabbiomed_name, X, y, split_indices, metadata)
        except Exception as e:
            failures.append(tabbiomed_name)
            import traceback

            print(f"  [ERROR] {tabbiomed_name}: {e}")
            traceback.print_exc()
    if raise_on_failure and failures:
        raise SystemExit(1)


def main():
    print("\n=== TabPFN-VOCS dataset generation ===\n")
    for name, X, y, split_indices, metadata in _collect_all(raise_on_failure=True):
        save_dataset(name, X, y, split_indices, metadata)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess TabPFN-VOCS metabolomics datasets for TabBioMed."
    )
    args = parser.parse_args()
    main()
