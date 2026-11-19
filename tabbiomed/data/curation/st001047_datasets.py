from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import hashlib
import json
import numpy as np

PROTOCOL = "st001047_qc_v1"
CONFIG = CONFIG_DIR
MANIFEST = CONFIG / "st001047_qc.json"
SPLITS = CONFIG / "st001047_outer_splits.json"
DESCRIPTION = "Urinary NMR classification of gastric cancer (GC=1; 43 samples) versus healthy controls (HE=0; 40 samples). Retain 52 of 149 source features using the CIMCB workflow criteria QC_RSD < 20% and Perc_missing < 10% from the supplied Peak sheet. Benign and quality-control samples are excluded. Feature values are log1p transformed; missing measurements are preserved for training-fitted preprocessing. The existing five stratified outer folds and three inner folds are retained. Sample IDs and source annotations are metadata, never features."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(values):
    values = np.ascontiguousarray(values).copy()
    if values.dtype.kind == "f":
        values[np.isnan(values)] = np.nan
    return hashlib.sha256(values.tobytes()).hexdigest()


def registry():
    reference = json.loads(MANIFEST.read_text())
    if (
        reference["protocol"] != PROTOCOL
        or digest(SPLITS) != reference["outer_splits_sha256"]
    ):
        raise ValueError("ST001047 source/split registry changed; reverify the source")
    return reference


def retained_splits():
    registry()
    return json.loads(SPLITS.read_text())


def verify_source(path):
    if digest(path) != registry()["source_sha256"]:
        raise ValueError("ST001047 requires the pinned source Excel release")


def verify_values(X, y):
    expected = registry()
    if (
        np.shape(X) != (83, 52)
        or np.asarray(X).dtype != np.dtype("float32")
        or np.shape(y) != (83,)
        or (value_digest(X) != expected["X_values_sha256"])
        or (value_digest(np.asarray(y, dtype=np.int64)) != expected["y_values_sha256"])
    ):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )


def task_metadata():
    reference = registry()
    return dict(
        description=DESCRIPTION,
        task="binary",
        num_classes=2,
        num_samples=83,
        num_features=52,
        cat_idx=[],
        force_scale=True,
        label="Metabolomics",
        source="Metabolomics Workbench study ST001047, processed via CIMCB/MetabComparisonBinaryML.",
        sub_labels=["NMR"],
        split_strategy="cv",
        url="https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST001047",
        curation_protocol=PROTOCOL,
        qc_filter=reference["qc_filter"],
        source_num_features=149,
        target_mapping=reference["target_mapping"],
        sample_metadata_file="sample_metadata.csv",
        source_manifest_sha256=digest(MANIFEST),
    )


def curate_tables(data, peaks):
    import pandas as pd

    reference = registry()
    if (
        len(peaks) != 149
        or data.Class.value_counts().to_dict() != reference["source_class_counts"]
    ):
        raise ValueError("ST001047 source classes/feature count changed")
    selected = data.Class.isin(["GC", "HE"])
    keep = (peaks.QC_RSD < 20) & (peaks.Perc_missing < 10)
    if (
        np.flatnonzero(keep).tolist() != reference["retained_feature_indices"]
        or np.flatnonzero(selected).tolist() != reference["source_row_indices"]
        or data.loc[selected, "SampleID"].tolist() != reference["source_sample_ids"]
    ):
        raise ValueError("ST001047 QC features or source sample alignment changed")
    names = peaks.loc[keep, "Name"].tolist()
    if names != [f["Name"] for f in reference["retained_features"]]:
        raise ValueError("ST001047 retained feature identities changed")
    X = np.log1p(data.loc[selected, names].to_numpy(dtype=np.float32))
    y = data.loc[selected, "Class"].map({"HE": 0, "GC": 1}).to_numpy(dtype=np.int64)
    verify_values(X, y)
    samples = pd.DataFrame(
        {
            "row_idx": range(len(y)),
            "source_row_idx": np.flatnonzero(selected),
            "sample_id": reference["source_sample_ids"],
            "source_diagnosis": data.loc[selected, "Class"].to_numpy(),
        }
    )
    return (X, y, task_metadata(), names, samples)
