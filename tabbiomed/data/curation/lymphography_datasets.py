from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
from collections import Counter
import hashlib
import json
import numpy as np

PROTOCOL = "lymphography_binary_v1"
MANIFEST = CONFIG_DIR / "lymphography_binary.json"
TARGET_MAPPING = {"0": "malign_lymph", "1": "metastases"}
DESCRIPTION = "Lymphography (OpenML ID 10), restricted to metastases (1; 81 samples) versus malignant lymph (0; 61 samples). The two normal and four fibrosis cases are excluded. All 18 source features and the source row order are retained. Evaluation uses five stratified outer folds and three stratified inner folds for hyperparameter selection; preprocessing is fitted within training partitions."


def registry():
    return json.loads(MANIFEST.read_text())


def value_digest(values):
    payload = json.dumps(
        np.asarray(values).tolist(), separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def select_cohort(X, labels, source_path):
    expected = registry()
    if (
        hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        != expected["source_sha256"]
    ):
        raise ValueError(
            "Lymphography source differs from the pinned OpenML-10 release"
        )
    if (
        dict(Counter(labels.astype(str))) != expected["source_class_counts"]
        or X.columns.tolist() != expected["feature_names"]
    ):
        raise ValueError(
            "Lymphography source classes/features differ from the verified release"
        )
    mask = labels.isin(TARGET_MAPPING.values())
    rows = np.flatnonzero(mask)
    if rows.tolist() != expected["source_row_indices"]:
        raise ValueError("Lymphography source row alignment changed")
    return (
        X.loc[mask].reset_index(drop=True),
        labels.loc[mask].reset_index(drop=True),
        rows,
    )


def verify_values(X, y):
    expected = registry()
    if (
        np.shape(X) != (142, 18)
        or np.shape(y) != (142,)
        or value_digest(X) != expected["X_values_sha256"]
        or (value_digest(y) != expected["y_values_sha256"])
    ):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )


def task_metadata():
    return dict(
        description=DESCRIPTION,
        task="binary",
        num_samples=142,
        num_features=18,
        num_classes=2,
        url="https://www.openml.org/d/10",
        source="OpenML_Dataset_ID_10",
        label="clinical & healthcare",
        target_mapping=TARGET_MAPPING,
        curation_protocol=PROTOCOL,
        source_num_samples=148,
        excluded_class_counts={"normal": 2, "fibrosis": 4},
        sample_metadata_file="sample_metadata.csv",
        split_strategy="cv",
        source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    )
