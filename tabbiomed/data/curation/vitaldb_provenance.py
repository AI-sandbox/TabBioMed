from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import hashlib
import json
import numpy as np
import pandas as pd

REGISTRY = CONFIG_DIR / "vitaldb_end_of_case.json"
NAME = "VitalDB"
PROTOCOL = "vitaldb_end_of_case_v1"
PREDICTION_TIME = "end_of_operative_anesthesia_case"
SOURCE_URL = "https://physionet.org/content/vitaldb/1.0.0/"
FORBIDDEN_FEATURES = {
    "icu_days",
    "los_days",
    "adm",
    "dis",
    "subjectid",
    "caseid",
    "death_inhosp",
}
STRING_FEATURES = {"department", "optype", "approach", "dx_group"}
DESCRIPTION = "VitalDB perioperative in-hospital mortality classification using 57 features available by the end of the operative/anesthesia case. ICU stay, hospital length of stay, admission/discharge timestamps and patient/case identifiers are excluded from X. Device/procedure indicators, grouped diagnoses and perioperative durations are retained; two invalid anesthesia durations are missing. ASA and laboratory missingness are preserved for training-fitted imputation. The stored patient-grouped, outcome-stratified fixed split is unchanged: 4,442/978/968 training/validation/test cases, containing 41/9/7 mortality-positive cases (57 cases from 48 patients). The small event counts limit validation and test precision."


def registry():
    return json.loads(REGISTRY.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_digest(X, names):
    frame = pd.DataFrame(X, columns=names)
    for name in names:
        frame[name] = (
            frame[name].astype("string")
            if name in STRING_FEATURES
            else pd.to_numeric(frame[name]).astype(float)
        )
    return hashlib.sha256(
        frame.to_csv(index=False, na_rep="<MISSING>", float_format="%.17g").encode()
    ).hexdigest()


def target_digest(y):
    return hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()


def split_digest(splits):
    return hashlib.sha256(
        json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sample_metadata(raw):
    return pd.DataFrame(
        {
            "row_index": np.arange(len(raw)),
            "case_id": raw.caseid.to_numpy(),
            "patient_id": raw.subjectid.to_numpy(),
        }
    )


def metadata_fields():
    return dict(
        description=DESCRIPTION,
        source="VitalDB",
        label="clinical & healthcare",
        task="binary",
        num_classes=2,
        prediction_time=PREDICTION_TIME,
        curation_protocol=PROTOCOL,
        source_manifest_sha256=digest(REGISTRY),
        url=SOURCE_URL,
        source_url=SOURCE_URL,
        license="CC-BY-4.0",
        target_mapping={"0": "no in-hospital death", "1": "in-hospital death"},
        split_strategy="fixed",
        split_group="patient",
        sample_metadata_file="sample_metadata.csv",
    )


def verify_values(X, y, feature_names):
    reference = registry()
    if (
        list(feature_names) != reference["feature_names"]
        or set(feature_names) & FORBIDDEN_FEATURES
        or np.shape(X) != (6388, 57)
        or (np.shape(y) != (6388,))
        or (not np.isin(y, [0, 1]).all())
        or (matrix_digest(X, feature_names) != reference["X_values_sha256"])
        or (target_digest(y) != reference["y_values_sha256"])
    ):
        raise ValueError(
            "VitalDB requires the verified 57-feature end-of-case task without post-case outcomes"
        )


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    if X is None:
        X, y = (np.load(root / "X.npy", allow_pickle=True), np.load(root / "y.npy"))
    names = json.loads((root / "feature_names.json").read_text())
    verify_values(X, y, names)
    md = json.loads((root / "metadata.json").read_text())
    if (
        md.get("name") != NAME
        or (md.get("num_samples"), md.get("num_features")) != (6388, 57)
        or md.get("curation_protocol") != PROTOCOL
        or (md.get("prediction_time") != PREDICTION_TIME)
        or (md.get("source_manifest_sha256") != digest(REGISTRY))
        or (md.get("task") != "binary")
        or (md.get("split_strategy") != "fixed")
        or (md.get("cat_idx") != registry()["cat_idx"])
    ):
        raise ValueError(
            "VitalDB metadata predates the verified end-of-case task; run scripts/refresh_vitaldb.py"
        )
    splits = np.load(root / "split_indices.npy", allow_pickle=True).item()
    if split_digest(splits) != registry()["split_indices_sha256"]:
        raise ValueError("VitalDB requires the stored patient-disjoint fixed split")
    if digest(root / "sample_metadata.csv") != registry()["sample_metadata_sha256"]:
        raise ValueError("VitalDB source patient/case mapping changed")
    return splits
