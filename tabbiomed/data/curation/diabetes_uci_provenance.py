from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import hashlib
import json
import numpy as np
import pandas as pd

REGISTRY = CONFIG_DIR / "diabetes_uci_first_encounter.json"
NAME = "Diabetes_UCI"
PROTOCOL = "diabetes_uci_first_encounter_categorical_v1"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
NOMINAL_ID_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)
NUMERIC_FEATURES = {
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "change",
    "diabetesMed",
}
DESCRIPTION = "UCI Diabetes 130-US Hospitals: 30-day readmission prediction at discharge. Retain the complete earliest encounter per patient, including unmeasured HbA1c/glucose; never fill fields from later encounters. Exclude discharge codes 11, 13, 14, 19, 20 and 21. Group the primary ICD-9 diagnosis and age bands; drop secondary/tertiary diagnoses, weight, payer code and identifiers. Admission type, discharge disposition and admission source are nominal categories despite their integer source codes. The general pipeline drops three constant medication columns, leaving 40 features for 69,973 distinct patients. Preserve the stored stratified 70/15/15 fixed split. Unmeasured HbA1c/glucose are explicit Not Measured categories; unknown race/specialty are explicit Missing categories. Encoding/scaling is fitted on training data."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    return json.loads(REGISTRY.read_text())


def matrix_digest(X, names):
    frame = pd.DataFrame(X, columns=names)
    for name in names:
        if name in NUMERIC_FEATURES:
            frame[name] = pd.to_numeric(frame[name]).astype(float)
        elif name in NOMINAL_ID_COLUMNS:
            frame[name] = pd.to_numeric(frame[name]).astype("Int64").astype("string")
        else:
            frame[name] = frame[name].astype("string")
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
    first = raw.sort_values(["patient_nbr", "encounter_id"]).drop_duplicates(
        "patient_nbr", keep="first"
    )
    first = first[~first.discharge_disposition_id.isin([11, 13, 14, 19, 20, 21])]
    return pd.DataFrame(
        {
            "row_index": np.arange(len(first)),
            "source_row_index": first.index,
            "patient_id": first.patient_nbr.to_numpy(),
            "encounter_id": first.encounter_id.to_numpy(),
        }
    )


def metadata_fields():
    return dict(
        description=DESCRIPTION,
        source="UCI Diabetes",
        label="clinical & healthcare",
        task="binary",
        num_classes=2,
        curation_protocol=PROTOCOL,
        prediction_time="discharge",
        source_manifest_sha256=digest(REGISTRY),
        source_url=SOURCE_URL,
        url=SOURCE_URL,
        license="CC-BY-4.0",
        target_mapping={
            "0": "no readmission within 30 days",
            "1": "readmitted within 30 days",
        },
        split_strategy="fixed",
        split_group="patient",
        sample_metadata_file="sample_metadata.csv",
    )


def verify_values(X, y, names):
    reference = registry()
    variants = [
        v for v in reference["versions"].values() if v["feature_names"] == list(names)
    ]
    if (
        len(variants) != 1
        or np.shape(X) != (69973, len(names))
        or np.shape(y) != (69973,)
        or (not np.isin(y, [0, 1]).all())
        or (target_digest(y) != reference["y_values_sha256"])
        or (matrix_digest(X, names) != variants[0]["X_values_sha256"])
    ):
        raise ValueError(
            "Diabetes_UCI requires verified complete first encounters without later-encounter lab values"
        )
    return variants[0]


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    if X is None:
        X, y = (np.load(root / "X.npy", allow_pickle=True), np.load(root / "y.npy"))
    names = json.loads((root / "feature_names.json").read_text())
    variant = verify_values(X, y, names)
    md = json.loads((root / "metadata.json").read_text())
    if (
        md.get("name") != NAME
        or md.get("task") != "binary"
        or (md.get("num_samples"), md.get("num_features")) != np.shape(X)
        or (md.get("curation_protocol") != PROTOCOL)
        or (md.get("prediction_time") != "discharge")
        or (md.get("source_manifest_sha256") != digest(REGISTRY))
        or (md.get("split_strategy") != "fixed")
        or (md.get("cat_idx") != variant["cat_idx"])
    ):
        raise ValueError(
            "Diabetes_UCI requires categorical admission/discharge/source codes; run scripts/refresh_diabetes_uci.py"
        )
    splits = np.load(root / "split_indices.npy", allow_pickle=True).item()
    if split_digest(splits) != registry()["split_indices_sha256"]:
        raise ValueError(
            "Diabetes_UCI requires the preserved patient-disjoint fixed split"
        )
    if digest(root / "sample_metadata.csv") != registry()["sample_metadata_sha256"]:
        raise ValueError("Diabetes_UCI first-encounter patient mapping changed")
    return splits
