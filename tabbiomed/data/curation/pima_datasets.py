from pathlib import Path
from .paths import CONFIG_DIR
import json
import numpy as np
from .clinical_coding import digest, numeric_digest

FEATURES = ["preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"]
MISSING_COLUMNS = [1, 2, 3, 4, 5]
PROTOCOL = "pima_zero_as_missing_v1"
REGISTRY = CONFIG_DIR / "pima_missing.json"
DESCRIPTION = "Pima diabetes classification (OpenML 37). Zero glucose, blood pressure, skinfold thickness, insulin and BMI denote missing measurements; zero pregnancies remain valid. Imputation is fitted during model training."


def repair_values(X):
    values = np.array(X, dtype=float, copy=True)
    if values.ndim != 2 or values.shape[1] != len(FEATURES):
        raise ValueError(
            "Pima requires the eight source features in their original order"
        )
    for column in MISSING_COLUMNS:
        values[values[:, column] == 0, column] = np.nan
    return values


def metadata_fields():
    return dict(curation_protocol=PROTOCOL, cat_idx=[], description=DESCRIPTION)


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    reference = json.loads(REGISTRY.read_text())
    md = json.loads((root / "metadata.json").read_text())
    if md.get("curation_protocol") != PROTOCOL or md.get("cat_idx") != []:
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )
    X = np.load(root / "X.npy", allow_pickle=True) if X is None else X
    y = np.load(root / "y.npy", allow_pickle=True) if y is None else y
    if list(X.shape) != [768, 8] or numeric_digest(X) != reference["numeric_X_sha256"]:
        raise ValueError("Pima feature values differ from the corrected source")
    if numeric_digest(np.asarray(y).ravel()) != reference["y_sha256"]:
        raise ValueError("Pima labels changed")
    if (
        root.parent.name != "unified_format"
        and digest(root / "split_indices.npy") != reference["split_sha256"]
    ):
        raise ValueError("Pima stored split changed")
