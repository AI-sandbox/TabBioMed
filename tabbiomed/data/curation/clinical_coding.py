from pathlib import Path
from .paths import CONFIG_DIR
from collections import defaultdict, deque
import hashlib
import json
import numpy as np

REPO = Path.cwd()
NHANES = "National_Health_and_Nutrition_Health_Survey"
NAMES = ("HeartDisease_UCI", "thyroid", NHANES)
PROTOCOL = "clinical_source_coding_v1"
REGISTRY = CONFIG_DIR / "clinical_source_coding.json"
THYROID_SOURCE = REPO / "data/raw/UCI_thyroid_ann"
THYROID_HASHES = {
    "ann-train.data": "3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6",
    "ann-test.data": "c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b",
}
FEATURES = {
    "thyroid": [
        "age",
        "sex",
        "on_thyroxine",
        "query_on_thyroxine",
        "on_antithyroid_medication",
        "sick",
        "pregnant",
        "thyroid_surgery",
        "I131_treatment",
        "query_hypothyroid",
        "query_hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
    ],
    NHANES: ["RIAGENDR", "PAQ605", "BMXBMI", "LBXGLU", "DIQ010", "LBXGLT", "LBXIN"],
}
CATEGORICAL = {
    "HeartDisease_UCI": [1, 2, 5, 6, 8, 10, 11, 12],
    "thyroid": list(range(1, 16)),
    NHANES: [0, 1, 4],
}
DESCRIPTIONS = {
    "HeartDisease_UCI": "Heart disease classification from four pooled UCI hospital cohorts. Zero cholesterol and resting blood pressure denote missing values. Missing categorical values remain NaN; imputation is fitted during model training. The stored stratified split evaluates new patients from the pooled cohorts.",
    "thyroid": "Three-class UCI ANN thyroid task in TALENT row order. Released UCI feature values are restored before TALENT per-source-file z-scoring; the 15 binary flags are categorical. Standard preprocessing removes hypopituitary, constant in training, leaving 20 features (14 categorical). Labels and the stored TabBioMed 70/15/15 split are preserved. Any model scaling is fitted on its training partition.",
    NHANES: "NHANES 2013-2014 age-group classification from seven predictors. RIAGENDR, PAQ605 and DIQ010 are categorical; PAQ605=7 (refused) is missing. DIQ010=3 (borderline) remains a distinct response. Age and respondent IDs are excluded. Labels and the stored TabBioMed 70/15/15 split are preserved.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_digest(values):
    array = np.array(values, dtype="<f8", order="C", copy=True)
    array[np.isnan(array)] = np.nan
    return hashlib.sha256(array.tobytes()).hexdigest()


def metadata_fields(name):
    return dict(
        curation_protocol=PROTOCOL,
        cat_idx=CATEGORICAL[name],
        description=DESCRIPTIONS[name],
    )


def restore_thyroid(X, y, source_dir=THYROID_SOURCE):
    buckets = defaultdict(deque)
    for filename, expected_hash in THYROID_HASHES.items():
        path = Path(source_dir) / filename
        if digest(path) != expected_hash:
            raise ValueError(f"Unexpected UCI thyroid source: {path}")
        raw = np.loadtxt(path)
        values = raw[:, :21]
        scale = values.std(axis=0, ddof=1)
        scale[scale == 0] = 1
        encoded = (values - values.mean(axis=0)) / scale
        for index, row in enumerate(encoded):
            key = tuple((float(format(value, ".6g")) for value in row))
            buckets[key].append((values[index], int(raw[index, 21]) - 1))
    restored = []
    for row, label in zip(np.asarray(X, dtype=float), np.asarray(y).ravel()):
        matches = buckets[tuple(row)]
        if not matches:
            raise ValueError(
                "Thyroid row does not match an unused UCI source occurrence"
            )
        values, source_label = matches.popleft()
        if int(label) != source_label:
            raise ValueError("Thyroid source label mismatch")
        restored.append(values)
    if any(buckets.values()) or len(restored) != 7200:
        raise ValueError("Thyroid source coverage is incomplete")
    return np.asarray(restored)


def repair_values(name, X, y, *, thyroid_source=THYROID_SOURCE):
    if name == "thyroid":
        if X.shape[1] == 20 or np.isin(np.asarray(X)[:, 1:16], [0, 1]).all():
            reference = json.loads(REGISTRY.read_text())["datasets"][name]
            key = "unified_numeric_X_sha256" if X.shape[1] == 21 else "numeric_X_sha256"
            if numeric_digest(X) != reference.get(key, reference["numeric_X_sha256"]):
                raise ValueError(
                    "Restored thyroid values do not match the verified source"
                )
            return np.array(X, copy=True)
        return restore_thyroid(X, y, thyroid_source)
    result = np.array(X, copy=True)
    columns = (3, 4) if name == "HeartDisease_UCI" else (1,)
    sentinel = 0 if name == "HeartDisease_UCI" else 7
    for col in columns:
        result[result[:, col].astype(float) == sentinel, col] = np.nan
    return result


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    name = root.name
    md = json.loads((root / "metadata.json").read_text())
    reference = json.loads(REGISTRY.read_text())["datasets"][name]
    unified = root.parent.name == "unified_format"
    expected_cat = CATEGORICAL[name] if unified else reference["cat_idx"]
    if (
        md.get("curation_protocol") != PROTOCOL
        or sorted(md.get("cat_idx", [])) != expected_cat
    ):
        raise ValueError(
            f"{name}Regenerate this dataset from its pinned raw source, then prepare it."
        )
    X = np.load(root / "X.npy", allow_pickle=True) if X is None else X
    y = np.load(root / "y.npy", allow_pickle=True) if y is None else y
    shape = (
        reference.get("unified_shape", reference["shape"])
        if unified
        else reference["shape"]
    )
    expected_X = (
        reference.get("unified_numeric_X_sha256", reference["numeric_X_sha256"])
        if unified
        else reference["numeric_X_sha256"]
    )
    if list(X.shape) != shape or numeric_digest(X) != expected_X:
        raise ValueError(f"{name}: source feature values changed")
    if numeric_digest(np.asarray(y).ravel()) != reference["y_sha256"]:
        raise ValueError(f"{name}: target identity changed")
    if digest(root / "split_indices.npy") != reference["split_sha256"]:
        raise ValueError(f"{name}: stored split changed")
