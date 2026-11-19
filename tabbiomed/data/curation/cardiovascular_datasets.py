from .paths import CONFIG_DIR
import json
import numpy as np
import pandas as pd

PROTOCOL = "cardiovascular_fixed_plausibility_v1"
BOUNDS = {
    "height_cm": [100, 230],
    "weight_kg": [20, 300],
    "systolic_mmHg": [50, 300],
    "diastolic_mmHg": [20, 200],
}
DESCRIPTION = "TALENT/OpenML 45547 cardiovascular disease classification. Records outside fixed inclusive plausibility bounds (height 100-230 cm, weight 20-300 kg, systolic pressure 50-300 mmHg, diastolic pressure 20-200 mmHg), or with diastolic pressure greater than or equal to systolic pressure, are excluded without consulting labels or correcting values. Exact duplicate feature patterns are collapsed; patterns with conflicting labels are excluded. The retained rows use a stratified 70/15/15 train/validation/test split; original TALENT split boundaries are not retained."


def invalid_measurements(X):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 11:
        raise ValueError("Cardiovascular requires its 11 original TALENT features")
    height, weight, systolic, diastolic = X[:, 1:5].astype(float).T
    values = dict(zip(BOUNDS, (height, weight, systolic, diastolic)))
    flags = {
        name: ~np.isfinite(value)
        | (value < BOUNDS[name][0])
        | (value > BOUNDS[name][1])
        for name, value in values.items()
    }
    flags["diastolic_ge_systolic"] = diastolic >= systolic
    return flags


def clean_rows(X, y):
    X, y = (np.asarray(X), np.asarray(y).ravel())
    if len(X) != len(y):
        raise ValueError("Cardiovascular X/y row counts disagree")
    flags = invalid_measurements(X)
    invalid = np.logical_or.reduce(list(flags.values()))
    retained = np.flatnonzero(~invalid)
    frame = pd.DataFrame(X[retained])
    numeric = frame.astype(float)
    groups = numeric.groupby(list(numeric.columns), sort=False, dropna=False).ngroup()
    labelled = pd.DataFrame({"group": groups, "y": y[retained]})
    conflicting = labelled.groupby("group")["y"].nunique().gt(1)
    keep = (
        ~labelled["group"].isin(conflicting.index[conflicting]) & ~numeric.duplicated()
    )
    indices = retained[keep.to_numpy()]
    stats = {
        "rows_before": len(X),
        "flagged_rows": int(invalid.sum()),
        "rule_counts": {key: int(value.sum()) for key, value in flags.items()},
        "conflicting_patterns": int(conflicting.sum()),
        "rows_after": len(indices),
    }
    return (X[indices], y[indices], indices, stats)


def verify_values(X, y):
    path = CONFIG_DIR / "cardiovascular_plausibility.json"
    reference = json.loads(path.read_text())
    if reference["protocol"] != PROTOCOL:
        raise ValueError("Unexpected cardiovascular plausibility protocol")
    if np.asarray(X).shape != (reference["num_samples"], 11):
        raise ValueError(
            "Cardiovascular requires the current plausibility-filtered cohort"
        )
    if any((mask.any() for mask in invalid_measurements(X).values())):
        raise ValueError(
            "Cardiovascular still contains measurements outside the fixed rules"
        )
    if pd.DataFrame(X).astype(float).duplicated().any():
        raise ValueError("Cardiovascular still contains duplicate feature patterns")
    if (
        np.bincount(np.asarray(y).astype(int).ravel()).tolist()
        != reference["class_counts"]
    ):
        raise ValueError("Cardiovascular labels disagree with the filtered cohort")
