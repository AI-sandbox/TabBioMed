import numpy as np
import pandas as pd

REFERENCE = "RandomForest::default"
MISSING_POLICIES = (
    "observed",
    "exclude_incomplete",
    "complete_case",
    "rf_imputed",
    "complete_case_relaxed",
)


def apply_missing_policy(frame, policy, reference=REFERENCE, required_models=None):
    if policy not in MISSING_POLICIES:
        raise ValueError(f"Unknown missing policy: {policy}")
    frame = frame.copy()
    if frame.duplicated(["dataset", "fold", "competitor"]).any():
        raise ValueError("Duplicate model/split result")
    frame["metric_value"] = pd.to_numeric(frame.metric_value, errors="raise")
    frame.loc[~np.isfinite(frame.metric_value), "metric_value"] = np.nan
    frame["imputed"] = False
    info = dict(
        policy=policy,
        input_cells=len(frame),
        missing_input_cells=int(frame.metric_value.isna().sum()),
        excluded_variants=[],
        excluded_datasets=[],
        imputed_cells=0,
    )
    if policy == "observed":
        incomplete = frame.groupby(["competitor", "dataset"]).metric_value.transform(
            lambda x: x.isna().any()
        )
        frame = frame[~incomplete].copy()
    elif policy == "exclude_incomplete":
        incomplete = frame.groupby("competitor").metric_value.apply(
            lambda x: x.isna().any()
        )
        info["excluded_variants"] = list(incomplete.index[incomplete])
        frame = frame[~frame.competitor.isin(info["excluded_variants"])].copy()
    elif policy == "complete_case":
        incomplete = frame.groupby("dataset").metric_value.apply(
            lambda x: x.isna().any()
        )
        info["excluded_datasets"] = list(incomplete.index[incomplete])
        frame = frame[~frame.dataset.isin(info["excluded_datasets"])].copy()
    elif policy == "complete_case_relaxed":
        if "model" not in frame:
            frame["model"] = frame.competitor.str.split("::").str[0]
        required = set(
            frame.model.unique() if required_models is None else required_models
        )
        available = frame[
            frame.metric_value.notna() & frame.model.isin(required)
        ].copy()
        coverage = (
            available.groupby("dataset")
            .model.nunique()
            .reindex(frame.dataset.unique(), fill_value=0)
        )
        info["excluded_datasets"] = sorted(coverage.index[coverage.lt(len(required))])
        info["required_models"] = sorted(required)
        info["dataset_rule"] = (
            "At least one observed Default or Tuned outer-test outcome for every required model"
        )
        frame = available[~available.dataset.isin(info["excluded_datasets"])].copy()
    else:
        baseline = (
            frame[frame.competitor.eq(reference)]
            .set_index(["dataset", "fold"])
            .metric_value
        )
        for index in frame.index[frame.metric_value.isna()]:
            key = (frame.at[index, "dataset"], frame.at[index, "fold"])
            value = baseline.get(key)
            if value is None or not np.isfinite(value):
                raise ValueError(
                    f"Cannot RF-impute {key}: same-split default Random Forest is unavailable"
                )
            frame.at[index, "metric_value"] = value
            frame.at[index, "imputed"] = True
        wholly_imputed = frame.groupby("competitor").imputed.all()
        info["excluded_variants"] = list(wholly_imputed.index[wholly_imputed])
        frame = frame[~frame.competitor.isin(info["excluded_variants"])].copy()
    info.update(
        output_cells=len(frame),
        imputed_cells=int(frame.imputed.sum()),
        n_datasets=frame.dataset.nunique(),
        n_competitors=frame.competitor.nunique(),
    )
    return (frame, info)
