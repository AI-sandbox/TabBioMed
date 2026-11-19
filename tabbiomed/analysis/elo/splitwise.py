import numpy as np
import pandas as pd
from .fitting import fit_coefficients
from .missing import REFERENCE


_HIGHER_IS_BETTER_METRICS = frozenset(
    {"roc_auc", "auc", "pr_auc", "average_precision", "accuracy", "f1", "r2"}
)


def _orient_scores(frame: pd.DataFrame) -> np.ndarray:
    if "higher_is_better" in frame.columns:
        maximize = frame["higher_is_better"].astype(bool).to_numpy()
    elif "metric" in frame.columns:
        maximize = (
            frame["metric"]
            .astype(str)
            .str.lower()
            .isin(_HIGHER_IS_BETTER_METRICS)
            .to_numpy()
        )
    elif "primary_metric" in frame.columns:
        maximize = (
            frame["primary_metric"]
            .astype(str)
            .str.lower()
            .isin(_HIGHER_IS_BETTER_METRICS)
            .to_numpy()
        )
    else:
        maximize = frame.task.eq("binary").to_numpy()
    values = frame.metric_value.to_numpy(dtype=float)
    return np.where(maximize, values, -values)


def battle_statistics(frame):
    if frame.empty or frame.metric_value.isna().any():
        raise ValueError("Elo requires finite observed or explicitly imputed results")
    frame = frame.copy()
    frame["score"] = _orient_scores(frame)
    names = sorted(frame.competitor.unique())
    wide = frame.pivot(
        index=["dataset", "fold"], columns="competitor", values="score"
    ).reindex(columns=names)
    datasets = sorted(frame.dataset.unique())
    task_index = {name: i for i, name in enumerate(datasets)}
    task_ids = np.array([task_index[d] for d in wide.index.get_level_values("dataset")])
    counts = np.bincount(task_ids, minlength=len(datasets))
    weights = 1.0 / counts[task_ids]
    matrix = wide.to_numpy()
    n = len(names)
    if n < 2:
        raise ValueError("Elo requires at least two competitors")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    design = np.zeros((2 * len(pairs), n))
    labels = np.tile([1, 0], len(pairs))
    per_dataset = np.zeros((len(datasets), 2 * len(pairs)))
    for pair, (i, j) in enumerate(pairs):
        a, b = (matrix[:, i], matrix[:, j])
        valid = np.isfinite(a) & np.isfinite(b)
        ties = valid & (a == b)
        design[2 * pair : 2 * pair + 2, i] = np.log(10)
        design[2 * pair : 2 * pair + 2, j] = -np.log(10)
        for column, wins in (
            (2 * pair, valid & (a > b)),
            (2 * pair + 1, valid & (a < b)),
        ):
            per_dataset[:, column] = np.bincount(
                task_ids,
                weights=weights * (wins.astype(float) + 0.5 * ties),
                minlength=len(datasets),
            )
    if per_dataset.sum() <= 0:
        raise ValueError("No comparable result pairs")
    return (names, datasets, design, labels, per_dataset)


def fit_elo(design, labels, counts):
    coefficients = fit_coefficients(design, labels, 2 * counts)
    return 1000 + 400 * coefficients


def rate(frame, bootstrap=200, seed=42, reference=REFERENCE, progress=None):
    if not isinstance(bootstrap, int) or isinstance(bootstrap, bool) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer")
    names, datasets, design, labels, statistics = battle_statistics(frame)
    if reference not in names:
        raise ValueError("The default Random Forest reference is absent")
    point = fit_elo(design, labels, statistics.sum(axis=0))
    if bootstrap > 0:
        samples = np.empty((bootstrap, len(names)))
        rng = np.random.default_rng(seed)
        for i in range(bootstrap):
            indices = rng.integers(0, len(datasets), len(datasets))
            samples[i] = fit_elo(design, labels, statistics[indices].sum(axis=0))
            if progress and (i + 1) % 50 == 0:
                progress(i + 1, bootstrap)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
    else:
        samples = None
        lo, hi = point.copy(), point.copy()
    offset = 1000 - point[names.index(reference)]
    table = pd.DataFrame(
        dict(
            competitor=names,
            elo=point + offset,
            elo_lo_95=lo + offset,
            elo_hi_95=hi + offset,
        )
    )
    gap_rows = []
    lookup = {name: i for i, name in enumerate(names)}
    for name in names:
        if name.endswith("::tuned"):
            model = name.split("::")[0]
            default = f"{model}::default"
            if default in lookup:
                a, b = (lookup[name], lookup[default])
                diff = float(point[a] - point[b])
                if samples is not None:
                    lower, upper = np.percentile(
                        samples[:, a] - samples[:, b], [2.5, 97.5]
                    )
                else:
                    lower, upper = diff, diff
                gap_rows.append(
                    dict(
                        model=model,
                        tuned_minus_default=diff,
                        gap_lo_95=lower,
                        gap_hi_95=upper,
                    )
                )
    return (
        table,
        pd.DataFrame(gap_rows),
        dict(
            n_datasets=len(datasets),
            n_competitors=len(names),
            bootstrap=bootstrap,
            seed=seed,
            split_weights="1 / number of outer splits per dataset",
            ties="half a win each",
            solver_tolerance=1e-10,
            calibration="RandomForest default = 1000",
            confidence_intervals="Dataset bootstrap, raw uncertainty with point-fit calibration offset",
        ),
    )
