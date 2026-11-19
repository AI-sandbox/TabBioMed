"""Metafeatures analysis."""

from __future__ import annotations
from tabbiomed.analysis.paths import OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.downloads import ensure_results
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import tempfile
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tabbiomed-metafeatures-mpl")
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
from tabbiomed.analysis.datasets import (
    load_benchmark,
    data_identity,
    dataset_directory,
    load_arrays,
    load_metadata,
)
import numpy as np
import pandas as pd
from scipy.stats import entropy, friedmanchisquare, rankdata, spearmanr, wilcoxon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tabbiomed.analysis.results import BENCHMARK_PLAN, RESULTS, read_result
from tabbiomed.analysis.ranks import normalized_scores_with_failures
from tabbiomed.analysis.common import load_benchmark_models_config
from tabbiomed.analysis.clustering import (
    CORE_FEATURES,
    descriptor_signature,
    run_diagnostics,
)

FAMILIES = ["Baselines", "Tree-based", "Neural networks", "Foundation models"]
COLORS = ["#4C4C4C", "#0072B2", "#E69F00", "#CC79A7"]
TARGETS = FAMILIES + ["gap_foundation_baselines", "gap_foundation_tree"]
FOUNDATION_ADVANTAGE = "gap_foundation_nonfoundation"
CLUSTER_K = 3
COMPARISON_COLUMNS = {
    "Baselines": "gap_foundation_baselines",
    "Tree-based": "gap_foundation_tree",
    "Neural networks": "gap_foundation_neural",
}
PYMFE_FEATURES = [
    "nr_cor_attr",
    "nr_outliers",
    "nr_norm",
    "cor",
    "cov",
    "sparsity",
    "kurtosis",
    "skewness",
    "attr_ent",
    "class_ent",
    "mut_inf",
    "ns_ratio",
    "eq_num_attr",
    "sd_ratio",
    "gravity",
]
DEPENDENT = {"class_ent", "mut_inf", "ns_ratio", "eq_num_attr", "sd_ratio", "gravity"}
DEFAULT_OUTPUT = _OUTPUT_ROOT / "metafeatures"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def sample_rows(indices, labels, maximum, rng, classification):
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= maximum:
        return np.sort(indices)
    if not classification:
        return np.sort(rng.choice(indices, maximum, replace=False))
    (_, inverse, counts) = np.unique(
        labels[indices], return_inverse=True, return_counts=True
    )
    if len(counts) > maximum:
        raise ValueError("Extraction row cap cannot represent every training class")
    chosen = [rng.choice(indices[inverse == group]) for group in range(len(counts))]
    remaining = np.setdiff1d(indices, chosen)
    chosen.extend(rng.choice(remaining, maximum - len(chosen), replace=False))
    return np.sort(np.asarray(chosen, dtype=int))


def encode_training_sample(raw, categorical):
    result = np.empty(raw.shape, dtype=float)
    for column in range(raw.shape[1]):
        series = pd.Series(raw[:, column])
        if column in categorical:
            (codes, _) = pd.factorize(series, sort=False)
            observed = codes[codes >= 0]
            replacement = int(np.bincount(observed).argmax()) if observed.size else 0
            result[:, column] = np.where(codes < 0, replacement, codes)
        else:
            numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
            numeric[~np.isfinite(numeric)] = np.nan
            observed = numeric[np.isfinite(numeric)]
            replacement = float(np.median(observed)) if observed.size else 0.0
            result[:, column] = np.nan_to_num(numeric, nan=replacement)
    return result


def mixed_correlation(encoded, categorical, rng, maximum):
    blocks = []
    for column in range(encoded.shape[1]):
        values = encoded[:, column]
        if column in categorical:
            levels = np.unique(values)
            blocks.extend(((values == level).astype(float) for level in levels[1:]))
        else:
            blocks.append(values)
    if len(blocks) > maximum:
        blocks = [
            blocks[i] for i in np.sort(rng.choice(len(blocks), maximum, replace=False))
        ]
    if not blocks:
        return (0.0, 0)
    matrix = np.column_stack(blocks)
    matrix = matrix[:, np.std(matrix, axis=0) > 0]
    if matrix.shape[1] < 2:
        return (0.0, int(matrix.shape[1]))
    correlations = np.corrcoef(matrix, rowvar=False)
    return (
        float(np.abs(correlations[np.tril_indices(matrix.shape[1], -1)]).mean()),
        int(matrix.shape[1]),
    )


def extract_dataset(job):
    from pymfe.mfe import MFE

    (name, info, protocol, output) = job
    identity = data_identity(name)
    signature = canonical_hash({"identity": identity, "protocol": protocol})
    target = Path(output) / "extraction" / f"{name}.json"
    if target.exists():
        cached = json.loads(target.read_text())
        if cached.get("signature") == signature:
            return (name, True)
    path = dataset_directory(name, arrays=True)
    metadata = load_metadata(name)
    try:
        x = np.load(path / "X.npy", mmap_mode="r", allow_pickle=False)
    except ValueError:
        x = np.load(path / "X.npy", allow_pickle=True)
    y = np.load(path / "y.npy", allow_pickle=True)
    (n, p) = x.shape
    if (n, p) != (metadata["num_samples"], metadata["num_features"]):
        raise ValueError(f"Metadata dimensions differ from X for {name}")
    splits = np.load(path / "split_indices.npy", allow_pickle=True).item()
    folds = splits["cv_folds"] if info["n_folds"] else [splits]
    if len(folds) != (info["n_folds"] or 1):
        raise ValueError(f"Invalid fold count for {name}")
    classification = info["task"] in ("binary", "classification")
    seed = (
        protocol["seed"] + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    ) % 2**32
    source_columns = np.sort(
        np.random.default_rng(seed).choice(
            p, min(p, protocol["max_features"]), replace=False
        )
    )
    categorical = [
        i
        for (i, column) in enumerate(source_columns)
        if column in metadata.get("cat_idx", [])
    ]
    results = []
    for fold_id, fold in enumerate(folds):
        rng = np.random.default_rng((seed + fold_id + 1) % 2**32)
        rows = sample_rows(fold["train"], y, protocol["max_rows"], rng, classification)
        if np.intersect1d(rows, np.r_[fold.get("val", []), fold["test"]]).size:
            raise ValueError(f"Held-out rows entered extraction for {name}")
        raw = np.array(x[np.ix_(rows, source_columns)])
        encoded = encode_training_sample(raw, set(categorical))
        features = [f for f in PYMFE_FEATURES if classification or f not in DEPENDENT]
        numeric_p = encoded.shape[1] - len(categorical)
        covariance_reason = None
        if classification:
            (_, class_counts) = np.unique(y[rows], return_counts=True)
            if numeric_p == 0 or class_counts.min() <= numeric_p:
                features.remove("sd_ratio")
                covariance_reason = (
                    "within-class covariance cannot have full column rank"
                )
        mfe = MFE(
            groups=["general", "statistical", "info-theory"],
            features=features,
            summary=["nanmean"],
            random_state=protocol["seed"],
            suppress_warnings=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mfe.fit(
                encoded,
                y[rows] if classification else None,
                cat_cols=categorical,
                transform_cat=None,
                precomp_groups=None,
                suppress_warnings=True,
            )
            (names, values) = mfe.extract(suppress_warnings=True)
        values = {
            key.replace(".nanmean", ".mean"): finite(value)
            for (key, value) in zip(names, values)
        }
        (mixed, encoded_p) = mixed_correlation(
            encoded, set(categorical), rng, protocol["max_features"]
        )
        if classification and covariance_reason:
            values["sd_ratio"] = None
        values.update(
            {
                "mixed_cor.mean": mixed,
                "nr_outliers_fraction": finite(values.get("nr_outliers") / numeric_p)
                if values.get("nr_outliers") is not None and numeric_p
                else None,
                "nr_norm_fraction": finite(values.get("nr_norm") / numeric_p)
                if values.get("nr_norm") is not None and numeric_p
                else None,
                "sample_missing_fraction": float(pd.isna(raw).mean()),
                "sample_zero_fraction": float((encoded == 0).mean()),
                "sample_constant_fraction": float(
                    (np.std(encoded, axis=0) == 0).mean()
                ),
            }
        )
        if classification:
            (_, counts) = np.unique(
                y[np.asarray(fold["train"], dtype=int)], return_counts=True
            )
            values["training_class_entropy"] = float(entropy(counts, base=2))
        results.append(
            {
                "fold": fold_id,
                "n_training_rows": len(fold["train"]),
                "n_extraction_rows": len(rows),
                "n_source_columns": len(source_columns),
                "n_numeric_columns": numeric_p,
                "n_mixed_encoded_columns": encoded_p,
                "sample_rows": rows.tolist(),
                "source_columns": source_columns.tolist(),
                "sample_sha256": hashlib.sha256(
                    encoded.tobytes() + json.dumps(y[rows].tolist()).encode()
                ).hexdigest(),
                "warnings": sorted(set((str(w.message) for w in caught)))[:20],
                "sd_ratio_unavailable": covariance_reason,
                "values": values,
            }
        )
    if data_identity(name) != identity:
        raise ValueError(f"Dataset changed while extracting {name}")
    save_json(
        target,
        {
            "dataset": name,
            "task": info["task"],
            "signature": signature,
            "identity": identity,
            "num_samples": n,
            "num_features": p,
            "metadata": {
                "domain": metadata.get("label"),
                "num_categorical": len(metadata.get("cat_idx", [])),
                "num_classes": metadata.get("num_classes"),
                "imbalance_ratio": metadata.get("imbalance_ratio"),
            },
            "folds": results,
        },
    )
    return (name, False)


def assemble_metafeatures(output, datasets):
    (rows, fold_rows) = ([], [])
    for dataset in sorted(datasets):
        saved = json.loads((output / "extraction" / f"{dataset}.json").read_text())
        if saved["task"] in ("binary", "classification"):
            y = load_arrays(dataset)[1]
            for fold in saved["folds"]:
                if "sample_rows" not in fold:
                    continue
                (_, counts) = np.unique(y[fold["sample_rows"]], return_counts=True)
                if fold["n_numeric_columns"] == 0:
                    fold["values"]["gravity"] = None
                if (
                    fold["n_numeric_columns"] == 0
                    or counts.min() <= fold["n_numeric_columns"]
                ):
                    fold["values"]["sd_ratio"] = None
                    fold["sd_ratio_unavailable"] = (
                        "within-class covariance cannot have full column rank"
                    )
        (n, p) = (saved["num_samples"], saved["num_features"])
        row = {
            "dataset": dataset,
            "task": saved["task"],
            "nr_inst": n,
            "nr_attr": p,
            "attr_to_inst": p / n,
            "inst_to_attr": n / p,
            "log10_samples": np.log10(n),
            "log10_features": np.log10(p),
            "log10_attr_to_inst": np.log10(p / n),
            "cat_fraction": saved["metadata"]["num_categorical"] / p,
            "nr_class": saved["metadata"].get("num_classes")
            if saved["task"] in ("binary", "classification")
            else np.nan,
            "nr_cat": saved["metadata"]["num_categorical"],
            "nr_num": p - saved["metadata"]["num_categorical"],
            "imbalance_ratio": saved["metadata"].get("imbalance_ratio"),
            "domain": str(saved["metadata"]["domain"]).lower(),
            "n_outer_training_partitions": len(saved["folds"]),
        }
        keys = set().union(*(fold["values"] for fold in saved["folds"]))
        for key in keys:
            values = [fold["values"].get(key) for fold in saved["folds"]]
            row[key] = (
                float(np.mean(values))
                if all((v is not None for v in values))
                else np.nan
            )
        for fold in saved["folds"]:
            fold_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold["fold"],
                    "n_training_rows": fold["n_training_rows"],
                    "n_extraction_rows": fold["n_extraction_rows"],
                    "n_source_columns": fold["n_source_columns"],
                    "sd_ratio_unavailable": fold.get("sd_ratio_unavailable"),
                    **fold["values"],
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("dataset").sort_index()
    frame.to_csv(output / "metafeatures.csv")
    pd.DataFrame(fold_rows).to_csv(
        output / "metafeatures_by_training_partition.csv", index=False
    )
    return frame


def foundation_advantage(family_scores):
    alternatives = [name for name in FAMILIES if name != "Foundation models"]
    return (
        family_scores["Foundation models"] - family_scores[alternatives].max(axis=1)
    ).rename(FOUNDATION_ADVANTAGE)


def foundation_family_contrasts(family_scores):
    return pd.DataFrame(
        {
            column: family_scores["Foundation models"] - family_scores[family]
            for (family, column) in COMPARISON_COLUMNS.items()
        },
        index=family_scores.index,
    )


def load_scores(plan, output):
    config = load_benchmark_models_config()
    models = config.expected_models_regression
    jobs = [
        (model, dataset, info)
        for (dataset, info) in sorted(plan["datasets"].items())
        for model in models
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda job: read_result(*job), jobs))
    frame = pd.DataFrame(rows)
    frame["family"] = frame.model.map(config.model_family)
    if frame.family.isna().any():
        raise ValueError("Unassigned model family")
    raw = frame.pivot(index="dataset", columns="model", values="metric_value").reindex(
        columns=models
    )
    tasks = frame.groupby("dataset").task.first()
    normalized = normalized_scores_with_failures(raw, tasks)
    ranks = pd.DataFrame(
        rankdata(-normalized, axis=1), index=raw.index, columns=raw.columns
    )
    family = pd.DataFrame(
        {
            name: normalized[
                [model for model in models if config.model_family[model] == name]
            ].max(axis=1)
            for name in FAMILIES
        }
    )
    for column, values in foundation_family_contrasts(family).items():
        family[column] = values
    family[FOUNDATION_ADVANTAGE] = foundation_advantage(family)
    frame.to_csv(output / "selected_results.csv", index=False)
    normalized.to_csv(output / "normalized_scores.csv")
    ranks.to_csv(output / "model_ranks.csv")
    family.to_csv(output / "family_best_scores.csv")
    return (frame, family, ranks, config)


def correlations(meta, family):
    rows = []
    for feature in meta.select_dtypes(include="number").columns:
        if feature in (
            "n_outer_training_partitions",
            "sample_zero_fraction",
        ) or feature.startswith("log10_"):
            continue
        for target in TARGETS:
            valid = (
                pd.concat([meta[feature], family[target]], axis=1)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(valid) < 10 or valid.nunique().min() < 2:
                continue
            test = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": len(valid),
                    "rho": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            )
    frame = pd.DataFrame(rows)
    frame["q_bh"] = multipletests(frame.p_value, method="fdr_bh")[1]
    frame["p_holm"] = multipletests(frame.p_value, method="holm")[1]
    return frame


def family_tests(family):
    matrix = family[FAMILIES]
    statistic = friedmanchisquare(*(matrix[name] for name in FAMILIES))
    rows = []
    for a, b in itertools.combinations(FAMILIES, 2):
        difference = matrix[a] - matrix[b]
        p = (
            wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue
            if difference.ne(0).any()
            else 1.0
        )
        rows.append(
            {
                "family_a": a,
                "family_b": b,
                "mean_a_minus_b": difference.mean(),
                "n": len(difference),
                "p_value": float(p),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = multipletests(frame.p_value, method="holm")[1]
    return (
        {
            "n": len(matrix),
            "statistic": float(statistic.statistic),
            "p_value": float(statistic.pvalue),
        },
        frame,
    )


def cluster_regimes(meta, family, output, *, classification_only=False):
    name = (
        "classification_entropy" if classification_only else "all_tasks_unsupervised"
    )
    columns = list(CORE_FEATURES)
    if classification_only:
        columns.append("ns_ratio")
        meta = meta[meta.task.isin(["binary", "classification"])]
    inputs = meta[columns].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    z = scaler.fit_transform(inputs)
    fit = KMeans(n_clusters=CLUSTER_K, n_init=50, random_state=42).fit(z)
    old_order = sorted(
        range(CLUSTER_K), key=lambda label: z[fit.labels_ == label, 0].mean()
    )
    relabel = {old: i + 1 for (i, old) in enumerate(old_order)}
    assignments = pd.DataFrame(
        {"cluster": [relabel[x] for x in fit.labels_]}, index=inputs.index
    )
    assignments = assignments.join(meta[["task", "domain", "nr_inst", "nr_attr"]]).join(
        family
    )
    for column, values in foundation_family_contrasts(assignments).items():
        assignments[column] = values
    assignments.to_csv(output / f"clusters_{name}.csv")
    centers = pd.DataFrame(
        [fit.cluster_centers_[label] for label in old_order],
        columns=columns,
        index=pd.Index(range(1, CLUSTER_K + 1), name="cluster"),
    )
    centers.to_csv(output / f"cluster_profiles_{name}.csv")
    pd.DataFrame(z, columns=columns, index=inputs.index).to_csv(
        output / f"cluster_inputs_z_{name}.csv"
    )
    summaries = []
    for comparator, column in COMPARISON_COLUMNS.items():
        compared = assignments.groupby("cluster").agg(
            n=(column, "size"),
            samples_median=("nr_inst", "median"),
            features_median=("nr_attr", "median"),
            gap_mean=(column, "mean"),
            gap_median=(column, "median"),
            positive_gap_count=(column, lambda s: int((s > 1e-12).sum())),
            positive_gap_fraction=(column, lambda s: float((s > 1e-12).mean())),
            tied_gap_fraction=(column, lambda s: float((s.abs() <= 1e-12).mean())),
        )
        compared["comparison"] = comparator
        summaries.append(compared.reset_index())
    summary = (
        pd.concat(summaries, ignore_index=True)
        .set_index(["cluster", "comparison"])
        .sort_index()
    )
    summary.to_csv(output / f"cluster_summary_{name}.csv")
    pd.crosstab(assignments.cluster, assignments.domain).to_csv(
        output / f"cluster_domains_{name}.csv"
    )
    pd.crosstab(assignments.cluster, assignments.task).to_csv(
        output / f"cluster_tasks_{name}.csv"
    )
    save_json(
        output / f"cluster_method_{name}.json",
        {
            "columns": columns,
            "n": len(inputs),
            "n_clusters": CLUSTER_K,
            "n_init": 50,
            "seed": 42,
            "silhouette": float(silhouette_score(z, fit.labels_)),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "performance_contrasts": {
                family: f"best foundation normalized score minus best {family} normalized score"
                for family in COMPARISON_COLUMNS
            },
            "excluded": sorted(set(meta.index) - set(inputs.index)),
            "interpretation": "Exploratory. Cluster membership never uses model performance. The all-task analysis omits classification-only ns_ratio.",
        },
    )
    return (name, assignments, centers, summary)


def binary_difficulty(frame, meta, output):
    binary = frame[frame.task.eq("binary")]
    raw = (
        binary.groupby(["dataset", "family"])
        .metric_value.max()
        .unstack()
        .reindex(columns=FAMILIES)
    )
    raw["best_overall"] = raw[FAMILIES].max(axis=1)
    raw["gap_foundation_nonfoundation"] = raw["Foundation models"] - raw[
        FAMILIES[:-1]
    ].max(axis=1)
    denominator = 1 - raw["Baselines"]
    raw["relative_ranking_error_reduction"] = (
        raw["Foundation models"] - raw["Baselines"]
    ) / denominator.where(denominator > 0)
    raw["difficulty"] = pd.cut(
        raw.best_overall,
        [0, 0.75, 0.9, 1.000001],
        right=False,
        labels=["Hard", "Intermediate", "Easy"],
    )
    raw.join(meta[["domain"]]).to_csv(output / "binary_difficulty.csv")
    summaries = raw.groupby("difficulty", observed=False).agg(
        n=("best_overall", "size"),
        n_defined_reductions=("relative_ranking_error_reduction", "count"),
        mean_relative_reduction=("relative_ranking_error_reduction", "mean"),
        median_relative_reduction=("relative_ranking_error_reduction", "median"),
    )
    summaries.to_csv(output / "binary_difficulty_summary.csv")
    difference = raw.gap_foundation_nonfoundation
    stats = {
        "n": len(raw),
        "foundation_wins": int((difference > 1e-12).sum()),
        "ties": int((difference.abs() <= 1e-12).sum()),
        "foundation_losses": int((difference < -1e-12).sum()),
        "absolute_margin_at_most_001": int((difference.abs() <= 0.01).sum()),
        "interpretation": "Empirical best test ROC-AUC groups, not estimates of irreducible noise. Relative reduction is in 1-AUC, not classification error.",
    }
    valid = raw[["best_overall", "relative_ranking_error_reduction"]].dropna()
    test = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    stats["difficulty_vs_relative_reduction"] = {
        "n": len(valid),
        "rho": finite(test.statistic),
        "p_value": finite(test.pvalue),
        "caveat": "Exploratory and mathematically coupled, both variables use the same observed test scores.",
    }
    save_json(output / "binary_statistics.json", stats)
    return (raw, stats)


def plot_outputs(meta, family, corr, clusters, binary, ranks, config, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )

    def save(fig, stem):
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    (fig, axes) = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for ax, feature, label in zip(
        axes, ["nr_inst", "nr_attr"], ["Samples", "Features"]
    ):
        ordered = meta.sort_values(feature).index
        for name, color in zip(FAMILIES, COLORS):
            values = family.loc[ordered, name]
            ax.scatter(
                meta.loc[ordered, feature], values, color=color, s=13, alpha=0.35
            )
            bins = pd.qcut(meta.loc[ordered, feature], 6, duplicates="drop")
            grouped = pd.DataFrame(
                {"x": meta.loc[ordered, feature], "y": values, "bin": bins}
            ).groupby("bin", observed=True)
            ax.plot(
                grouped.x.median(),
                grouped.y.mean(),
                color=color,
                marker="o",
                label=name,
            )
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylim(-0.02, 1.04)
    axes[0].set_ylabel("Best observed normalized score within family")
    axes[1].legend(loc="lower left", fontsize=11)
    fig.tight_layout()
    save(fig, "01_family_performance_vs_size")
    features = [
        "nr_inst",
        "nr_attr",
        "attr_to_inst",
        "nr_class",
        "imbalance_ratio",
        "cat_fraction",
        "sample_missing_fraction",
        "mixed_cor.mean",
        "cor.mean",
        "cov.mean",
        "nr_cor_attr",
        "nr_norm_fraction",
        "nr_outliers_fraction",
        "sparsity.mean",
        "kurtosis.mean",
        "skewness.mean",
        "attr_ent.mean",
        "class_ent",
        "mut_inf.mean",
        "ns_ratio",
        "eq_num_attr",
        "sd_ratio",
        "gravity",
    ]
    values = corr.pivot(index="feature", columns="target", values="rho").reindex(
        index=features, columns=TARGETS
    )
    adjusted = corr.pivot(
        index="feature", columns="target", values="p_holm"
    ).reindex_like(values)
    annotations = values.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "")
    annotations = annotations + adjusted.map(
        lambda v: "*" if pd.notna(v) and v < 0.05 else ""
    )
    labels = {
        "nr_inst": "Samples",
        "nr_attr": "Features",
        "attr_to_inst": "Features / samples",
        "nr_class": "Classes",
        "imbalance_ratio": "Class imbalance ratio",
        "cat_fraction": "Categorical feature fraction",
        "sample_missing_fraction": "Missing-value fraction",
        "mixed_cor.mean": "Mean absolute correlation (mixed)",
        "cor.mean": "Mean absolute correlation (numeric)",
        "cov.mean": "Mean absolute covariance",
        "nr_cor_attr": "Highly correlated pair fraction",
        "nr_norm_fraction": "Normally distributed feature fraction",
        "nr_outliers_fraction": "Outlier-bearing feature fraction",
        "sparsity.mean": "Repeated-value sparsity",
        "kurtosis.mean": "Mean kurtosis",
        "skewness.mean": "Mean skewness",
        "attr_ent.mean": "Mean feature entropy",
        "class_ent": "Class entropy",
        "mut_inf.mean": "Mean feature–target mutual information",
        "ns_ratio": "Feature entropy / mutual information ratio",
        "eq_num_attr": "Equivalent number of attributes",
        "sd_ratio": "Covariance homogeneity",
        "gravity": "Class centroid distance",
    }
    counts = corr.groupby("feature").n.first()
    (fig, ax) = plt.subplots(figsize=(13, 11.5))
    sns.heatmap(
        values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=annotations,
        fmt="",
        ax=ax,
        cbar_kws={"label": "Spearman correlation"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(
        [f"{labels[feature]} (n={counts[feature]})" for feature in features], rotation=0
    )
    ax.set_xticklabels(
        [
            "Baselines",
            "Tree-based",
            "Neural",
            "Foundation",
            "Foundation − Baselines",
            "Foundation − Tree",
        ],
        rotation=35,
        ha="right",
    )
    ax.set_title("* Holm-adjusted p < 0.05 across the full correlation scan")
    fig.tight_layout()
    save(fig, "02_metafeature_correlations")
    for name, assignments, centers, _ in clusters:
        (fig, axes) = plt.subplots(2, 2, figsize=(11.5, 8.5))
        counts = assignments.groupby("cluster").size()
        profile = axes[0, 0]
        limit = float(np.abs(centers.to_numpy()).max())
        sns.heatmap(
            centers,
            cmap="RdBu_r",
            center=0,
            vmin=-limit,
            vmax=limit,
            annot=True,
            fmt=".1f",
            ax=profile,
            cbar_kws={"label": "Cluster mean (z-score)", "shrink": 0.8},
        )
        profile.set_xticklabels(
            [
                {
                    "log10_samples": "Log\nsamples",
                    "log10_attr_to_inst": "Log(features /\nsamples)",
                    "mixed_cor.mean": "Mean absolute\ncorrelation",
                    "ns_ratio": "Entropy /\nMI ratio",
                }[column]
                for column in centers.columns
            ],
            rotation=0,
            ha="center",
            fontsize=13 if len(centers.columns) == 3 else 11,
        )
        profile.set_yticklabels(
            [f"C{i} (n={counts.loc[i]})" for i in centers.index], rotation=0
        )
        profile.set_ylabel("")
        profile.set_title("$\\mathbf{A.}$ Dataset properties", loc="left", fontsize=16)
        contrasts = assignments[list(COMPARISON_COLUMNS.values())]
        (low, high) = (float(contrasts.min().min()), float(contrasts.max().max()))
        padding = max(0.04, 0.06 * (high - low))
        labels = {
            "Baselines": "classical",
            "Tree-based": "tree-based",
            "Neural networks": "neural",
        }
        for letter, (comparison_family, column), ax in zip(
            "BCD", COMPARISON_COLUMNS.items(), [axes[0, 1], axes[1, 0], axes[1, 1]]
        ):
            sns.boxplot(
                data=assignments.reset_index(),
                x="cluster",
                y=column,
                color=dict(zip(FAMILIES, COLORS))[comparison_family],
                ax=ax,
                showfliers=False,
                linewidth=1,
                saturation=0.65,
            )
            jitter = np.random.default_rng(42).uniform(-0.09, 0.09, len(assignments))
            ax.scatter(
                assignments.cluster.to_numpy() - 1 + jitter,
                assignments[column],
                color=".2",
                s=12,
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )
            ax.axhline(0, color=".35", linewidth=1, zorder=2)
            ax.set_ylim(low - padding, high + padding)
            ax.set_ylabel("Normalized score difference")
            ax.set_xlabel("Cluster")
            ax.set_xticks(
                range(len(centers)),
                [f"C{i}\n(n={counts.loc[i]})" for i in centers.index],
            )
            ax.set_title(
                f"$\\mathbf{{{letter}.}}$ Foundation − {labels[comparison_family]}",
                loc="left",
                fontsize=16,
            )
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=".90", linewidth=0.6)
        fig.tight_layout()
        save(fig, f"03_archetypes_{name}")
    melted = binary.reset_index().melt(
        id_vars=["dataset", "difficulty"],
        value_vars=FAMILIES,
        var_name="Family",
        value_name="ROC-AUC",
    )
    (fig, axes) = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.boxplot(
        data=melted,
        x="difficulty",
        y="ROC-AUC",
        hue="Family",
        palette=dict(zip(FAMILIES, COLORS)),
        ax=axes[0],
        fliersize=2,
    )
    axes[0].set_xlabel("Observed task difficulty")
    axes[0].legend(fontsize=10, loc="lower left")
    axes[1].scatter(
        binary.best_overall,
        binary.relative_ranking_error_reduction,
        s=25,
        color="#0072B2",
    )
    axes[1].axhline(0, color=".5", linewidth=0.8)
    axes[1].set_xlabel(f"Best observed ROC-AUC across {len(config.expected_models_regression)} models")
    axes[1].set_ylabel("Foundation vs baseline reduction in 1 − AUC")
    fig.tight_layout()
    save(fig, "04_binary_task_difficulty")
    domains = family[FAMILIES].join(meta.domain).groupby("domain").mean()
    domains.to_csv(output / "domain_family_best_means.csv")
    (fig, ax) = plt.subplots(figsize=(8, 5.3))
    sns.heatmap(domains, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0.5, vmax=1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    save(fig, "05_domain_family_performance")
    mean = ranks.mean().sort_values()
    (fig, ax) = plt.subplots(figsize=(8, 7))
    ax.barh(
        mean.index,
        mean,
        color=[dict(zip(FAMILIES, COLORS))[config.model_family[m]] for m in mean.index],
    )
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean rank across {len(ranks)} datasets")
    fig.tight_layout()
    save(fig, "06_model_mean_ranks")


def analyze(plan, protocol, output):
    meta = assemble_metafeatures(output, plan["datasets"])
    diagnostic_method = output / "clustering_diagnostic_method.json"
    if not diagnostic_method.exists() or json.loads(diagnostic_method.read_text()).get(
        "descriptor_signature"
    ) != descriptor_signature(meta):
        run_diagnostics(meta.sort_index(), output)
    diagnostic_table = pd.read_csv(output / "clustering_candidate_diagnostics.csv")
    reference = diagnostic_table[
        diagnostic_table.feature_set.eq("size_correlation")
        & diagnostic_table.algorithm.eq("kmeans")
    ]
    save_json(
        output / "clustering_selection.json",
        {
            "algorithm": "kmeans",
            "k": CLUSTER_K,
            "features": CORE_FEATURES,
            "model_performance_used": False,
            "diagnostics": reference[
                ["k", "silhouette", "ari_median", "ari_p10", "min_cluster_size"]
            ].to_dict(orient="records"),
            "rationale": "Retain the interpretable size/correlation descriptors. Three clusters provide higher subsampling stability and better separation than four, and more structural detail than the slightly better-separated but less stable two-cluster solution. Added descriptors did not justify replacing this simpler representation.",
        },
    )
    (frame, family, ranks, config) = load_scores(plan, output)
    corr = correlations(meta, family)
    corr.to_csv(output / "metafeature_correlations.csv", index=False)
    exact = corr[corr.feature.isin(["nr_inst", "nr_attr", "attr_to_inst"])].copy()
    exact.to_csv(output / "size_dimensionality_correlations.csv", index=False)
    (friedman, paired) = family_tests(family)
    save_json(output / "family_friedman.json", friedman)
    paired.to_csv(output / "family_wilcoxon_holm.csv", index=False)
    family_summary = pd.DataFrame(
        {
            "n_models": pd.Series(config.model_family)
            .loc[frame.model.unique()]
            .value_counts(),
            "mean_best_normalized": family[FAMILIES].mean(),
            "datasets_at_global_best": family[FAMILIES].eq(1).sum(),
        }
    ).loc[FAMILIES]
    family_summary.to_csv(output / "family_summary.csv")
    regimes = family.join(meta[["nr_inst", "nr_attr"]])
    regimes["regime"] = np.where(
        regimes.nr_attr > regimes.nr_inst, "features > samples", "features <= samples"
    )
    dimension_summary = regimes.groupby("regime")[FAMILIES].mean()
    dimension_summary.insert(0, "n", regimes.groupby("regime").size())
    dimension_summary["foundation_tree_gap_mean"] = regimes.groupby(
        "regime"
    ).gap_foundation_tree.mean()
    dimension_summary["foundation_tree_positive"] = regimes.groupby(
        "regime"
    ).gap_foundation_tree.apply(lambda s: int((s > 1e-12).sum()))
    dimension_summary["foundation_advantage_mean"] = regimes.groupby("regime")[
        FOUNDATION_ADVANTAGE
    ].mean()
    dimension_summary["foundation_advantage_positive"] = regimes.groupby("regime")[
        FOUNDATION_ADVANTAGE
    ].apply(lambda s: int((s > 1e-12).sum()))
    dimension_summary.to_csv(output / "dimension_regime_summary.csv")
    clusters = [
        cluster_regimes(meta, family, output),
        cluster_regimes(meta, family, output, classification_only=True),
    ]
    binary, _ = binary_difficulty(frame, meta, output)
    plot_outputs(meta, family, corr, clusters, binary, ranks, config, output)
    manifest = {
        "results_root": str(RESULTS.resolve()),
        "benchmark_manifest": str(BENCHMARK_PLAN),
        "benchmark_sha256": sha(BENCHMARK_PLAN),
        "script_sha256": sha(__file__),
        "models": sorted(frame.model.unique()),
        "n_models": frame.model.nunique(),
        "n_datasets": frame.dataset.nunique(),
        "selection_counts": frame.metric_source.value_counts().to_dict(),
        "protocol": protocol,
        "clustering_comparisons": list(COMPARISON_COLUMNS),
        "cluster_k": CLUSTER_K,
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "statsmodels",
                "pymfe",
                "matplotlib",
            ]
        },
    }
    save_json(output / "manifest.json", manifest)
    rho = corr[corr.feature.eq("nr_inst") & corr.target.isin(FAMILIES)].set_index(
        "target"
    )
    save_json(
        output / "verification.json",
        {
            "n_datasets": len(meta),
            "n_models": frame.model.nunique(),
            "n_pairs": len(frame),
            "n_training_partitions": sum(
                (5 if info["n_folds"] else 1 for info in plan["datasets"].values())
            ),
            "all_tasks_cluster_coverage": len(clusters[0][1]),
            "all_normalized_finite": bool(
                np.isfinite(pd.read_csv(output / "normalized_scores.csv", index_col=0))
                .all()
                .all()
            ),
            "report_hashes_verified_after_analysis": all(
                (
                    sha(row.report_path) == row.report_sha256
                    for row in frame.itertuples()
                )
            ),
            "all_tasks_metadata_split_array_identities_reverified": all(
                (
                    data_identity(name)
                    == json.loads((output / "extraction" / f"{name}.json").read_text())[
                        "identity"
                    ]
                    for name in plan["datasets"]
                )
            ),
        },
    )
    print(rho[["rho", "p_value", "q_bh"]].to_string(), flush=True)
    print(f"Saved full analysis to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=["all", "extract", "analyze"], default="all")
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Extract in reverse dataset order, sharing verified caches",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Extraction smoke test only, final analysis requires the configured dataset scope",
    )
    args = parser.parse_args()
    if args.stage != "extract":
        ensure_results()
    plan = load_benchmark(BENCHMARK_PLAN)
    protocol = {
        "version": 2,
        "seed": 42,
        "max_rows": args.max_rows,
        "max_features": args.max_features,
        "pymfe_version": importlib.metadata.version("pymfe"),
        "features": PYMFE_FEATURES,
        "summary": "nanmean within partition, all-fold finite mean across CV partitions",
        "partitions": "stored training only, all five outer training partitions for CV",
        "preprocessing": "sample-fitted median numerical and modal categorical imputation",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "extraction").mkdir(exist_ok=True)
    if args.stage in ("all", "extract"):
        datasets = args.datasets or sorted(plan["datasets"])
        if args.reverse:
            datasets = list(reversed(datasets))
        jobs = [
            (name, plan["datasets"][name], protocol, str(args.output))
            for name in datasets
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(extract_dataset, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                (name, reused) = future.result()
                print(
                    f"{i}/{len(jobs)} {name} {('verified cache' if reused else 'extracted')}",
                    flush=True,
                )
    if args.stage in ("all", "analyze"):
        for name in plan["datasets"]:
            saved = json.loads(
                (args.output / "extraction" / f"{name}.json").read_text()
            )
            if saved["signature"] != canonical_hash(
                {"identity": data_identity(name), "protocol": protocol}
            ):
                raise ValueError(f"Extraction settings or identity differ for {name}")
        analyze(plan, protocol, args.output)


if __name__ == "__main__":
    main()
