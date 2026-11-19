from __future__ import annotations
import hashlib
import json
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

CORE_FEATURES = ["log10_samples", "log10_attr_to_inst", "mixed_cor.mean"]
FEATURE_SETS = {
    "size_only": CORE_FEATURES[:2],
    "size_correlation": CORE_FEATURES,
    "with_missingness": CORE_FEATURES + ["sample_missing_fraction"],
    "with_categorical": CORE_FEATURES + ["cat_fraction"],
    "with_missingness_categorical": CORE_FEATURES
    + ["sample_missing_fraction", "cat_fraction"],
    "with_sparsity": CORE_FEATURES + ["sparsity.mean"],
}


def descriptor_signature(meta):
    columns = sorted(set().union(*FEATURE_SETS.values()))
    content = meta[columns].sort_index().to_json(orient="split", double_precision=12)
    return hashlib.sha256(content.encode()).hexdigest()


def fit_labels(values, k, algorithm, seed=42, n_init=50):
    if algorithm == "kmeans":
        return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(
            values
        )
    if algorithm == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(values)
    raise ValueError(algorithm)


def run_diagnostics(meta, output, *, repeats=50, seed=42):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    subsets = [
        np.sort(rng.choice(len(meta), int(0.8 * len(meta)), replace=False))
        for _ in range(repeats)
    ]
    rows, stability = ([], [])
    for name, columns in FEATURE_SETS.items():
        raw = meta[columns].to_numpy(dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError(
                f"All-task cluster descriptors must be defined for every dataset: {name}"
            )
        z = StandardScaler().fit_transform(raw)
        for algorithm in ("kmeans", "ward"):
            for k in range(2, 9):
                labels = fit_labels(z, k, algorithm, seed=seed)
                aris = []
                for repeat, indices in enumerate(subsets):
                    sampled = StandardScaler().fit_transform(raw[indices])
                    sampled_labels = fit_labels(
                        sampled, k, algorithm, seed=seed + repeat, n_init=10
                    )
                    value = adjusted_rand_score(labels[indices], sampled_labels)
                    aris.append(value)
                    stability.append(
                        dict(
                            feature_set=name,
                            algorithm=algorithm,
                            k=k,
                            repeat=repeat,
                            ari=value,
                        )
                    )
                row = dict(
                    feature_set=name,
                    algorithm=algorithm,
                    k=k,
                    n=len(raw),
                    min_cluster_size=int(np.bincount(labels).min()),
                    sizes=",".join(map(str, sorted(np.bincount(labels)))),
                    silhouette=float(silhouette_score(z, labels)),
                    ari_mean=float(np.mean(aris)),
                    ari_median=float(np.median(aris)),
                    ari_p10=float(np.quantile(aris, 0.1)),
                )
                rows.append(row)
            print(f"Checked {name}, {algorithm}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "clustering_candidate_diagnostics.csv", index=False)
    pd.DataFrame(stability).to_csv(
        output / "clustering_subsample_stability.csv", index=False
    )
    (output / "clustering_diagnostic_method.json").write_text(
        json.dumps(
            {
                "n_datasets": len(meta),
                "feature_sets": FEATURE_SETS,
                "k_values": list(range(2, 9)),
                "algorithms": ["kmeans", "ward"],
                "seed": seed,
                "repeats": repeats,
                "subsample_fraction": 0.8,
                "n_init_reference": 50,
                "n_init_subsample": 10,
                "scaling": "StandardScaler refitted separately in each 80% subsample",
                "stability": "ARI between the full partition restricted to sampled rows and the refitted subsample partition",
                "model_performance_used": False,
                "descriptor_signature": descriptor_signature(meta),
                "interpretation": "Compare silhouette within a fixed descriptor space. Scores across different descriptor sets are not directly comparable. Examine minimum group size and subsampling stability before adding descriptors or clusters.",
            },
            indent=2,
        )
        + "\n"
    )
    return table
