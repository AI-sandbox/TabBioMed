from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import json
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from tabbiomed.data.curation.gastrulation_datasets import digest, read_column

REGISTRY = CONFIG_DIR / "singlecell_sources.json"
PROTOCOL = "singlecell_source_annotations_cell_split_v1"
NAMES = ("sc_muris_age", "sc_muris_celltype", "sc_muraro_celltype")
DESCRIPTIONS = {
    "sc_muris_age": "Classify recorded mouse age (3, 18 or 24 months) from kidney Smart-seq2 expression in Tabula Muris Senis.",
    "sc_muris_celltype": "Classify nine released kidney cell-type annotations from Tabula Muris Senis Smart-seq2 expression.",
    "sc_muraro_celltype": "Classify eight released human pancreas cell-type annotations from Muraro et al. CEL-Seq2 expression; exclude the three epsilon cells.",
}


def registry():
    return json.loads(REGISTRY.read_text())


def stratified_split(labels, seed=42):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    holdout = np.floor(counts * 0.15).astype(int)
    remaining = round(len(labels) * 0.15) - holdout.sum()
    order = np.argsort(-(counts * 0.15 - holdout), kind="stable")
    holdout[order[:remaining]] += 1
    if (holdout < 1).any() or (counts - 2 * holdout < 1).any():
        raise ValueError("Every class must occur in all three partitions")
    rng = np.random.default_rng(seed)
    splits = {role: [] for role in ("train", "val", "test")}
    for label, n in zip(classes, holdout):
        rows = rng.permutation(np.flatnonzero(labels == label))
        splits["val"].extend(rows[:n].tolist())
        splits["test"].extend(rows[n : 2 * n].tolist())
        splits["train"].extend(rows[2 * n :].tolist())
    return {role: sorted(rows) for role, rows in splits.items()}


def csr(node):
    shape = node.attrs["shape"] if "shape" in node.attrs else node["shape"][:]
    return sparse.csr_matrix(
        (node["data"][:], node["indices"][:], node["indptr"][:]), shape=shape
    )


def load_source(raw_root, name):
    ref = registry()["sources"][
        "pancreas" if name == "sc_muraro_celltype" else "kidney"
    ]
    path = Path(raw_root) / ref["filename"]
    if digest(path) != ref["sha256"]:
        raise ValueError(f"Source hash mismatch: {path}")
    with h5py.File(path) as f:
        if name == "sc_muraro_celltype":
            X, genes = (csr(f["exprs"]), f["var_names"].asstr()[:])
            obs = pd.DataFrame(
                {
                    "cell_id": f["obs_names"].asstr()[:],
                    "donor_id": f["obs/donor"].asstr()[:],
                    "batch": f["obs/batch"].asstr()[:],
                    "target": f["obs/cell_type1"].asstr()[:],
                }
            )
        else:
            X = csr(f["raw/X"])
            genes = read_column(
                f["raw/var"], f["raw/var"].attrs.get("_index", "_index")
            )
            obs = pd.DataFrame(
                {
                    "cell_id": read_column(
                        f["obs"], f["obs"].attrs.get("_index", "index")
                    ),
                    "donor_id": read_column(f["obs"], "donor_id"),
                    "age": read_column(f["obs"], "age"),
                    "cell_type": read_column(f["obs"], "cell_type"),
                    "sex": read_column(f["obs"], "sex"),
                }
            )
            obs["target"] = obs["age" if name == "sc_muris_age" else "cell_type"]
            if not np.equal(X.data, np.floor(X.data)).all():
                raise ValueError(
                    "Kidney raw/X must contain unnormalized integer counts"
                )
    obs.insert(0, "source_row", np.arange(len(obs)))
    if name == "sc_muraro_celltype":
        keep = obs.target.ne("epsilon").to_numpy()
        if (~keep).sum() != 3:
            raise ValueError("Unexpected epsilon-cell count")
        X, obs = (X[keep], obs.loc[keep].reset_index(drop=True))
    if obs.cell_id.duplicated().any() or len(set(genes)) != len(genes):
        raise ValueError("Source cell/gene identities must be unique")
    if (
        not np.isfinite(X.data).all()
        or (X.data < 0).any()
        or (np.asarray(X.sum(axis=1)) <= 0).any()
    ):
        raise ValueError("Invalid source expression")
    return (X, genes, obs, ref)


def nonconstant_training_genes(X, train):
    training = X[train].tocsc()
    return np.flatnonzero(
        (training.max(axis=0).toarray() - training.min(axis=0).toarray()).ravel() > 0
    )


def ingest(raw_root, output_root, name, *, force=False):
    if name not in NAMES:
        raise ValueError(name)
    target = Path(output_root) / name
    if target.exists() and (not force):
        verify_prepared(target)
        return target
    counts, genes, obs, ref = load_source(raw_root, name)
    classes = sorted(obs.target.unique())
    y = np.array([classes.index(label) for label in obs.target], dtype=np.int64)
    splits = stratified_split(y)
    selected = nonconstant_training_genes(counts, splits["train"])
    X = counts[:, selected].toarray().astype(np.float32)
    obs.insert(0, "row_index", np.arange(len(obs)))
    for role, rows in splits.items():
        obs.loc[rows, "partition"] = role
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "X.npy", X)
    np.save(target / "y.npy", y)
    np.save(target / "split_indices.npy", splits)
    obs.to_csv(target / "sample_metadata.csv", index=False)
    pd.DataFrame(
        {
            "feature_index": np.arange(len(selected)),
            "source_gene_index": selected,
            "gene": genes[selected],
        }
    ).to_csv(target / "feature_metadata.csv", index=False)
    (target / "feature_names.json").write_text(
        json.dumps(genes[selected].tolist(), indent=2) + "\n"
    )
    md = dict(
        name=name,
        task="classification",
        num_classes=len(classes),
        classes=classes,
        num_samples=len(y),
        num_features=X.shape[1],
        cat_idx=[],
        label="Single-Cell",
        sub_labels=["RNA-seq"],
        description=DESCRIPTIONS[name],
        source=ref["citation"],
        source_url=ref["url"],
        publication_doi=ref["doi"],
        split_strategy="fixed",
        split_unit="cell",
        curation_protocol=PROTOCOL,
        source_manifest_sha256=digest(REGISTRY),
        feature_representation=ref["representation"],
        feature_selection="training_nonconstant_genes",
        sample_metadata_file="sample_metadata.csv",
        feature_metadata_file="feature_metadata.csv",
        READY=True,
    )
    (target / "metadata.json").write_text(json.dumps(md, indent=2) + "\n")
    return target


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    md = json.loads((root / "metadata.json").read_text())
    name = md["name"]
    ref = registry()["datasets"][name]
    if md.get("curation_protocol") != PROTOCOL or md.get(
        "source_manifest_sha256"
    ) != digest(REGISTRY):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )
    for filename, expected in ref["prepared_sha256"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"{name}: changed prepared {filename}")
    if X is None:
        X = np.load(root / "X.npy", mmap_mode="r")
    if y is None:
        y = np.load(root / "y.npy")
    if (
        list(X.shape) != ref["shape"]
        or len(y) != X.shape[0]
        or (not np.isfinite(X).all())
    ):
        raise ValueError("Invalid source arrays")
    if (
        md.get("task") != "classification"
        or md.get("num_classes") != len(ref["classes"])
        or md.get("classes") != ref["classes"]
        or (md.get("cat_idx") != [])
        or ([md.get("num_samples"), md.get("num_features")] != ref["shape"])
        or (md.get("split_strategy") != "fixed")
        or (md.get("split_unit") != "cell")
    ):
        raise ValueError("Invalid class metadata")
    samples = pd.read_csv(root / "sample_metadata.csv")
    if samples.cell_id.duplicated().any() or not np.array_equal(
        samples.row_index, np.arange(len(y))
    ):
        raise ValueError("Invalid sample identity/order")
    if not np.array_equal(samples.target, np.asarray(ref["classes"])[y]):
        raise ValueError("Labels disagree with source sample annotations")
    splits = np.load(root / "split_indices.npy", allow_pickle=True).item()
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Unexpected split protocol")
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(splits.values()))), np.arange(len(y))
    )
    for role, rows in splits.items():
        if (
            len(rows) != ref["split_sizes"][role]
            or set(y[rows]) != set(range(len(ref["classes"])))
            or (not samples.iloc[rows].partition.eq(role).all())
        ):
            raise ValueError(f"Invalid {role} partition")
    return splits
