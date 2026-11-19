from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import argparse
import hashlib
import json
import shutil
import urllib.request
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path.cwd()
NAME = "sc_gastrulation_stage"
PROTOCOL = "gastrulation_day_pool_holdout_v1"
REGISTRY = CONFIG_DIR / "gastrulation_stage.json"
SAMPLE_MAP = CONFIG_DIR / "gastrulation_atlas_samples.csv"
SOURCE_URL = "https://ndownloader.figshare.com/files/28095525"
DESCRIPTION = "Mouse gastrulation collection-day regression (Pijuan-Sala et al., 2019): predict recorded embryonic day E6.5-E8.5 from expression, not inferred pseudotime. 10,000 cells with embryo-pool-disjoint 7,000/1,500/1,500 train/validation/test partitions. Select 10,000 nonconstant genes by analytic Pearson residual variance on training cells only. Sample, pool, stage and cell-type annotations are not predictors. Stages with too few independent pools cannot occur in every partition."


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def registry():
    return json.loads(REGISTRY.read_text())


def fetch_source(raw_root):
    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "gastrulation.h5ad"
    if not p.exists():
        temp = p.with_suffix(".download")
        try:
            with (
                urllib.request.urlopen(SOURCE_URL, timeout=180) as response,
                temp.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if digest(temp) != registry()["source_sha256"]:
                raise ValueError("Unexpected gastrulation source bytes")
            temp.replace(p)
        finally:
            temp.unlink(missing_ok=True)
    if digest(p) != registry()["source_sha256"]:
        raise ValueError("Gastrulation source hash mismatch")
    return p


def read_column(group, key):
    obj = group[key]
    if isinstance(obj, h5py.Group):
        cats = obj["categories"].asstr()[:]
        codes = obj["codes"][:]
        if (codes < 0).any():
            raise ValueError(f"Missing source metadata: {key}")
        return cats[codes]
    if "__categories" in group and key in group["__categories"]:
        categories = group["__categories"][key]
        cats = categories.asstr()[:] if categories.dtype.kind in "OS" else categories[:]
        codes = obj[:]
        if (codes < 0).any():
            raise ValueError(f"Missing source metadata: {key}")
        return cats[codes]
    return obj.asstr()[:] if obj.dtype.kind in "OS" else obj[:]


def pearson_select(training_counts, n_features=10000, block_size=256):
    counts = sparse.csc_matrix(training_counts, dtype=np.float64)
    if (
        not np.isfinite(counts.data).all()
        or (counts.data < 0).any()
        or (not np.equal(counts.data, np.floor(counts.data)).all())
    ):
        raise ValueError("Pearson selection requires finite nonnegative UMI counts")
    n, p = counts.shape
    row_sum = np.asarray(counts.sum(axis=1)).ravel()
    col_sum = np.asarray(counts.sum(axis=0)).ravel()
    if row_sum.min() <= 0:
        raise ValueError("Zero-count training cell")
    total = row_sum.sum()
    variances = np.full(p, -np.inf)
    clip = np.sqrt(n)
    for start in range(0, p, block_size):
        end = min(start + block_size, p)
        observed = counts[:, start:end].toarray()
        mu = row_sum[:, None] * (col_sum[start:end] / total)[None, :]
        denominator = np.sqrt(mu + mu**2 / 100.0)
        residual = np.divide(
            observed - mu, denominator, out=np.zeros_like(mu), where=denominator > 0
        )
        residual = np.clip(residual, -clip, clip)
        values = residual.var(axis=0)
        values[np.ptp(observed, axis=0) == 0] = -np.inf
        variances[start:end] = values
    if np.isfinite(variances).sum() < n_features:
        raise ValueError("Not enough nonconstant training genes")
    selected = np.sort(np.argsort(-variances, kind="stable")[:n_features])
    return (selected, variances)


def choose_cells(obs, seed=42):
    rng = np.random.default_rng(seed)
    roles = {"train": [], "val": [], "test": []}
    two_pool_count = 0
    for stage in sorted(obs.embryonic_day.unique()):
        pools = np.sort(obs.loc[obs.embryonic_day == stage, "pool_index"].unique())
        pools = rng.permutation(pools)
        if len(pools) >= 3:
            roles["val"].append(int(pools[0]))
            roles["test"].append(int(pools[1]))
            roles["train"].extend(map(int, pools[2:]))
        elif len(pools) == 2:
            roles["train"].append(int(pools[0]))
            roles[("val", "test")[two_pool_count % 2]].append(int(pools[1]))
            two_pool_count += 1
        else:
            roles["train"].append(int(pools[0]))
    selected, splits, start = ([], {}, 0)
    for role, count in [("train", 7000), ("val", 1500), ("test", 1500)]:
        candidates = np.flatnonzero(obs.pool_index.isin(roles[role]))
        if len(candidates) < count:
            raise ValueError(f"Insufficient cells in independent {role} pools")
        rows = np.sort(rng.choice(candidates, size=count, replace=False))
        selected.extend(rows.tolist())
        splits[role] = list(range(start, start + count))
        start += count
    samples = obs.iloc[selected].reset_index(drop=True).copy()
    samples.insert(0, "row_index", np.arange(len(samples)))
    samples["partition"] = np.repeat(["train", "val", "test"], [7000, 1500, 1500])
    if set(samples.embryonic_day) != set(samples.iloc[splits["train"]].embryonic_day):
        raise ValueError("Training must contain every selected stage")
    return (samples, splits, roles)


def load_source_metadata(path):
    with h5py.File(path) as f:
        if list(f["X"].attrs["shape"]) != registry()["source_shape"]:
            raise ValueError("Unexpected source dimensions")
        obs = f["obs"]
        if read_column(obs, "doublet").any() or read_column(obs, "stripped").any():
            raise ValueError("Unexpected doublet or stripped-nucleus source records")
        cells = read_column(obs, obs.attrs.get("_index", "_index"))
        stage = read_column(obs, "stage")
        sample = read_column(obs, "sample").astype(int)
        frame = pd.DataFrame(
            {
                "source_row": np.arange(len(cells)),
                "cell_id": cells,
                "sample": sample,
                "stage": stage,
            }
        )
    atlas = pd.read_csv(SAMPLE_MAP)
    frame = frame.merge(
        atlas.rename(columns={"stage": "source_stage"}),
        on="sample",
        how="left",
        validate="many_to_one",
    )
    if frame.cell_id.duplicated().any() or frame.pool_index.isna().any():
        raise ValueError("Nonunique cells or missing source sample identity")
    if not frame.stage.eq(frame.source_stage).all():
        raise ValueError("Cell stage disagrees with independent atlas sample metadata")
    frame = frame.loc[
        frame.stage.str.fullmatch(
            "E(?:6\\.5|6\\.75|7\\.0|7\\.25|7\\.5|7\\.75|8\\.0|8\\.25|8\\.5)"
        )
    ].copy()
    frame["embryonic_day"] = frame.stage.str[1:].astype(float)
    return frame.reset_index(drop=True)


def read_source_rows(path, indices):
    with h5py.File(path) as f:
        node = f["X"]
        shape = tuple(node.attrs["shape"])
        indptr = node["indptr"][:]
        data, columns, offsets = ([], [], [0])
        for row in indices:
            lo, hi = indptr[row : row + 2]
            data.append(node["data"][lo:hi])
            columns.append(node["indices"][lo:hi])
            offsets.append(offsets[-1] + hi - lo)
        X = sparse.csr_matrix(
            (np.concatenate(data), np.concatenate(columns), offsets),
            shape=(len(indices), shape[1]),
        )
        genes = read_column(f["var"], f["var"].attrs.get("_index", "_index"))
    return (X, genes)


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    ref = registry()
    md = json.loads((root / "metadata.json").read_text())
    if (
        md.get("curation_protocol") != PROTOCOL
        or md.get("task") != "regression"
        or md.get("source_manifest_sha256") != digest(REGISTRY)
    ):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )
    for filename, expected in ref["prepared_sha256"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"Gastrulation identity mismatch: {filename}")
    if X is not None:
        if (
            X.shape != (10000, 10000)
            or y.shape != (10000,)
            or (not np.isfinite(X).all())
            or (not np.isfinite(y).all())
        ):
            raise ValueError("Unexpected gastrulation arrays")
        if not set(np.unique(y)) <= set(np.arange(6.5, 8.51, 0.25)):
            raise ValueError("Expected embryonic days, not encoded classes")
    splits = np.load(root / "split_indices.npy", allow_pickle=True).item()
    samples = pd.read_csv(root / "sample_metadata.csv")
    if {role: len(splits[role]) for role in ("train", "val", "test")} != ref[
        "split_sizes"
    ]:
        raise ValueError("Expected 70/15/15 gastrulation partitions")
    if not np.array_equal(
        np.sort(np.concatenate(list(splits.values()))), np.arange(len(samples))
    ):
        raise ValueError("Split rows must partition the dataset exactly once")
    if y is not None and (not np.array_equal(y, samples.embryonic_day.to_numpy())):
        raise ValueError("Targets must equal recorded collection days")
    groups = [
        set(samples.pool_index.iloc[splits[role]]) for role in ("train", "val", "test")
    ]
    if any((groups[i] & groups[j] for i in range(3) for j in range(i))):
        raise ValueError("Embryo pools cross partitions")
    return splits


def ingest(raw_root, output_root, *, force=False):
    source = fetch_source(raw_root)
    if digest(SAMPLE_MAP) != registry()["sample_map_sha256"]:
        raise ValueError("Atlas sample map changed")
    target = Path(output_root) / NAME
    if target.exists() and (not force):
        verify_prepared(target)
        return target
    obs = load_source_metadata(source)
    samples, splits, _ = choose_cells(obs)
    counts, genes = read_source_rows(source, samples.source_row.to_numpy())
    selected, scores = pearson_select(counts[splits["train"]])
    X = counts[:, selected].toarray().astype(np.float32)
    y = samples.embryonic_day.to_numpy(dtype=np.float32)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "X.npy", X)
    np.save(target / "y.npy", y)
    np.save(target / "split_indices.npy", splits)
    samples.to_csv(target / "sample_metadata.csv", index=False)
    features = pd.DataFrame(
        {
            "feature_index": np.arange(len(selected)),
            "source_gene_index": selected,
            "gene": genes[selected],
            "training_pearson_residual_variance": scores[selected],
        }
    )
    features.to_csv(target / "feature_metadata.csv", index=False)
    names = [f"{i}__{gene}" for i, gene in zip(selected, genes[selected])]
    (target / "feature_names.json").write_text(json.dumps(names, indent=2) + "\n")
    md = dict(
        name=NAME,
        source="Pijuan-Sala et al. (2019), mouse gastrulation atlas via scVelo",
        source_url=SOURCE_URL,
        publication_doi="10.1038/s41586-019-0933-9",
        description=DESCRIPTION,
        label="Single-Cell",
        sub_labels=["RNA-seq", "Embryonic collection day"],
        task="regression",
        num_samples=10000,
        num_features=10000,
        num_classes=None,
        cat_idx=[],
        split_strategy="fixed",
        curation_protocol=PROTOCOL,
        target_units="embryonic_days",
        feature_representation="source_UMI_counts",
        source_manifest_sha256=digest(REGISTRY),
        sample_metadata_file="sample_metadata.csv",
        feature_metadata_file="feature_metadata.csv",
        READY=True,
    )
    (target / "metadata.json").write_text(
        json.dumps(md, indent=2, ensure_ascii=False) + "\n"
    )
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=REPO / "data/raw/scvelo/Gastrulation"
    )
    parser.add_argument("--output", type=Path, default=REPO / "data/unified_format")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(ingest(args.input, args.output, force=args.force))
