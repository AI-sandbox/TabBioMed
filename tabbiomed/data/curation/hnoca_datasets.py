from pathlib import Path
import argparse
import json
import itertools
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path.cwd()
RAW_PATH = REPO_ROOT / "data/raw/HNOCA/hnoca_cleaned.h5ad"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/unified_format"
LABEL_KEY = "annot_level_4_rev2"
N_CLASSES = 30
SPLIT_SIZES = {"train": 70000, "val": 15000, "test": 15000}
N_TOP_GENES = 22000
COUNTS_LAYER = "counts_lengthnorm"
RANDOM_SEED = 42
ROW_BLOCK = 20000
TARGET_FRACTIONS = np.array([0.7, 0.15, 0.15])
MIN_EXPECTED_PER_CLASS = 15.0
SOURCE = "He et al. (2024), Human Neural Organoid Cell Atlas (HNOCA), hnoca_cleaned.h5ad downloaded from https://zenodo.org/records/14161275."
DESCRIPTION = "Single-cell RNA-seq of human neural organoids from the HNOCA atlas (27 publications). Task: predict the HNOCA harmonized level-4 cell-type annotation (annot_level_4_rev2) among its {n_classes} most frequent classes. The split is donor- and publication-disjoint: groups are connected components of the publication-cell line graph and are assigned to train/val/test (train: {train_pubs}; val: {val_pubs}; test: {test_pubs}); then {n_train}/{n_val}/{n_test} cells are sampled within the splits, stratified by class. Features: log1p(CPM) expression of {n_genes} genes selected with analytic Pearson residuals (scanpy, flavor='pearson_residuals') computed on the raw counts of the training cells only."
LABEL = "Single-Cell"
SUB_LABELS = ["RNA-seq"]
TASK = "classification"


def _read_obs(path: Path) -> pd.DataFrame:
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs[[LABEL_KEY, "publication", "cell_line"]].copy()
    var_names = adata.var_names.to_numpy()
    symbols = (
        adata.var["gene_symbol"].astype(str).to_numpy()
        if "gene_symbol" in adata.var
        else var_names
    )
    adata.file.close()
    obs.attrs["var_names"], obs.attrs["gene_symbols"] = (var_names, symbols)
    return obs


def _stratified_subsample(
    labels: np.ndarray, candidates: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    cls, counts = np.unique(labels[candidates], return_counts=True)
    quota = counts / counts.sum() * n
    take = np.floor(quota).astype(int)
    for i in np.argsort(-(quota - take))[: n - take.sum()]:
        take[i] += 1
    chosen = [
        rng.choice(candidates[labels[candidates] == c], size=t, replace=False)
        for c, t in zip(cls, take)
    ]
    return np.sort(np.concatenate(chosen))


def _read_rows(
    path: Path, rows: np.ndarray, n_vars: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    rows = np.sort(rows)
    out = {"X": ([], [], [0]), COUNTS_LAYER: ([], [], [0])}
    with h5py.File(path, "r") as f:
        groups = {"X": f["X"], COUNTS_LAYER: f["layers"][COUNTS_LAYER]}
        indptr = groups["X"]["indptr"][:]
        assert np.array_equal(indptr, groups[COUNTS_LAYER]["indptr"][:]), (
            "X/counts sparsity differs"
        )
        n_obs = len(indptr) - 1
        for b0 in range(0, n_obs, ROW_BLOCK):
            b1 = min(n_obs, b0 + ROW_BLOCK)
            sel = rows[(rows >= b0) & (rows < b1)]
            if len(sel) == 0:
                continue
            lo, hi = (indptr[b0], indptr[b1])
            for key, g in groups.items():
                data, idx = (g["data"][lo:hi], g["indices"][lo:hi].astype(np.int32))
                d_list, i_list, p_list = out[key]
                for r in sel:
                    s, e = (indptr[r] - lo, indptr[r + 1] - lo)
                    d_list.append(data[s:e])
                    i_list.append(idx[s:e])
                    p_list.append(p_list[-1] + (e - s))
            print(f"    rows {b0:>9,}-{b1:>9,}: kept {len(sel)}", flush=True)
    mats = []
    for key in ("X", COUNTS_LAYER):
        d_list, i_list, p_list = out[key]
        mats.append(
            sparse.csr_matrix(
                (np.concatenate(d_list), np.concatenate(i_list), np.array(p_list)),
                shape=(len(rows), n_vars),
            )
        )
    return (mats[0], mats[1])


def _pearson_residual_hvg(
    counts_train: sparse.csr_matrix, n_top_genes: int
) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    expressed = np.asarray((counts_train > 0).sum(axis=0)).ravel() > 0
    print(
        f"  genes expressed in training cells: {expressed.sum()} / {counts_train.shape[1]}"
    )
    if expressed.sum() < n_top_genes:
        raise ValueError("fewer expressed genes than n_top_genes")
    tmp = ad.AnnData(counts_train[:, expressed].astype(np.float32))
    sc.experimental.pp.highly_variable_genes(
        tmp, flavor="pearson_residuals", n_top_genes=n_top_genes, inplace=True
    )
    selected_expressed = np.where(tmp.var["highly_variable"].to_numpy())[0]
    return np.sort(np.where(expressed)[0][selected_expressed])


def connected_components(publication: pd.Series, cell_line: pd.Series) -> pd.Series:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pubs = publication.astype(str).to_numpy()
    lines = cell_line.astype(str).to_numpy()
    for p in np.unique(pubs):
        find("P:" + p)
    edges = pd.DataFrame({"p": pubs, "c": lines})
    edges = edges[edges.c.str.lower() != "unknown"].drop_duplicates()
    for p, c in edges.itertuples(index=False):
        parent[find("P:" + p)] = find("C:" + c)
    roots = np.array([find("P:" + p) for p in pubs])
    names = {}
    for r in np.unique(roots):
        names[r] = " | ".join(sorted(np.unique(pubs[roots == r])))
    return pd.Series(
        [names[r] for r in roots], index=publication.index, name="component"
    )


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_pm = np.where(p > 0, p * np.log(p / m), 0.0).sum(-1)
        kl_qm = np.where(q > 0, q * np.log(q / m), 0.0).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def assign_components(
    labels: pd.Series,
    components: pd.Series,
    target_sizes=(70000, 15000, 15000),
    chunk: int = 500000,
) -> tuple[dict[str, int], pd.DataFrame]:
    counts = pd.crosstab(components, labels)
    comps = counts.sum(1).sort_values(ascending=False).index.tolist()
    counts = counts.loc[comps].to_numpy(dtype=np.float64)
    total = counts.sum()
    giant, rest = (counts[0], counts[1:])
    n_rest = len(rest)
    best = None
    digits = (
        np.array(list(itertools.product(range(3), repeat=n_rest)), dtype=np.int8)
        if n_rest <= 12
        else None
    )
    n_total = 3**n_rest
    for start in range(0, n_total, chunk):
        idx = np.arange(start, min(n_total, start + chunk), dtype=np.int64)
        if digits is not None:
            a = digits[idx]
        else:
            a = np.stack(
                [idx // 3**k % 3 for k in range(n_rest - 1, -1, -1)], axis=1
            ).astype(np.int8)
        onehot = np.stack([a == s for s in range(3)], axis=-1).astype(np.float64)
        split_cls = np.einsum("ncs,ck->nsk", onehot, rest)
        split_cls[:, 0, :] += giant
        sizes = split_cls.sum(-1)
        frac = sizes / total
        with np.errstate(divide="ignore", invalid="ignore"):
            share = split_cls / sizes[..., None]
        share = np.nan_to_num(share)
        expected = share * np.asarray(target_sizes, dtype=np.float64)[None, :, None]
        missing = (expected[:, 1:, :] < MIN_EXPECTED_PER_CLASS).sum((1, 2))
        missing += (split_cls[:, 0, :] < 1).sum(-1) * 100
        size_dev = np.abs(frac - TARGET_FRACTIONS[None, :]).sum(-1)
        js = _js(share[:, 1, :], share[:, 0, :]) + _js(share[:, 2, :], share[:, 0, :])
        empty = (sizes[:, 1] == 0) | (sizes[:, 2] == 0)
        score = np.where(empty, np.inf, missing * 10.0 + size_dev + 0.1 * js)
        i = int(np.argmin(score))
        if best is None or score[i] < best[0]:
            best = (
                float(score[i]),
                a[i].copy(),
                int(missing[i]),
                float(size_dev[i]),
                float(js[i]),
            )
    _, a, missing, size_dev, js = best
    assignment = {comps[0]: 0}
    assignment.update({c: int(s) for c, s in zip(comps[1:], a)})
    summary = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "cells": [
                sum(
                    (counts[i].sum() for i, c in enumerate(comps) if assignment[c] == s)
                )
                for s in range(3)
            ],
        }
    )
    summary["fraction"] = summary.cells / total
    summary.attrs.update(missing=missing, size_dev=size_dev, js=js)
    return (assignment, summary)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    print(f"Reading obs from {args.raw_path} ...", flush=True)
    obs = _read_obs(args.raw_path)
    labels_all = obs[LABEL_KEY].astype(str).to_numpy()
    top_classes = (
        obs[LABEL_KEY].astype(str).value_counts().head(N_CLASSES).index.tolist()
    )
    in_task = np.isin(labels_all, top_classes)
    print(
        f"  {len(obs):,} cells; top {N_CLASSES} '{LABEL_KEY}' classes cover {in_task.sum():,} cells"
    )
    task_obs = obs[in_task]
    comp = connected_components(task_obs["publication"], task_obs["cell_line"])
    assignment, summary = assign_components(
        task_obs[LABEL_KEY].astype(str), comp, target_sizes=tuple(SPLIT_SIZES.values())
    )
    print(summary.to_string(index=False), summary.attrs)
    split_of_cell = comp.map(assignment).to_numpy()
    split_names = list(SPLIT_SIZES)
    rng = np.random.default_rng(RANDOM_SEED)
    task_rows = np.where(in_task)[0]
    chosen = {}
    for s, name in enumerate(split_names):
        chosen[name] = _stratified_subsample(
            labels_all, task_rows[split_of_cell == s], SPLIT_SIZES[name], rng
        )
        print(f"  {name}: sampled {len(chosen[name]):,} cells")
    all_rows = np.sort(np.concatenate([chosen[n] for n in split_names]))
    print(f"Streaming {len(all_rows):,} cells from the h5ad ...", flush=True)
    var_names, symbols = (obs.attrs["var_names"], obs.attrs["gene_symbols"])
    X_sel, counts_sel = _read_rows(args.raw_path, all_rows, len(var_names))
    pos = {r: i for i, r in enumerate(all_rows)}
    order = {n: np.array([pos[r] for r in chosen[n]]) for n in split_names}
    print(
        f"Selecting {N_TOP_GENES} genes by Pearson residuals on training counts ...",
        flush=True,
    )
    genes = _pearson_residual_hvg(counts_sel[order["train"]], N_TOP_GENES)
    row_order = np.concatenate([order[n] for n in split_names])
    X = X_sel[row_order][:, genes].toarray().astype(np.float32)
    le = LabelEncoder().fit(sorted(top_classes))
    y = le.transform(labels_all[all_rows[row_order]]).astype(np.int64)
    bounds = np.cumsum([0] + [SPLIT_SIZES[n] for n in split_names])
    split_indices = {
        n: list(range(bounds[i], bounds[i + 1])) for i, n in enumerate(split_names)
    }
    pubs = {
        n: sorted(
            set(task_obs["publication"].astype(str).to_numpy()[split_of_cell == s])
        )
        for s, n in enumerate(split_names)
    }
    name = f"hnoca_cells{N_CLASSES}"
    metadata = {
        "name": name,
        "source": SOURCE,
        "description": DESCRIPTION.format(
            n_classes=N_CLASSES,
            n_genes=len(genes),
            n_train=SPLIT_SIZES["train"],
            n_val=SPLIT_SIZES["val"],
            n_test=SPLIT_SIZES["test"],
            train_pubs="; ".join(pubs["train"]),
            val_pubs="; ".join(pubs["val"]),
            test_pubs="; ".join(pubs["test"]),
        ),
        "label": LABEL,
        "sub_labels": SUB_LABELS,
        "task": TASK,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "cat_idx": [],
        "num_classes": N_CLASSES,
        "label_key": LABEL_KEY,
        "classes": le.classes_.tolist(),
    }
    out_dir = args.output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "split_indices.npy", split_indices)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump([f"{symbols[g]}|{var_names[g]}" for g in genes], f)
    with open(out_dir / "split_groups.json", "w") as f:
        json.dump(
            {
                "split_by_component": {
                    c: split_names[s] for c, s in assignment.items()
                },
                "publications_by_split": pubs,
                "cells_before_subsampling": {
                    n: int((split_of_cell == s).sum())
                    for s, n in enumerate(split_names)
                },
                "class_counts": {
                    n: pd.Series(y[split_indices[n]])
                    .map(dict(enumerate(le.classes_)))
                    .value_counts()
                    .to_dict()
                    for n in split_names
                },
                "atlas_cell_index": {
                    n: all_rows[order[n]].tolist() for n in split_names
                },
            },
            f,
            indent=1,
        )
    print(
        f"Saved {out_dir}: X={X.shape} y={y.shape} train/val/test={[len(split_indices[n]) for n in split_names]}"
    )


if __name__ == "__main__":
    main()
