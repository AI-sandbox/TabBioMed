from tabbiomed.data.catalog import dataset_info
from .paths import CONFIG_DIR
import json
import os
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/tahoe-100M"
UNIFIED_DIR = "data/unified_format"
base_plateX_name = "plateX_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
PLATE_PATHS = {
    f"plate{X}": f"{RAW_PATH}/{base_plateX_name.replace('X', str(X))}"
    for X in range(1, 15)
}
CELL_NAME_TRAIN_PLATES = ["plate1", "plate4", "plate7"]
CELL_NAME_VAL_PLATES = ["plate2", "plate5", "plate8"]
CELL_NAME_TEST_PLATES = ["plate3", "plate6", "plate9"]
CELL_NAME_N_TRAIN = 5000
CELL_NAME_N_VAL = 1071
CELL_NAME_N_TEST = 1071
CELL_NAME_DIMS = [5000, 1000, 500]
CELL_NAME_METHODS = ["pearson_residuals"]
G2M_TRAIN_PLATES = ["plate1", "plate4", "plate7"]
G2M_VAL_PLATES = ["plate2", "plate5", "plate8"]
G2M_TEST_PLATES = ["plate3", "plate6", "plate9"]
G2M_N_TRAIN = 5000
G2M_N_VAL = 1071
G2M_N_TEST = 1071
G2M_DIMS = [5000]
G2M_METHODS = ["pearson_residuals"]
BATCH_14_PLATES = [f"plate{i}" for i in range(1, 15)]
BATCH_N_PER_PLATE = 2000
BATCH_DIM = 5000
BATCH_VAL_FRAC = 0.15
BATCH_TEST_FRAC = 0.15
RANDOM_SEED = 42
DESCRIPTION = "Single-cell RNA-seq data from the Tahoe-100M dataset (ARC Institute, 2025). Cells are subsampled; gene selection uses training cells only, preserving raw UMI counts."
SOURCE = "ARC Institute - Tahoe-100M (2025)."
LABEL = "Single-Cell"
SUB_LABELS = ["RNA-seq"]


def _to_dense(X_sparse):
    if sparse.issparse(X_sparse):
        return X_sparse.toarray()
    return np.asarray(X_sparse)


def build_metadata(
    name,
    source,
    description,
    label,
    sub_labels,
    X,
    task,
    num_classes=None,
    cat_idx=None,
):
    if cat_idx is None:
        cat_idx = []
    meta = {
        "name": name,
        "source": source,
        "description": description,
        "label": label,
        "sub_labels": sub_labels,
        "task": task,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "cat_idx": cat_idx,
    }
    if num_classes is not None:
        meta["num_classes"] = num_classes
    return meta


def save_dataset(X, y, split_indices, metadata, out_dir):
    dataset_info(metadata["name"])
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X.npy", X)
    np.save(f"{out_dir}/y.npy", y)
    if split_indices is not None:
        np.save(f"{out_dir}/split_indices.npy", split_indices)
        split_str = f"train={len(split_indices['train'])}, val={len(split_indices['val'])}, test={len(split_indices['test'])}"
    else:
        split_str = "splits=None (managed by pipeline)"
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved dataset to {out_dir}  (X={X.shape}, {split_str})")


def _load_and_subsample_sparse(plate_name, n_samples, rng):
    path = PLATE_PATHS[plate_name]
    print(f"  Loading {plate_name} (backed mode) ...")
    adata = sc.read_h5ad(path, backed="r")
    n_cells = adata.n_obs
    expected = json.loads((CONFIG_DIR / "tahoe_batch_sampling.json").read_text())[
        "plate_rows"
    ][plate_name]
    if n_cells != expected:
        raise ValueError(f"Tahoe source plate row count differs: {plate_name}")
    cell_idx = rng.choice(n_cells, size=min(n_samples, n_cells), replace=False)
    cell_idx = np.sort(cell_idx)
    adata_sub = adata[cell_idx, :].to_memory()
    adata_sub.obs["plate"] = plate_name
    return adata_sub


def _random_split_indices(n_samples):
    perm = np.random.RandomState(RANDOM_SEED).permutation(n_samples)
    n_test = int(BATCH_TEST_FRAC * n_samples)
    n_val = int(BATCH_VAL_FRAC * n_samples)
    return {
        "train": perm[n_test + n_val :].tolist(),
        "val": perm[n_test : n_test + n_val].tolist(),
        "test": perm[:n_test].tolist(),
    }


def build_cell_name_datasets(rng):
    print("\n--- Constructing Cell Name Datasets ---")
    plates_to_load = (
        CELL_NAME_TRAIN_PLATES + CELL_NAME_VAL_PLATES + CELL_NAME_TEST_PLATES
    )
    n_samples_dict = {
        **{p: CELL_NAME_N_TRAIN for p in CELL_NAME_TRAIN_PLATES},
        **{p: CELL_NAME_N_VAL for p in CELL_NAME_VAL_PLATES},
        **{p: CELL_NAME_N_TEST for p in CELL_NAME_TEST_PLATES},
    }
    adatas = []
    for p in plates_to_load:
        adata_sub = _load_and_subsample_sparse(p, n_samples_dict[p], rng)
        adatas.append(adata_sub)
    print("Concatenating cell name plates...")
    merged = sc.concat(adatas, join="inner")
    sizes = [n_samples_dict[p] for p in plates_to_load]
    cumulative = np.cumsum([0] + sizes)
    plate_ranges = {
        p: (cumulative[i], cumulative[i + 1]) for i, p in enumerate(plates_to_load)
    }
    train_idx = np.concatenate(
        [np.arange(*plate_ranges[p]) for p in CELL_NAME_TRAIN_PLATES]
    )
    val_idx = np.concatenate(
        [np.arange(*plate_ranges[p]) for p in CELL_NAME_VAL_PLATES]
    )
    test_idx = np.concatenate(
        [np.arange(*plate_ranges[p]) for p in CELL_NAME_TEST_PLATES]
    )
    split_indices = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    y_str = merged.obs["cell_name"].values.astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_str).astype(np.int64)
    n_classes = len(le.classes_)
    for dim in CELL_NAME_DIMS:
        for method in CELL_NAME_METHODS:
            print(f"Applying unified preprocessing for dim={dim}, method={method}")
            adata_proc = apply_unified_preprocessing(
                merged.copy(), fit_idx=train_idx, n_top_genes=dim, method=method
            )
            X_dense = _to_dense(adata_proc.X).astype(np.float32)
            out_name = f"tahoe_cell_name_{dim}"
            desc = (
                DESCRIPTION
                + f" Task: classify cancer cell-line identity. Method: {method} (genes selected on training cells), Dim: {dim}."
            )
            meta = build_metadata(
                name=out_name,
                source=SOURCE,
                description=desc,
                label=LABEL,
                sub_labels=SUB_LABELS + ["Cell-line Classification"],
                X=X_dense,
                task="classification",
                num_classes=n_classes,
            )
            save_dataset(
                X_dense, y_enc, split_indices, meta, f"{UNIFIED_DIR}/{out_name}"
            )
            del X_dense, adata_proc


def build_g2m_score_datasets(rng):
    print("\n--- Constructing G2M Score Datasets ---")
    plates_to_load = G2M_TRAIN_PLATES + G2M_VAL_PLATES + G2M_TEST_PLATES
    n_samples_dict = {
        **{p: G2M_N_TRAIN for p in G2M_TRAIN_PLATES},
        **{p: G2M_N_VAL for p in G2M_VAL_PLATES},
        **{p: G2M_N_TEST for p in G2M_TEST_PLATES},
    }
    adatas = []
    for p in plates_to_load:
        adata_sub = _load_and_subsample_sparse(p, n_samples_dict[p], rng)
        adatas.append(adata_sub)
    print("Concatenating G2M Score plates...")
    merged = sc.concat(adatas, join="inner")
    sizes = [n_samples_dict[p] for p in plates_to_load]
    cumulative = np.cumsum([0] + sizes)
    plate_ranges = {
        p: (cumulative[i], cumulative[i + 1]) for i, p in enumerate(plates_to_load)
    }
    train_idx = np.concatenate([np.arange(*plate_ranges[p]) for p in G2M_TRAIN_PLATES])
    val_idx = np.concatenate([np.arange(*plate_ranges[p]) for p in G2M_VAL_PLATES])
    test_idx = np.concatenate([np.arange(*plate_ranges[p]) for p in G2M_TEST_PLATES])
    split_indices = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    y_reg = merged.obs["G2M_score"].values.astype(np.float32)
    for dim in G2M_DIMS:
        for method in G2M_METHODS:
            print(f"Applying unified preprocessing for dim={dim}, method={method}")
            adata_proc = apply_unified_preprocessing(
                merged.copy(), fit_idx=train_idx, n_top_genes=dim, method=method
            )
            X_dense = _to_dense(adata_proc.X).astype(np.float32)
            out_name = "tahoe_g2m_score"
            desc = (
                DESCRIPTION
                + f" Task: reconstruct the source expression-derived G2/M score (regression). Method: {method} (genes selected on training cells), Dim: {dim}."
            )
            meta = build_metadata(
                name=out_name,
                source=SOURCE,
                description=desc,
                label=LABEL,
                sub_labels=SUB_LABELS + ["G2M Score Regression"],
                X=X_dense,
                task="regression",
                num_classes=None,
            )
            save_dataset(
                X_dense, y_reg, split_indices, meta, f"{UNIFIED_DIR}/{out_name}"
            )
            del X_dense, adata_proc, merged


def build_batch_integration_datasets(rng):
    sampling = json.loads((CONFIG_DIR / "tahoe_batch_sampling.json").read_text())
    rng = np.random.default_rng()
    rng.bit_generator.state = sampling["bit_generator_state"]
    print("\n--- Constructing Batch Integration Datasets ---")
    for name, plates in [("14plates", BATCH_14_PLATES)]:
        print(f"Processing Batch Integration: {name}")
        adatas = []
        for p in plates:
            adata_sub = _load_and_subsample_sparse(p, BATCH_N_PER_PLATE, rng)
            adatas.append(adata_sub)
        print(f"Concatenating {name} plates...")
        merged = sc.concat(adatas, join="inner")
        split_indices = _random_split_indices(merged.n_obs)
        y_str = merged.obs["plate"].values.astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y_str).astype(np.int64)
        n_classes = len(le.classes_)
        print(f"Selecting genes by pearson_residuals...")
        adata_proc = apply_unified_preprocessing(
            merged.copy(),
            fit_idx=split_indices["train"],
            n_top_genes=BATCH_DIM,
            method="pearson_residuals",
        )
        X_dense = _to_dense(adata_proc.X).astype(np.float32)
        out_name = f"tahoe_batch_integration_{name}"
        desc = (
            DESCRIPTION
            + f" Task: classify plate identity across {name}; plates also differ in drug panels. Genes selected by pearson_residuals on training cells."
        )
        meta = build_metadata(
            name=out_name,
            source=SOURCE,
            description=desc,
            label=LABEL,
            sub_labels=SUB_LABELS + ["Batch Integration"],
            X=X_dense,
            task="classification",
            num_classes=n_classes,
        )
        save_dataset(X_dense, y_enc, split_indices, meta, f"{UNIFIED_DIR}/{out_name}")
        del X_dense, adata_proc, merged, adatas


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    build_cell_name_datasets(rng)
    build_g2m_score_datasets(rng)
    build_batch_integration_datasets(rng)
    print("\nAll datasets generated successfully.")


def apply_unified_preprocessing(
    adata, fit_idx, n_top_genes=None, method="pearson_residuals"
):
    if method != "pearson_residuals":
        raise ValueError("The benchmark uses Pearson-residual gene selection")
    train = adata[np.asarray(fit_idx)].copy()
    if "counts" in train.layers:
        train.X = train.layers["counts"].copy()
    sc.experimental.pp.highly_variable_genes(
        train, n_top_genes=n_top_genes, flavor="pearson_residuals", inplace=True
    )
    return adata[:, train.var["highly_variable"].to_numpy()].copy()


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Curate tahoe benchmark data").parse_args()
    main()
