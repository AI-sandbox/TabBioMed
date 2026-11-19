import numpy as np
import os
import json
import gc
import pandas as pd
from .paths import GENOMIC_ANNOTATIONS_DIR
from tabbiomed.data.catalog import dataset_info

RAW_DIR = "data/raw/1KG"
LABELS_DIR = GENOMIC_ANNOTATIONS_DIR
UNIFIED_DIR = "data/unified_format"
sexcheck_datasets = [
    dict(dataset_info(name)["metadata"])
    for name in ("sexcheck_chrX_50k", "sexcheck_chrX_500", "sexcheck_chrX_50")
]


def save_dataset(X, y, output_dir, metadata):
    dataset_info(metadata["name"])
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def imput_mv(X, prob: float, sample_scale: int = 1, y=None):
    if sample_scale < 1:
        raise ValueError("sample_scale must be >= 1")
    X_base = np.asarray(X)
    _, n_features = X_base.shape
    X_expanded = X_base
    if sample_scale > 1:
        assert y is not None, "y labels must be provided when sample_scale > 1"
        X_expanded = np.repeat(X_base, sample_scale, axis=0)
        y = np.repeat(y, sample_scale, axis=0)
    X_imputed = X_expanded.astype(float, copy=True)
    mask = np.random.rand(X_imputed.shape[0], n_features) < prob
    X_imputed[mask] = np.nan
    print(
        f"Applied missingness: {prob * 100:.1f}% of values set to NaN.\nFinal shape: {X_imputed.shape}"
    )
    return (X_imputed, y)


def get_X_y(data, labels: dict, label_type: str):
    X = data.calldata_gt
    X = X.astype(float)
    X[X == -9] = np.nan
    X = np.sum(X, axis=2).T
    y_list = []
    keep_indices = []
    for i, sample_id in enumerate(data.samples):
        if sample_id in labels:
            y_list.append(labels[sample_id])
            keep_indices.append(i)
    X = X[keep_indices]
    if label_type == "sex":
        y = np.array(y_list, dtype=float)
        y[y == 0] = np.nan
        valid_mask = ~np.isnan(y)
    elif label_type == "categorical":
        y = np.array(y_list, dtype=object)
        valid_mask = ~pd.isna(y)
    elif label_type == "continuous":
        y = np.array(y_list, dtype=float)
        valid_mask = ~np.isnan(y).any(axis=1)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"  [{label_type.upper()}] Final shape: X={X.shape}, y={y.shape}")
    return (X, y)


def load_labels():
    pca_path_train = f"{LABELS_DIR}/metadata_train.tsv"
    pca_path_test = f"{LABELS_DIR}/metadata_test.tsv"
    pca_df_train = pd.read_csv(pca_path_train, sep="\t")
    pca_df_test = pd.read_csv(pca_path_test, sep="\t")
    pca_df = pd.concat([pca_df_train, pca_df_test], ignore_index=True)
    pca_df = pca_df[["Sample", "x1", "x2", "x3"]]
    pca_df["ID"] = pca_df["Sample"].apply(lambda x: x[:-3])
    pca_df = pca_df[["ID", "x1", "x2", "x3"]].groupby(["ID"]).mean().reset_index()
    pca_dict = {
        row["ID"]: (row["x1"], row["x2"], row["x3"]) for _, row in pca_df.iterrows()
    }
    geo_path = f"{LABELS_DIR}/1KG_HGDP_geog_metadata.tsv"
    geo_df = pd.read_csv(geo_path, sep="\t")
    continental_df = geo_df[["Sample", "Superpopulation code"]]
    subpopulation_df = geo_df[["Sample", "Population code"]]
    coordinates_df = geo_df[["Sample", "Latitude", "Longitude"]]
    continntal_dict = {
        row["Sample"]: row["Superpopulation code"]
        for _, row in continental_df.iterrows()
    }
    subpopulation_dict = {
        row["Sample"]: row["Population code"] for _, row in subpopulation_df.iterrows()
    }
    coordinates_dict = {
        row["Sample"]: (row["Latitude"], row["Longitude"])
        for _, row in coordinates_df.iterrows()
    }
    ped_file = f"{LABELS_DIR}/integrated_call_samples_v3.20250704.ALL.ped"
    df_1kg = pd.read_csv(ped_file, sep="\t")
    df_1kg.rename(columns={"Individual ID": "IID", "Gender": "Sex"}, inplace=True)
    df_1kg = df_1kg[["IID", "Sex"]]
    hgdp_file = f"{LABELS_DIR}/igsr_HGDP_projct_samples.tsv"
    df_hgdp = pd.read_csv(hgdp_file, sep="\t")
    df_hgdp.rename(columns={"Sample name": "IID"}, inplace=True)
    df_hgdp = df_hgdp[["IID", "Sex"]]
    df_hgdp["Sex"] = df_hgdp["Sex"].apply(
        lambda x: 1 if x in ["M", 1] else 2 if x in ["F", 2] else 0
    )
    sex_df = pd.concat([df_1kg, df_hgdp], ignore_index=True)
    sex_dict = {row["IID"]: row["Sex"] for _, row in sex_df.iterrows()}
    return {
        "pca": pca_dict,
        "continental": continntal_dict,
        "subpopulation": subpopulation_dict,
        "coordinates": coordinates_dict,
        "sex": sex_dict,
    }


def build_and_save_chrX_datasets(sex_labels, test=False):
    from snputils.snp.io import read_pgen

    pgens = ["chrX_50k.pgen", "chrX_500.pgen", "chrX_50.pgen"]
    for pgen_file, metadata in zip(pgens, sexcheck_datasets):
        print(f"Building sexcheck dataset from {pgen_file}...")
        data = read_pgen(os.path.join(RAW_DIR, pgen_file), separator="\t")
        X, y = get_X_y(data, sex_labels, label_type="sex")
        from tabbiomed.data.curation.sexcheck_datasets import (
            PROTOCOL,
            split_note,
            grouped_splits,
            source_individuals,
        )

        _, _, individuals, source_y = source_individuals(RAW_DIR, metadata["name"])
        np.testing.assert_array_equal(y, source_y)
        copies = 3 if "50k" in pgen_file else 1
        if "50k" in pgen_file:
            print(f"  -> Applying 95% synthetic missingness and sample triplication...")
            X, y = imput_mv(X, prob=0.95, sample_scale=3, y=y)
        splits = grouped_splits(np.repeat(individuals, copies), y)
        metadata = {
            **metadata,
            "curation_protocol": PROTOCOL,
            "split_strategy": "fixed",
            "split_group": "individual",
            "num_source_groups": len(individuals),
            "sample_metadata_file": "sample_metadata.csv",
            "description": metadata["description"] + " " + split_note(metadata["name"]),
        }
        output_dir = os.path.join(UNIFIED_DIR, metadata["name"])
        save_dataset(X, y, output_dir, metadata)
        import csv

        np.save(os.path.join(output_dir, "split_indices.npy"), splits)
        with open(os.path.join(output_dir, "sample_metadata.csv"), "w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row", "individual", "masked_copy", "target"])
            writer.writerows(
                (
                    (i, individual, i % copies, int(y[i] - 1))
                    for i, individual in enumerate(splits["groups"])
                )
            )
        print(f"Dataset {metadata['name']} saved to {output_dir}\n")
        gc.collect()


def build_and_save_chr22_datasets(labels):
    from snputils.snp.io import read_pgen

    sources = [
        ("chr22_1000_consec", [("subpop_1000_consec", "subpopulation", "categorical")]),
        (
            "chr22_1k",
            [
                ("continental_1k", "continental", "categorical"),
                ("coords_1k", "coordinates", "continuous"),
                ("pca_1k", "pca", "continuous"),
            ],
        ),
        (
            "chr22_50k",
            [
                ("continental_50k", "continental", "categorical"),
                ("subpop_50k", "subpopulation", "categorical"),
                ("coords_50k", "coordinates", "continuous"),
                ("pca_50k", "pca", "continuous"),
                ("continental_50k_missing", "continental", "categorical"),
            ],
        ),
    ]
    for prefix, tasks in sources:
        data = read_pgen(os.path.join(RAW_DIR, prefix + ".pgen"), separator="\t")
        for name, key, kind in tasks:
            X, y = get_X_y(data, labels[key], label_type=kind)
            if name.endswith("_missing"):
                X, y = imput_mv(X, prob=0.6, sample_scale=1, y=y)
            save_dataset(
                X,
                y,
                os.path.join(UNIFIED_DIR, name),
                dict(dataset_info(name)["metadata"]),
            )


def main():
    np.random.seed(42)
    labels = load_labels()
    build_and_save_chrX_datasets(labels["sex"])
    build_and_save_chr22_datasets(labels)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Curate genomics benchmark data").parse_args()
    main()
