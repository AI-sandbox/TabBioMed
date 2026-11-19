from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import hashlib
import json
import numpy as np
import pandas as pd

PROTOCOL = "lgg_xena_methylation450_v1"
REGISTRY = CONFIG_DIR
MANIFEST = REGISTRY / "multiomics_lgg_source_manifest.json"
PROBES = REGISTRY / "multiomics_lgg_methylation_probes.csv"
SPLITS = REGISTRY / "multiomics_lgg_outer_splits.json"
DESCRIPTION = "TCGA lower-grade glioma three-class classification, with 247 distinct patients. CNV (5,000), mRNA (3,435) and miRNA (102) features and targets are retained from MLOmics. DNA methylation is reconstructed from TCGA/Xena HumanMethylation450 by verified CpG identifiers and primary-tumour patient barcodes: 4,705 of the 5,000 original probes are assayed; the 295 unassayed probes are excluded. Missing beta values remain missing for fold-fitted imputation. Existing five outer folds and three-fold inner CV are retained. Patient identifiers are metadata, never features."


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def registry():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["protocol"] != PROTOCOL or digest(PROBES) != manifest["mapping_sha256"]:
        raise ValueError("LGG CpG identity registry changed; reverify the source")
    mapping = pd.read_csv(PROBES)
    if (
        mapping.source_probe_row.tolist() != list(range(5000))
        or not mapping.probe.is_unique
        or (not (mapping.n_candidates == 1).all())
        or (mapping.on_lgg450.sum() != 4705)
    ):
        raise ValueError("Invalid LGG probe identity/assay coverage")
    if digest(SPLITS) != manifest["outer_splits_registry_sha256"]:
        raise ValueError("LGG retained outer folds changed")
    return (manifest, mapping)


def retained_splits():
    registry()
    return json.loads(SPLITS.read_text())


def load_verified_methylation(directory):
    manifest, mapping = registry()
    path = Path(directory) / "LGG_Methy_verified.csv"
    if not path.exists() or digest(path) != manifest["verified_methylation_csv_sha256"]:
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )
    table = pd.read_csv(path, index_col=0)
    if (
        table.index.tolist() != mapping.loc[mapping.on_lgg450, "probe"].tolist()
        or table.columns.tolist() != manifest["sample_ids"]
    ):
        raise ValueError("LGG verified methylation has changed CpG/sample identities")
    return table


def verify_values(X, y):
    manifest, _ = registry()
    if (
        np.asarray(X).shape != (247, 13242)
        or np.asarray(X).dtype != np.dtype("float32")
        or value_digest(X) != manifest["X_values_sha256"]
        or (value_digest(np.asarray(y, dtype=np.int64)) != manifest["y_values_sha256"])
    ):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )


def source_metadata():
    manifest, _ = registry()
    return dict(
        source="MLOmics GS-LGG with TCGA/Xena HumanMethylation450",
        description=DESCRIPTION,
        source_url=manifest["source_metadata_url"],
        methylation_protocol=PROTOCOL,
        sample_metadata_file="sample_metadata.csv",
        source_manifest_sha256=digest(MANIFEST),
        split_strategy="cv",
        omics_feature_counts={"CNV": 5000, "Methy": 4705, "miRNA": 102, "mRNA": 3435},
    )


def download_methylation(destination):
    import requests
    import tempfile
    import os

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest, mapping = registry()
    probes = mapping.loc[mapping.on_lgg450, "probe"].tolist()
    samples = [sample.replace(".", "-") for sample in manifest["sample_ids"]]
    blocks = []
    for start in range(0, len(probes), 200):
        wanted = probes[start : start + 200]
        query = "(fetch [{:table %s :samples %s :columns %s}])" % tuple(
            json.dumps(value) for value in (manifest["source_dataset"], samples, wanted)
        )
        response = requests.post(
            manifest["source_hub"] + "/data/",
            data=query.encode(),
            headers={"Content-Type": "text/plain"},
            timeout=90,
        )
        response.raise_for_status()
        values = np.asarray(response.json(), dtype=np.float64)
        if values.shape != (len(wanted), len(samples)):
            raise ValueError("Xena returned different probe or patient dimensions")
        blocks.append(values)
    descriptor, filename = tempfile.mkstemp(
        prefix=".methylation-", suffix=".csv", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(filename)
    try:
        pd.DataFrame(
            np.concatenate(blocks), index=probes, columns=manifest["sample_ids"]
        ).to_csv(temporary, index_label="CpG_probe")
        if digest(temporary) != manifest["verified_methylation_csv_sha256"]:
            raise ValueError("TCGA/Xena values differ from the pinned source release")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the source LGG CpG measurements"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/raw/MultiOmics_Datasets/GS-LGG/Verified/LGG_Methy_verified.csv"
        ),
    )
    args = parser.parse_args()
    print(download_methylation(args.output))
