from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import csv
import hashlib
import json
import warnings
import numpy as np

REPO = Path.cwd()
PROTOCOL = "mush21_strain_outer_v1"
MAPPING = CONFIG_DIR / "mush21_sample_metadata.csv"
MANIFEST = CONFIG_DIR / "mush21_source_manifest.json"
CULTURE_CODES = {"0": "PA", "1": "EX", "2": "HY", "3": "SC", "4": "GA", "5": "SO"}
SPECIES = {
    "0": "Entomortierella galaxiae",
    "1": "Linnemannia exigua",
    "2": "Linnemannia hyalina",
    "3": "Linnemannia sclerotiella",
    "4": "Linnemannia gamsii",
    "5": "Linnemannia solitaria",
}
DESCRIPTION = "PTR-ToF-MS fungal species classification: 593 measurements and 383 peaks from 25 biological strains of six species, measured in pure culture and co-culture with Pseudomonas helmanticensis (Probst et al., Scientific Reports 2023). Labels 0-5 correspond to Entomortierella galaxiae, Linnemannia exigua, L. hyalina, L. sclerotiella, L. gamsii and L. solitaria. Five strain-grouped outer folds test every measurement once. The three requested inner folds reduce to two in each outer fold because of limited class support at the strain level. The final-fit validation holdout is also strain-disjoint. Sample, strain and condition identifiers are metadata, not features."


def registry():
    return json.loads(MANIFEST.read_text())


def load_source_mapping():
    reference = registry()
    if hashlib.sha256(MAPPING.read_bytes()).hexdigest() != reference["mapping_sha256"]:
        raise ValueError(
            "PT-Mush21 strain mapping changed; reverify original study metadata"
        )
    with MAPPING.open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["row"]) for r in rows] != list(range(593)):
        raise ValueError(
            "PT-Mush21 requires all 593 verified sample identities in order"
        )
    strains = {r["biological_strain"] for r in rows}
    if len(strains) != 25 or len({r["source_ID"] for r in rows}) != 50:
        raise ValueError("PT-Mush21 requires 25 strains measured under two conditions")
    for strain in strains:
        selected = [r for r in rows if r["biological_strain"] == strain]
        if {r["cultivation_mode"] for r in selected} != {"c", "+"} or len(
            {r["target"] for r in selected}
        ) != 1:
            raise ValueError(f"Invalid strain/condition identities: {strain}")
    return rows


def verify_values(X, y, source_ids=None):
    X, y = (np.asarray(X), np.asarray(y))
    if X.shape != (593, 383) or y.shape != (593,):
        raise ValueError("PT-Mush21 requires its verified 593 x 383 source matrix")
    reference = registry()
    for key, array in [
        ("X_values_sha256", np.ascontiguousarray(X, dtype="<f4")),
        ("y_values_sha256", np.ascontiguousarray(y, dtype="<i8")),
    ]:
        if hashlib.sha256(array.tobytes()).hexdigest() != reference[key]:
            raise ValueError(
                "PT-Mush21 features, labels or row order changed; reverify strain mapping"
            )
    rows = load_source_mapping()
    if not np.array_equal(y, [int(r["target"]) for r in rows]):
        raise ValueError("PT-Mush21 labels disagree with verified species identities")
    if source_ids is not None and list(source_ids) != [r["source_ID"] for r in rows]:
        raise ValueError("PT-Mush21 source IDs disagree with the verified row mapping")
    return rows


def grouped_splits(X, y, source_ids=None, *, seed=42):
    from tabbiomed.cv_splits import make_grouped_outer_cv

    rows = verify_values(X, y, source_ids)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The least populated class in y has only"
        )
        splits = make_grouped_outer_cv(
            [r["biological_strain"] for r in rows], y, "classification", seed=seed
        )
    splits.update(
        group_split_protocol=PROTOCOL,
        group_source="Probst2023 PTR_data.txt:class_name+type",
        source_culture_ids=[r["source_ID"] for r in rows],
        source_sample_ids=[
            r["sample_id"] + "|cycle=" + r["measurement_cycle"] for r in rows
        ],
    )
    return splits


def validate_splits(X, y, splits):
    from tabbiomed.cv_splits import validate_grouped_outer_cv

    rows = verify_values(X, y)
    expected = [r["biological_strain"] for r in rows]
    if (
        splits.get("group_split_protocol") != PROTOCOL
        or splits.get("groups") != expected
        or splits.get("source_culture_ids") != [r["source_ID"] for r in rows]
        or (
            splits.get("source_sample_ids")
            != [r["sample_id"] + "|cycle=" + r["measurement_cycle"] for r in rows]
        )
    ):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )
    validate_grouped_outer_cv(splits, expected)
    return np.asarray(expected)


def source_metadata():
    reference = registry()
    return dict(
        description=DESCRIPTION,
        split_strategy="cv",
        split_group="biological_strain",
        outer_split_protocol=PROTOCOL,
        num_strains=25,
        num_source_groups=50,
        publication_doi=reference["publication_doi"],
        source_url="https://doi.org/" + reference["publication_doi"],
        target_mapping=CULTURE_CODES,
        class_names=[SPECIES[str(i)] for i in range(6)],
        strain_mapping_sha256=reference["mapping_sha256"],
        sample_metadata_file="sample_metadata.csv",
    )


def verify_original_source(source_path, release_path=None):
    import pandas as pd

    reference = registry()
    if (
        hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        != reference["source_sha256"]
    ):
        raise ValueError("Original PT-Mush21 study source changed")
    release_path = release_path or REPO / "data/raw" / reference["release_file"]
    if (
        hashlib.sha256(Path(release_path).read_bytes()).hexdigest()
        != reference["release_sha256"]
    ):
        raise ValueError("TabPFN-VOCS source release changed")
    source = pd.read_csv(source_path, sep=";")
    source = source.loc[(source.type_sample == "funghi") & (source.outliers == "no")]
    source = source.reset_index().rename(columns={"index": "source_row"})
    released = json.loads(Path(release_path).read_text())[reference["release_key"]]
    features = pd.DataFrame(released["X"])
    rows = verify_values(
        features.to_numpy(dtype=np.float32), np.asarray(released["y"]), released["IDs"]
    )
    checks = {
        "source_row": source.source_row.astype(str),
        "sample_id": source["sample"],
        "measurement_cycle": source.cycle.astype(str),
        "species_code": source.class_name,
        "strain_number": source.strain.astype(int).astype(str),
        "strain_id": source.type,
        "technical_replicate": source.replica,
        "cultivation_mode": source.trt,
    }
    for field, actual in checks.items():
        if actual.tolist() != [r[field] for r in rows]:
            raise ValueError(f"Original source identity mismatch: {field}")
    common = [c for c in features if c in source]
    raw, normalized = (source[common].to_numpy(float), features[common].to_numpy(float))
    denominator = (raw * normalized).sum(1) / (normalized**2).sum(1)
    error = float(np.abs(raw - normalized * denominator[:, None]).max())
    if len(common) != 114 or error > 1e-09:
        raise ValueError(
            "Original study concentrations no longer match the benchmark samples"
        )
    return dict(
        matched_samples=len(rows),
        shared_named_peaks=len(common),
        max_abs_reconstruction_error=error,
        biological_strains=25,
    )
