from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import csv
import hashlib
import json
import numpy as np

MAPPING = CONFIG_DIR / "tox171_sample_metadata.csv"
MANIFEST = CONFIG_DIR / "tox171_source_manifest.json"
PROTOCOL = "tox171_donor_outer_v1"
CLASS_NAMES = [
    "healthy control",
    "skin cancer control",
    "limited toxicity control",
    "acute toxicity",
]
DESCRIPTION = "Retrospective four-group classification of healthy controls, skin-cancer controls, limited-radiation-toxicity controls, and acute-radiation-toxicity cases. The 171 gene-expression profiles represent 57 donors, each measured in donor-derived lymphoblastoid cells under mock, ultraviolet, and ionizing-radiation conditions (GEO GDS968/GSE1725; Rieger et al., PNAS 2004). Five donor-grouped outer folds test every profile once; three donor-grouped inner folds select hyperparameters. The final-fit validation holdout is also donor-disjoint. Donor and GEO sample identifiers are split metadata, not features."


def load_source_mapping():
    manifest = json.loads(MANIFEST.read_text())
    if hashlib.sha256(MAPPING.read_bytes()).hexdigest() != manifest["mapping_sha256"]:
        raise ValueError("TOX_171 source mapping changed; reverify against GEO GDS968")
    with MAPPING.open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["row"]) for row in rows] != list(range(171)):
        raise ValueError("TOX_171 mapping must identify all 171 rows in order")
    if len({row["geo_sample"] for row in rows}) != 171:
        raise ValueError("TOX_171 GEO accessions must be unique")
    donors = {row["individual"] for row in rows}
    if len(donors) != 57:
        raise ValueError("TOX_171 requires 57 source donors")
    for donor in donors:
        measurements = [row for row in rows if row["individual"] == donor]
        if (
            len(measurements) != 3
            or {r["agent"] for r in measurements} != {"none", "IR", "UV"}
            or len({r["target"] for r in measurements}) != 1
        ):
            raise ValueError(f"Invalid TOX_171 donor measurements: {donor}")
    return rows


def verify_values(X, y):
    manifest = json.loads(MANIFEST.read_text())
    X, y = (np.asarray(X), np.asarray(y).reshape(-1))
    if X.shape != (171, 5748) or y.shape != (171,):
        raise ValueError("TOX_171 requires the verified 171 x 5748 source matrix")
    encoded = np.unique(y, return_inverse=True)[1]
    values = (
        ("X_values_sha256", np.ascontiguousarray(X, dtype="<f8")),
        ("encoded_y_values_sha256", np.ascontiguousarray(encoded, dtype="<i8")),
    )
    for key, array in values:
        if hashlib.sha256(array.tobytes()).hexdigest() != manifest[key]:
            raise ValueError(
                "TOX_171 measurements/row order or labels changed; reverify donor mapping"
            )
    rows = load_source_mapping()
    if not np.array_equal(encoded, [int(row["target"]) for row in rows]):
        raise ValueError("TOX_171 labels disagree with source donor groups")
    return rows


def grouped_splits(X, y, *, seed=42):
    from tabbiomed.cv_splits import make_grouped_outer_cv

    rows = verify_values(X, y)
    result = make_grouped_outer_cv(
        [r["individual"] for r in rows], y, "classification", seed=seed
    )
    result.update(
        group_split_protocol=PROTOCOL,
        group_source="GEO GDS968:individual",
        source_sample_ids=[r["geo_sample"] for r in rows],
    )
    return result


def source_metadata():
    manifest = json.loads(MANIFEST.read_text())
    return dict(
        description=DESCRIPTION,
        source="GEO_GDS968_GSE1725",
        distribution_source="scikit-feature TOX_171.mat",
        source_url="https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS968",
        publication_doi=manifest["publication_doi"],
        num_donors=57,
        class_names=CLASS_NAMES,
        split_strategy="cv",
        split_group="donor",
        sample_metadata_file="sample_metadata.csv",
        outer_split_protocol=PROTOCOL,
        donor_mapping_sha256=manifest["mapping_sha256"],
    )
