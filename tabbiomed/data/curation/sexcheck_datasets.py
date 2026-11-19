from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR, GENOMIC_ANNOTATIONS_DIR
import csv
import hashlib
import json
import numpy as np
from sklearn.model_selection import train_test_split

NAME = "sexcheck_chrX_50k"
NAMES = ("sexcheck_chrX_50", "sexcheck_chrX_500", NAME)
PROTOCOL = "sexcheck_individual_fixed_v1"
SPLIT_NOTE = "Three independently masked rows represent each of 4,026 individuals. A seed-42, sex-stratified 70/15/15 split assigns individuals before expanding to rows; every individual stays in one train/validation/test partition. Individual IDs are split metadata, never predictors. X/y and masks are preserved."


def split_note(name):
    if name == NAME:
        return SPLIT_NOTE
    if name not in NAMES:
        raise ValueError(f"Unknown sex-check task: {name}")
    return "A seed-42, sex-stratified 70/15/15 split assigns the 4,026 individuals to 2,818 training, 604 validation and 604 test rows, using the same individual assignments as sexcheck_chrX_50k. Individual IDs are split metadata, never predictors. X/y are preserved."


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_individuals(raw, name=NAME):
    raw = Path(raw)
    labels = {}
    with (
        GENOMIC_ANNOTATIONS_DIR / "integrated_call_samples_v3.20250704.ALL.ped"
    ).open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            labels[row["Individual ID"]] = row["Gender"]
    with (GENOMIC_ANNOTATIONS_DIR / "igsr_HGDP_projct_samples.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            labels[row["Sample name"]] = {"M": "1", "F": "2", "1": "1", "2": "2"}.get(
                row["Sex"], "0"
            )
    if name not in NAMES:
        raise ValueError(f"Unknown sex-check task: {name}")
    with (raw / f"{name.removeprefix('sexcheck_')}.psam").open() as f:
        samples = [row["#IID"] for row in csv.DictReader(f, delimiter="\t")]
    keep = [i for i, sample in enumerate(samples) if labels.get(sample) in ("1", "2")]
    ids = [samples[i] for i in keep]
    y = np.asarray([int(labels[sample]) for sample in ids])
    if len(ids) != 4026 or len(set(ids)) != len(ids):
        raise ValueError(
            "Sex-check source must identify 4,026 distinct labelled individuals"
        )
    return (samples, np.asarray(keep), ids, y)


def validate_grouped_split(splits, y):
    y = np.asarray(y)
    if splits.get("group_split_protocol") != PROTOCOL or "groups" not in splits:
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )
    groups = np.asarray(splits["groups"])
    if groups.ndim != 1 or len(groups) != len(y):
        raise ValueError("Expected one individual ID per row")
    parts = {key: np.asarray(splits.get(key, [])) for key in ("train", "val", "test")}
    if any(
        (v.ndim != 1 or not len(v) or v.dtype.kind not in "iu" for v in parts.values())
    ):
        raise ValueError("Each partition needs nonempty integer row indices")
    if not np.array_equal(
        np.sort(np.concatenate(list(parts.values()))), np.arange(len(y))
    ):
        raise ValueError("Fixed splits must partition every row exactly once")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(groups[parts[a]]) & set(groups[parts[b]]):
            raise ValueError(f"Individual crosses {a}/{b} boundary")
    for rows in parts.values():
        if set(y[rows]) != set(y):
            raise ValueError("Every partition must contain both sex labels")
    return groups


def grouped_splits(groups, y, seed=42):
    groups, y = (np.asarray(groups), np.asarray(y))
    if groups.ndim != 1 or y.ndim != 1 or len(groups) != len(y):
        raise ValueError("Expected aligned individual IDs and labels")
    unique, inverse = np.unique(groups, return_inverse=True)
    values = [np.unique(y[inverse == i]) for i in range(len(unique))]
    if any((len(v) != 1 for v in values)) or len(np.unique(y)) != 2:
        raise ValueError("Each individual must have one of two sex labels")
    labels = np.asarray([v[0] for v in values])
    train, heldout = train_test_split(
        np.arange(len(unique)), test_size=0.3, stratify=labels, random_state=seed
    )
    val, test = train_test_split(
        heldout, test_size=0.5, stratify=labels[heldout], random_state=seed
    )
    splits = {
        role: np.flatnonzero(np.isin(inverse, indices)).tolist()
        for role, indices in (("train", train), ("val", val), ("test", test))
    }
    splits.update(
        groups=groups.tolist(),
        group_split_protocol=PROTOCOL,
        group_source="gnomAD PGEN sample order joined to 1KG/IGSR reported sex",
        split_seed=seed,
    )
    validate_grouped_split(splits, y)
    return splits


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    name = metadata["name"]
    if name not in NAMES:
        raise ValueError(f"Unknown sex-check task: {name}")
    splits = np.load(root / "split_indices.npy", allow_pickle=True).item()
    y = np.load(root / "y.npy") if y is None else np.asarray(y)
    groups = validate_grouped_split(splits, y)
    if (
        metadata.get("curation_protocol") != PROTOCOL
        or metadata.get("split_strategy") == "cv"
    ):
        raise ValueError(f"{name} requires the corrected fixed-split metadata")
    receipt = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    mapping = CONFIG_DIR / f"{name}_sample_metadata.csv"
    if digest(mapping) != receipt["mapping_sha256"]:
        raise ValueError(
            "Sex-check individual mapping changed; reverify source identity"
        )
    with mapping.open() as stream:
        rows = list(csv.DictReader(stream))
    encoded = np.unique(y, return_inverse=True)[1]
    if (
        len(rows) != len(y)
        or [int(r["row"]) for r in rows] != list(range(len(y)))
        or (not np.array_equal(groups, [r["individual"] for r in rows]))
        or (not np.array_equal(encoded, [int(r["target"]) for r in rows]))
    ):
        raise ValueError(
            "Sex-check rows/labels/groups disagree with the verified source mapping"
        )
    if splits != grouped_splits(groups, y):
        raise ValueError(
            "Sex-check requires its pinned stratified 70/15/15 individual split"
        )
    shape = (np.load(root / "X.npy", mmap_mode="r") if X is None else X).shape
    version = "unified_format" if root.parent.name == "unified_format" else "processed"
    expected = receipt["arrays"][version]
    if list(shape) != expected["shape"]:
        raise ValueError(
            "Sex-check feature/row dimensions changed; reprepare the repair"
        )
    for name in ("X.npy", "y.npy"):
        if digest(root / name) != expected[name]:
            raise ValueError(
                f"Sex-check {name} changed; reverify source mapping and prepare fresh runs"
            )
    return groups
