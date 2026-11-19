from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import argparse
import hashlib
import json
import shutil
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabbiomed.data.curation.pmlb_datasets import value_digest

REPO = Path.cwd()
REGISTRY = CONFIG_DIR / "metabric_grade.json"
NAME = "metabric_grade"
PROTOCOL = "metabric_clinical_grade_expression_v1"
DESCRIPTION = "Three-class histological grade (1, 2, 3) from primary breast-tumour gene expression in METABRIC: 1,892 distinct patients and 20,603 released expression measurements. Exclude 88 expression-profiled patients without a recorded grade. Use only the supplied log2 microarray intensities, preserving 13 missing cells for training-fitted imputation. Clinical annotations and patient/sample IDs are not features. This is a retrospective tumour molecular/pathology task. Preserve source feature-row identity even when gene annotations repeat. Use a patient-disjoint, grade-stratified 70/15/15 split (1,324/284/284); fit any additional preprocessing only on training rows."


def registry():
    return json.loads(REGISTRY.read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def frame_digest(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def feature_names_digest(names):
    return hashlib.sha256((json.dumps(names, indent=2) + "\n").encode()).hexdigest()


def split_digest(splits):
    return hashlib.sha256(
        json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fetch_sources(raw_root):
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    for item in registry()["files"].values():
        target = raw_root / item["filename"]
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".download")
            try:
                request = urllib.request.Request(
                    item["url"], headers={"User-Agent": "TabBioMed/1.0"}
                )
                with (
                    urllib.request.urlopen(request, timeout=180) as response,
                    temporary.open("wb") as stream,
                ):
                    shutil.copyfileobj(response, stream)
                if digest(temporary) != item["sha256"]:
                    raise ValueError(f"METABRIC source hash mismatch: {target.name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if digest(target) != item["sha256"]:
            raise ValueError(f"Changed METABRIC source: {target}")


def make_splits(y):
    rows = np.arange(len(y))
    train, remaining = train_test_split(
        rows, test_size=0.3, stratify=y, random_state=42
    )
    val, test = train_test_split(
        remaining, test_size=0.5, stratify=y[remaining], random_state=42
    )
    return dict(train=train.tolist(), val=val.tolist(), test=test.tolist())


def source_annotations(raw_root):
    raw_root = Path(raw_root)
    expression = raw_root / "data_mrna_illumina_microarray.txt"
    header = pd.read_csv(expression, sep="\t", nrows=0).columns.tolist()
    if header[:2] != ["Hugo_Symbol", "Entrez_Gene_Id"] or len(set(header[2:])) != 1980:
        raise ValueError("Unexpected METABRIC expression sample IDs")
    samples = pd.read_csv(raw_root / "data_clinical_sample.txt", sep="\t", comment="#")
    patients = pd.read_csv(
        raw_root / "data_clinical_patient.txt", sep="\t", comment="#"
    )
    clinical = (
        samples.merge(patients, on="PATIENT_ID", validate="one_to_one")
        .set_index("SAMPLE_ID")
        .loc[header[2:]]
    )
    grade = clinical.GRADE
    if not grade.dropna().isin([1, 2, 3]).all() or int(grade.isna().sum()) != 88:
        raise ValueError("Unexpected METABRIC grade labels")
    keep = grade.notna()
    selected = clinical.loc[keep]
    if (
        selected.PATIENT_ID.nunique() != 1892
        or not selected.SAMPLE_TYPE.eq("Primary").all()
        or (not selected.SEX.eq("Female").all())
    ):
        raise ValueError("Expected distinct primary-tumour patients")
    sample_map = pd.DataFrame(
        dict(
            row_index=np.arange(len(selected)),
            source_expression_column=np.flatnonzero(keep),
            sample_id=selected.index,
            patient_id=selected.PATIENT_ID.to_numpy(),
            histological_grade=selected.GRADE.to_numpy(dtype=int),
            source_cohort=selected.COHORT.to_numpy(dtype=int),
        )
    )
    features = pd.read_csv(
        expression, sep="\t", usecols=[0, 1], dtype=str, keep_default_na=False
    )
    features.insert(0, "source_row_index", np.arange(len(features)))
    features.insert(0, "feature_index", np.arange(len(features)))
    features["feature_name"] = [
        f"metabric_row_{i}__{symbol}__{entrez}"
        for i, symbol, entrez in features[
            ["source_row_index", "Hugo_Symbol", "Entrez_Gene_Id"]
        ].itertuples(index=False, name=None)
    ]
    return (sample_map, features)


def load_source(raw_root):
    for item in registry()["files"].values():
        if digest(Path(raw_root) / item["filename"]) != item["sha256"]:
            raise ValueError(f"Changed METABRIC source: {item['filename']}")
    samples, features = source_annotations(raw_root)
    values = pd.read_csv(
        Path(raw_root) / "data_mrna_illumina_microarray.txt",
        sep="\t",
        usecols=samples.sample_id.tolist(),
        dtype=np.float64,
    )
    if values.columns.tolist() != samples.sample_id.tolist():
        raise ValueError("METABRIC expression/clinical sample order differs")
    X = values.to_numpy().T.copy()
    y = samples.histological_grade.to_numpy(dtype=np.int64) - 1
    verify_values(X, y)
    if (
        frame_digest(samples) != registry()["sample_metadata_sha256"]
        or frame_digest(features) != registry()["feature_metadata_sha256"]
    ):
        raise ValueError("METABRIC sample or feature identity changed")
    return (X, y, samples, features)


def verify_values(X, y):
    reference = registry()
    if (
        np.shape(X) != (1892, 20603)
        or np.shape(y) != (1892,)
        or value_digest(X) != reference["X_values_sha256"]
        or (value_digest(y) != reference["y_values_sha256"])
    ):
        raise ValueError(
            "Regenerate this dataset from its pinned raw source, then prepare it."
        )


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    reference = registry()
    if X is None:
        X, y = (np.load(root / "X.npy"), np.load(root / "y.npy"))
    verify_values(X, y)
    md = json.loads((root / "metadata.json").read_text())
    manifest_hash = digest(REGISTRY)
    source_manifest_matches = md.get("source_manifest_sha256") == manifest_hash
    if (
        md.get("curation_protocol") != PROTOCOL
        or md.get("task") != "classification"
        or (not source_manifest_matches)
        or (md.get("feature_representation") != "source_log2_microarray_intensity")
        or (md.get("license") != "ODbL-1.0")
        or (md.get("split_strategy") != "fixed")
    ):
        raise ValueError("METABRIC requires the pinned grade-task metadata")
    for filename, key in [
        ("sample_metadata.csv", "sample_metadata_sha256"),
        ("feature_metadata.csv", "feature_metadata_sha256"),
    ]:
        if digest(root / filename) != reference[key]:
            raise ValueError(f"METABRIC provenance changed: {filename}")
    names = json.loads((root / "feature_names.json").read_text())
    if feature_names_digest(names) != reference["feature_names_sha256"]:
        raise ValueError("METABRIC feature names or order changed")
    splits = np.load(root / "split_indices.npy", allow_pickle=True).item()
    if split_digest(splits) != reference["split_indices_sha256"]:
        raise ValueError(
            "METABRIC requires the pinned patient-stratified 70/15/15 split"
        )
    return splits


def task_metadata():
    return dict(
        name=NAME,
        description=DESCRIPTION,
        task="classification",
        num_samples=1892,
        num_features=20603,
        num_classes=3,
        cat_idx=[],
        target_mapping={"0": "grade 1", "1": "grade 2", "2": "grade 3"},
        target_definition="Released clinical GRADE (1, 2, 3); labels encoded as 0, 1, 2",
        source="METABRIC / cBioPortal (Curtis et al., 2012)",
        source_url=registry()["source_url"],
        url=registry()["source_url"],
        publication_doi="10.1038/nature10983",
        license="ODbL-1.0",
        label="transcriptomics",
        curation_protocol=PROTOCOL,
        source_manifest_sha256=digest(REGISTRY),
        source_revision=registry()["source_revision"],
        source_num_samples=1980,
        source_num_features=20603,
        feature_representation="source_log2_microarray_intensity",
        sample_metadata_file="sample_metadata.csv",
        feature_metadata_file="feature_metadata.csv",
        outer_split_protocol="patient_stratified_70_15_15_v1",
        split_strategy="fixed",
        READY=False,
    )


def ingest(raw_root, output_root, *, force=False):
    target = Path(output_root) / NAME
    if target.exists() and (not force):
        verify_prepared(target)
        return target
    X, y, samples, features = load_source(raw_root)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "X.npy", X)
    np.save(target / "y.npy", y)
    np.save(target / "split_indices.npy", make_splits(y))
    samples.to_csv(target / "sample_metadata.csv", index=False)
    features.to_csv(target / "feature_metadata.csv", index=False)
    (target / "feature_names.json").write_text(
        json.dumps(features.feature_name.tolist(), indent=2) + "\n"
    )
    (target / "metadata.json").write_text(json.dumps(task_metadata(), indent=2) + "\n")
    shutil.copyfile(Path(raw_root) / "LICENSE", target / "LICENSE")
    verify_prepared(target, X, y)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "data/raw/METABRIC")
    parser.add_argument("--output", type=Path, default=REPO / "data/unified_format")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch_sources(args.input)
    print(f"Unified {NAME}: {ingest(args.input, args.output, force=args.force)}")


if __name__ == "__main__":
    main()
