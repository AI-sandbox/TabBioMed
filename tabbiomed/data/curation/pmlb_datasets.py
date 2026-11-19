from __future__ import annotations
from pathlib import Path
from .paths import CONFIG_DIR
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd

REPO = Path.cwd()
REGISTRY = CONFIG_DIR / "pmlb_tasks.json"
NAMES = ("dermatology", "spectf", "ecoli")
PROTOCOL = "source_verified_pmlb_v1"
DESCRIPTIONS = {
    "dermatology": "Six-way erythemato-squamous diagnosis after clinical and histopathological examination (366 patients, 34 features). Restore original UCI numeric ages instead of PMLB string-category codes; preserve eight unknown ages as NaN. The 33 other descriptors and all diagnoses match UCI exactly in row order.",
    "spectf": "Normal (0; 55) versus abnormal (1; 212) cardiac SPECT, using 44 supplied rest/stress image summaries per patient. Remove 82 redundant PMLB copies; the 267 retained records exactly match the corrected UCI train/test union. Pool that cohort for benchmark CV rather than using the UCI 80/187 split.",
    "ecoli": "Five-class protein subcellular localization (327 proteins, five supplied sequence-derived features: mcg, gvh, lip, alm1, alm2). Retain PMLB classes cp, im, imU, om, pp; exclude the original imL, imS and omL classes. Omit the supervised aac discriminant because its training population is unverified, and remove constant chg. Protein IDs are never features.",
}


def registry():
    return json.loads(REGISTRY.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_digest(values):
    array = np.asarray(values, dtype="<f8").copy(order="C")
    array[np.isnan(array)] = np.nan
    return hashlib.sha256(
        json.dumps(array.shape).encode() + array.tobytes()
    ).hexdigest()


def fetch_sources(raw_root, names=NAMES):
    reference = registry()
    for name in names:
        for item in reference["datasets"][name]["files"].values():
            path = Path(raw_root) / name / item["filename"]
            if not path.exists():
                request = urllib.request.Request(
                    item["url"], headers={"User-Agent": "TabBioMed/1.0"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = response.read()
                if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError(f"{name}: downloaded source hash mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            if digest(path) != item["sha256"]:
                raise ValueError(
                    f"{name}: source file differs from pinned release: {path}"
                )


def load_corrected(name, raw_root):
    entry = registry()["datasets"][name]
    files = {}
    for key, item in entry["files"].items():
        path = Path(raw_root) / name / item["filename"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"{name}: source file differs from pinned release: {path}")
        files[key] = path
    frame = pd.read_csv(files["pmlb"], sep="\t")
    source_rows = np.arange(len(frame))
    samples = pd.DataFrame({"pmlb_row_index": source_rows})
    with zipfile.ZipFile(files["uci"]) as archive:
        if name == "dermatology":
            raw = pd.read_csv(
                io.BytesIO(archive.read("dermatology.data")), header=None, dtype=str
            )
            numeric = raw.replace("?", np.nan).astype(float)
            np.testing.assert_array_equal(
                numeric.drop(columns=33), frame.drop(columns="Age")
            )
            codes = raw[33].map({v: i for i, v in enumerate(sorted(raw[33].unique()))})
            np.testing.assert_array_equal(codes, frame.Age)
            frame["Age"] = numeric[33].to_numpy()
            samples["uci_row_index"] = source_rows
        elif name == "spectf":
            source = pd.concat(
                [
                    pd.read_csv(io.BytesIO(archive.read(part)), header=None)
                    for part in ("SPECTF.train", "SPECTF.test")
                ],
                ignore_index=True,
            )
            ordered = frame[["target"] + list(frame.drop(columns="target"))]
            if (
                len(source) != 267
                or source.duplicated().any()
                or set(map(tuple, source.to_numpy()))
                != set(map(tuple, ordered.to_numpy()))
            ):
                raise ValueError(
                    "SPECTF does not match the corrected 267-record UCI release"
                )
            keep = ~frame.duplicated()
            samples = samples.loc[keep].reset_index(drop=True)
            frame = frame.loc[keep].reset_index(drop=True)
            lookup = {tuple(row): i for i, row in enumerate(source.to_numpy())}
            samples["uci_pooled_row_index"] = [
                lookup[tuple(row)]
                for row in frame[
                    ["target"] + list(frame.drop(columns="target"))
                ].to_numpy()
            ]
            samples["uci_split"] = np.where(
                samples.uci_pooled_row_index < 80, "train", "test"
            )
        elif name == "ecoli":
            source = pd.read_csv(
                io.BytesIO(archive.read("ecoli.data")), sep="\\s+", header=None
            )
            classes = {int(k): v for k, v in entry["pmlb_target_mapping"].items()}
            keep = source[8].isin(classes.values())
            retained = source.loc[keep].reset_index(drop=True)
            np.testing.assert_allclose(
                retained.iloc[:, 1:8].to_numpy(),
                frame.drop(columns="target"),
                rtol=0,
                atol=6e-08,
            )
            np.testing.assert_array_equal(retained[8], frame.target.map(classes))
            if retained[0].nunique() != 327 or frame.chg.nunique() != 1:
                raise ValueError(
                    "Ecoli source IDs or constant-column invariant changed"
                )
            samples["uci_row_index"] = np.flatnonzero(keep)
            samples["protein_id"] = retained[0]
            frame = frame.drop(columns=["aac", "chg"])
        else:
            raise ValueError(f"Unsupported PMLB task: {name}")
    features = frame.drop(columns="target")
    if (
        features.columns.tolist() != entry["feature_names"]
        or features.duplicated().any()
    ):
        raise ValueError(f"{name}: feature order or uniqueness changed")
    labels = sorted((int(k) for k in entry["pmlb_target_mapping"]))
    y = frame.target.map({label: i for i, label in enumerate(labels)}).to_numpy(
        dtype=np.int64
    )
    X = features.to_numpy(dtype=np.float64)
    verify_values(name, X, y)
    return (X, y, samples)


def verify_values(name, X, y):
    entry = registry()["datasets"][name]
    if (
        np.shape(X) != (entry["num_samples"], entry["num_features"])
        or np.shape(y) != (entry["num_samples"],)
        or value_digest(X) != entry["X_values_sha256"]
        or (value_digest(y) != entry["y_values_sha256"])
    ):
        raise ValueError(
            f"{name}Regenerate this dataset from its pinned raw source, then prepare it."
        )


def task_metadata(name):
    reference = registry()
    entry = reference["datasets"][name]
    return dict(
        name=name,
        description=DESCRIPTIONS[name]
        + " Use five stratified outer folds and three inner folds; fit imputation, encoding and scaling within training partitions.",
        task=entry["task"],
        num_samples=entry["num_samples"],
        num_features=entry["num_features"],
        num_classes=len(entry["target_mapping"]),
        cat_idx=entry["cat_idx"],
        target_mapping=entry["target_mapping"],
        label=entry["label"],
        source=f"PMLB ({name}); verified against original UCI release",
        url=entry["uci_url"],
        license="CC BY 4.0",
        source_url=entry["files"]["pmlb"]["url"],
        curation_protocol=PROTOCOL,
        source_manifest_sha256=digest(REGISTRY),
        source_num_samples=entry["source_num_samples"],
        source_num_features=entry["source_num_features"],
        feature_representation=entry["corrections"],
        sample_metadata_file="sample_metadata.csv",
        split_strategy="cv",
        READY=False,
    )


def ingest(name, raw_root, output_root, *, force=False):
    target = Path(output_root) / name
    if target.exists() and (not force):
        verify_values(name, np.load(target / "X.npy"), np.load(target / "y.npy"))
        return target
    X, y, samples = load_corrected(name, raw_root)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "X.npy", X)
    np.save(target / "y.npy", y)
    samples.to_csv(target / "sample_metadata.csv", index=False)
    (target / "feature_names.json").write_text(
        json.dumps(registry()["datasets"][name]["feature_names"], indent=2) + "\n"
    )
    (target / "metadata.json").write_text(
        json.dumps(task_metadata(name), indent=2) + "\n"
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=NAMES, default=list(NAMES))
    parser.add_argument("--input", type=Path, default=REPO / "data/raw/PMLB")
    parser.add_argument("--output", type=Path, default=REPO / "data/unified_format")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fetch_sources(args.input, args.datasets)
    for name in args.datasets:
        path = ingest(name, args.input, args.output, force=args.force)
        print(f"Unified {name}: {path}; run the general pipeline for stored CV splits.")


if __name__ == "__main__":
    main()
