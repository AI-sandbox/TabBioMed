from pathlib import Path
import json
import numpy as np

PROTOCOL = "qsar_unique_descriptors_v1"
DESCRIPTION = "Environmental ready-biodegradability classification from 41 molecular descriptors. The TALENT release contains 1,054 rows (UCI lists 1,055); collapsing three redundant descriptor rows with agreeing labels leaves 1,051 rows. Retained values are unchanged. A stratified 70/15/15 split is generated after deduplication. Molecular structures/SMILES are absent from this release, so this is a random-split descriptor benchmark, not a scaffold-generalization test. Grouped with molecular-property tasks under the benchmark drug-discovery domain."


def clean_rows(X, y):
    X, y = (np.asarray(X), np.asarray(y).ravel())
    if (
        X.ndim != 2
        or X.shape[1] != 41
        or len(X) != len(y)
        or (not np.isfinite(X.astype(float)).all())
    ):
        raise ValueError(
            "QSAR requires aligned labels and 41 finite descriptor columns"
        )
    _, first, inverse = np.unique(
        X.astype(float), axis=0, return_index=True, return_inverse=True
    )
    if not np.array_equal(y, y[first][inverse]):
        raise ValueError("Conflicting QSAR labels for identical descriptor rows")
    retained = np.sort(first)
    return (X[retained], y[retained], retained)


def verify_prepared(root, X=None, y=None):
    root = Path(root)
    X = np.load(root / "X.npy", allow_pickle=True) if X is None else X
    y = (
        np.load(root / "y.npy", allow_pickle=True).ravel()
        if y is None
        else np.asarray(y).ravel()
    )
    md = json.loads((root / "metadata.json").read_text())
    unique_X, _, _ = clean_rows(X, y)
    if (
        X.shape != (1051, 41)
        or len(unique_X) != len(X)
        or md.get("curation_protocol") != PROTOCOL
    ):
        raise ValueError(
            "QSAR requires the deduplicated 1,051-row release; prepare_qsar.py"
        )
    if np.bincount(y.astype(int)).tolist() != [698, 353]:
        raise ValueError("Unexpected QSAR class counts")
    from tabbiomed.data.curation.talent_datasets import _make_talent_split_indices

    expected = _make_talent_split_indices(y, "binary")
    actual = np.load(root / "split_indices.npy", allow_pickle=True).item()
    if set(actual) != set(expected) or any(
        (not np.array_equal(actual[k], expected[k]) for k in expected)
    ):
        raise ValueError("QSAR requires its stratified split after deduplication")
