from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import os
import json
import random
import argparse
import numpy as np

REPO_ROOT = Path.cwd()
UNIFIED_DIR = os.path.join(REPO_ROOT, "data/unified_format")
RAW_DIR = os.path.join(REPO_ROOT, "data/raw/MoleculeNet")
CACHE_DIR = os.path.join(REPO_ROOT, "data/raw/MoleculeNet/deepchem_cache")
RANDOM_SEED = 42
SCAFFOLD_FRACS = (0.7, 0.15, 0.15)
ESOL_SPLIT_SEED = 42
SOURCE = "MoleculeNet (https://moleculenet.org) ,  loaded via DeepChem."
LABEL = "Drug Discovery"
SUB_LABELS_QM = ["Quantum Mechanics", "Coulomb Matrix"]
SUB_LABELS_ESOL = ["Physical Chemistry", "Fingerprint"]
SUB_LABELS_HIV = ["Biophysics", "Fingerprint"]
SUB_LABELS_TOX = ["Physiology", "Fingerprint"]
SUB_LABELS_SIDER = ["Physiology", "Fingerprint"]
_QM_TARGET_SPLIT_NOTE = "Target: z-scored with the training-split mean and SD (stored as target_mean / target_std), following DeepChem's default target normalization for this loader. Split: random 70/15/15 (seed 42)."
DATASET_DESCRIPTIONS = {
    "qm8": "Electronic spectra and excited state energy of small molecules calculated by multiple quantum mechanic methods. Task: predict the '{task}' label using Coulomb matrix features. "
    + _QM_TARGET_SPLIT_NOTE,
    "qm9": "Geometric, energetic, electronic and thermodynamic properties of DFT-modelled small molecules. Task: predict '{task}' using Coulomb matrix features. "
    + _QM_TARGET_SPLIT_NOTE,
    "esol": "Water solubility data for common organic small molecules. Target: normalized log solubility (measured log solubility in mols per litre, z-scored with the training-split mean and SD, stored as target_mean / target_std). Features: Morgan fingerprints (radius=2). Split: balanced scaffold split 70/15/15 (chemprop scaffold_balanced, seed 42: scaffold sets larger than half the validation or test size are placed first, the remaining sets follow in random order, filling train, then validation, then test).",
    "hiv": "Experimentally measured abilities of small molecules to inhibit HIV replication. Binary classification task. Features: Morgan fingerprints (radius=2). Split: scaffold-based 70/15/15 (DeepChem ScaffoldSplitter).",
    "tox21": "Qualitative toxicity measurements on 12 biological targets including nuclear receptors and stress response pathways. Task: predict '{task}'. Features: Morgan fingerprints (radius=2).",
    "sider": "Database of marketed drugs and adverse drug reactions (ADR) grouped into 27 system organ classes. Task: predict '{task}'. Features: Morgan fingerprints (radius=2).",
}


def _coulomb_matrix_features(dataset) -> np.ndarray:
    X_list = [np.array(x).flatten() for x in dataset.X]
    max_len = max((len(x) for x in X_list))
    X = np.zeros((len(X_list), max_len), dtype=np.float32)
    for i, x in enumerate(X_list):
        X[i, : len(x)] = x
    return X


def _morgan_features(smiles_list: list, n_bits: int) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            fps.append(np.array(fp, dtype=np.float32))
    return np.array(fps, dtype=np.float32)


def save_dataset(
    name: str, X: np.ndarray, y: np.ndarray, metadata: dict, split_indices: dict = None
):
    dataset_info(metadata["name"])
    out_dir = os.path.join(UNIFIED_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    if split_indices is not None:
        np.save(os.path.join(out_dir, "split_indices.npy"), split_indices)
        si_info = f"  split: train={len(split_indices['train'])} val={len(split_indices['val'])} test={len(split_indices['test'])}"
    else:
        si_info = "  split: not provided"
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"  Saved → {out_dir}  X={X.shape}  y={y.shape}{si_info}")


def build_metadata(
    name: str,
    description: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    label: str,
    sub_labels: list,
    cat_idx: list = None,
) -> dict:
    meta = {
        "name": name,
        "source": SOURCE,
        "description": description,
        "label": label,
        "sub_labels": sub_labels,
        "task": task,
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "cat_idx": cat_idx if cat_idx is not None else [],
    }
    if task in ("binary", "classification"):
        meta["num_classes"] = int(len(np.unique(y)))
    return meta


def _remove_missing_labels(X: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> tuple:
    valid = np.ones(len(y), dtype=bool)
    if np.issubdtype(y.dtype, np.floating):
        valid &= ~np.isnan(y)
    elif y.dtype == object:
        for i, val in enumerate(y):
            try:
                if np.isnan(float(val)):
                    valid[i] = False
            except (TypeError, ValueError):
                valid[i] = False
    if w is not None:
        w_arr = np.array(w)
        if w_arr.ndim == 1:
            valid &= w_arr != 0
        elif w_arr.ndim == 2 and w_arr.shape[1] == 1:
            valid &= w_arr[:, 0] != 0
    n_removed = int((~valid).sum())
    if n_removed > 0:
        print(f"    Removed {n_removed} samples with missing/zero-weight labels.")
    return (X[valid], y[valid], valid)


def _load_smiles_dataset(loader_name: str, subdir: str):
    import deepchem as dc

    loader = getattr(dc.molnet, loader_name)
    tasks, datasets, _ = loader(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter=None,
        transformers=[],
        reload=False,
        data_dir=os.path.join(RAW_DIR, subdir),
        save_dir=os.path.join(CACHE_DIR, subdir),
    )
    parts = list(datasets)
    smiles = [s for ds in parts for s in ds.X]
    y = np.concatenate([ds.y for ds in parts], axis=0)
    w = np.concatenate([ds.w for ds in parts], axis=0)
    return (list(tasks), smiles, y, w)


def _scaffold_split(smiles: list, fracs=SCAFFOLD_FRACS) -> dict:
    import deepchem as dc

    ds = dc.data.NumpyDataset(
        X=np.zeros((len(smiles), 1)), ids=np.array(smiles, dtype=object)
    )
    train, valid, test = dc.splits.ScaffoldSplitter().split(
        ds, frac_train=fracs[0], frac_valid=fracs[1], frac_test=fracs[2]
    )
    split = {
        "train": sorted((int(i) for i in train)),
        "val": sorted((int(i) for i in valid)),
        "test": sorted((int(i) for i in test)),
    }
    n = len(smiles)
    assert sorted(split["train"] + split["val"] + split["test"]) == list(range(n))
    print(
        f"    Scaffold split ,  train:{len(split['train'])} ({len(split['train']) / n:.1%}) val:{len(split['val'])} ({len(split['val']) / n:.1%}) test:{len(split['test'])} ({len(split['test']) / n:.1%})"
    )
    return split


def _balanced_scaffold_split(
    smiles: list, fracs=SCAFFOLD_FRACS, seed: int = ESOL_SPLIT_SEED
) -> dict:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    n = len(smiles)
    sets = {}
    for i, smi in enumerate(smiles):
        sets.setdefault(
            MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False), []
        ).append(i)
    n_train, n_val = (int(fracs[0] * n), int(fracs[1] * n))
    n_test = n - n_train - n_val
    big = [s for s in sets.values() if len(s) > n_val / 2 or len(s) > n_test / 2]
    small = [
        s for s in sets.values() if not (len(s) > n_val / 2 or len(s) > n_test / 2)
    ]
    rng = random.Random(seed)
    rng.shuffle(big)
    rng.shuffle(small)
    train, valid, test = ([], [], [])
    for s in big + small:
        if len(train) + len(s) <= n_train:
            train += s
        elif len(valid) + len(s) <= n_val:
            valid += s
        else:
            test += s
    split = {"train": sorted(train), "val": sorted(valid), "test": sorted(test)}
    assert sorted(train + valid + test) == list(range(n))
    print(
        f"    Balanced scaffold split (seed {seed}) ,  train:{len(train)} ({len(train) / n:.1%}) val:{len(valid)} ({len(valid) / n:.1%}) test:{len(test)} ({len(test) / n:.1%})"
    )
    return split


def _random_split(n: int, fracs=SCAFFOLD_FRACS, seed: int = RANDOM_SEED) -> dict:
    perm = np.random.RandomState(seed).permutation(n)
    n_test, n_val = (int(fracs[2] * n), int(fracs[1] * n))
    split = {
        "train": sorted((int(i) for i in perm[n_test + n_val :])),
        "val": sorted((int(i) for i in perm[n_test : n_test + n_val])),
        "test": sorted((int(i) for i in perm[:n_test])),
    }
    print(
        f"    Random split (seed {seed}) ,  train:{len(split['train'])} val:{len(split['val'])} test:{len(split['test'])}"
    )
    return split


def _standardize_target(y: np.ndarray, split: dict) -> tuple:
    y_train = y[split["train"]].astype(np.float64)
    mean, std = (float(y_train.mean()), float(y_train.std()))
    print(f"    Target z-scored with training mean {mean:.6g} and SD {std:.6g}")
    y_std = ((y.astype(np.float64) - mean) / std).astype(np.float32)
    return (y_std, {"target_mean": mean, "target_std": std})


def _fingerprint_task_datasets(
    loader_name: str,
    subdir: str,
    wanted: list,
    task_type: str,
    desc_key: str,
    split_fn=_scaffold_split,
) -> list:
    tasks, smiles, y_all, w_all = _load_smiles_dataset(loader_name, subdir)
    n_total = len(smiles)
    n_bits = 1024 if n_total > 2000 else 2048
    print(f"    {n_total} molecules, {len(tasks)} tasks. Morgan FP bits: {n_bits}.")
    X_feat = _morgan_features(smiles, n_bits=n_bits)
    smiles_arr = np.array(smiles, dtype=object)
    lower = [t.lower() for t in tasks]
    results = []
    for task_name, ds_name in wanted:
        if task_name.lower() not in lower:
            raise ValueError(f"Task '{task_name}' not found; available: {tasks}")
        t = lower.index(task_name.lower())
        X_clean, y_clean, valid = _remove_missing_labels(
            X_feat, y_all[:, t].astype(np.float64), w=w_all[:, t]
        )
        if task_type == "binary":
            y_clean = y_clean.astype(np.int64)
        else:
            y_clean = y_clean.astype(np.float32)
        print(f"    Task '{tasks[t]}' → '{ds_name}': {len(y_clean)} samples retained.")
        split = split_fn(list(smiles_arr[valid]))
        desc = DATASET_DESCRIPTIONS[desc_key].format(task=tasks[t])
        results.append((ds_name, X_clean, y_clean, split, [], desc, task_type, None))
    return results


QM8_TASKS = ["E1-CC2", "f1-CC2"]
QM9_TASKS = ["gap", "g298"]


def _build_qm(
    loader_name: str,
    subdir: str,
    max_atoms: int,
    wanted: list,
    prefix: str,
    desc_key: str,
) -> list:
    import deepchem as dc

    print(f"\n[{prefix}]")
    print("  Loading dataset via DeepChem CoulombMatrix featurizer …")
    tasks, datasets, _ = getattr(dc.molnet, loader_name)(
        featurizer=dc.feat.CoulombMatrix(max_atoms=max_atoms),
        splitter=None,
        transformers=[],
        reload=True,
        data_dir=os.path.join(RAW_DIR, subdir),
        save_dir=os.path.join(CACHE_DIR, subdir),
    )
    (dataset,) = datasets
    X_feat = _coulomb_matrix_features(dataset)
    print(f"    Full dataset: {X_feat.shape[0]} samples, {len(tasks)} tasks.")
    results = []
    for task_name in wanted:
        if task_name not in tasks:
            print(f"    Task '{task_name}' not found in {prefix}; skipping.")
            continue
        t = list(tasks).index(task_name)
        X_clean, y_clean, _ = _remove_missing_labels(
            X_feat, dataset.y[:, t].astype(np.float64), w=dataset.w[:, t]
        )
        print(f"    Task '{task_name}': {len(y_clean)} samples retained.")
        split = _random_split(len(y_clean))
        y_clean, target_stats = _standardize_target(y_clean, split)
        desc = DATASET_DESCRIPTIONS[desc_key].format(task=task_name)
        results.append(
            (
                f"{prefix}_{task_name}",
                X_clean,
                y_clean,
                split,
                [],
                desc,
                "regression",
                target_stats,
            )
        )
    return results


def build_qm8() -> list:
    return _build_qm("load_qm8", "qm8", 26, QM8_TASKS, "QM8", "qm8")


def build_qm9() -> list:
    return _build_qm("load_qm9", "qm9", 29, QM9_TASKS, "QM9", "qm9")


def build_esol() -> list:
    print("\n[ESOL]")
    results = _fingerprint_task_datasets(
        "load_delaney",
        "esol",
        [("measured log solubility in mols per litre", "ESOL")],
        task_type="regression",
        desc_key="esol",
        split_fn=_balanced_scaffold_split,
    )
    normalized = []
    for name, X, y, split, cat_idx, desc, task_type, _ in results:
        y, target_stats = _standardize_target(y, split)
        normalized.append((name, X, y, split, cat_idx, desc, task_type, target_stats))
    return normalized


def build_hiv() -> list:
    print("\n[HIV]")
    return _fingerprint_task_datasets(
        "load_hiv", "hiv", [("HIV_active", "HIV")], task_type="binary", desc_key="hiv"
    )


TOX21_TASKS = [
    ("NR-AhR", "Tox21_NRAhR"),
    ("NR-ER", "Tox21_NRER"),
    ("SR-MMP", "Tox21_SRMMP"),
]


def build_tox21() -> list:
    print("\n[Tox21]")
    return _fingerprint_task_datasets(
        "load_tox21", "tox21", TOX21_TASKS, task_type="binary", desc_key="tox21"
    )


SIDER_TASKS = [
    ("Nervous system disorders", "SIDER_nervous"),
    ("Gastrointestinal disorders", "SIDER_gastro"),
]


def build_sider() -> list:
    print("\n[SIDER]")
    return _fingerprint_task_datasets(
        "load_sider", "sider", SIDER_TASKS, task_type="binary", desc_key="sider"
    )


BUILDERS = {
    "qm8": build_qm8,
    "qm9": build_qm9,
    "esol": build_esol,
    "hiv": build_hiv,
    "tox21": build_tox21,
    "sider": build_sider,
}
DEFAULT_DATASETS = ["qm8", "qm9", "esol", "hiv", "tox21", "sider"]


def main(datasets=DEFAULT_DATASETS):
    print("=" * 60)
    print("MoleculeNet dataset builder (DeepChem)")
    print("=" * 60)
    all_datasets = [d for key in datasets for d in BUILDERS[key]()]
    for name, X, y, split, cat_idx, desc, task_type, target_stats in all_datasets:
        if name.startswith("QM"):
            sl = SUB_LABELS_QM
        elif name == "ESOL":
            sl = SUB_LABELS_ESOL
        elif name == "HIV":
            sl = SUB_LABELS_HIV
        elif name.startswith("Tox21"):
            sl = SUB_LABELS_TOX
        elif name.startswith("SIDER"):
            sl = SUB_LABELS_SIDER
        else:
            sl = []
        meta = build_metadata(
            name=name,
            description=desc,
            X=X,
            y=y,
            task=task_type,
            label=LABEL,
            sub_labels=sl,
            cat_idx=cat_idx,
        )
        if target_stats is not None:
            meta.update(target_stats)
        save_dataset(name, X, y, meta, split_indices=split)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build MoleculeNet datasets for the TabBioMed benchmark"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(BUILDERS),
        default=DEFAULT_DATASETS,
        help="Which dataset(s) to build (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=UNIFIED_DIR,
        help="Base output directory (default: %(default)s)",
    )
    args = parser.parse_args()
    UNIFIED_DIR = str(args.output_dir)
    main(args.dataset)
