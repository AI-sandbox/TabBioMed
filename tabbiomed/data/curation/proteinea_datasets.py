import os
import json
import numpy as np
import pandas as pd
import requests
from itertools import product
from collections import Counter

SOURCE_SOLUBILITY = "proteinea/solubility (Hugging Face). Binary protein solubility prediction dataset. Labels indicate whether a protein is soluble (1) or insoluble (0). Separate train, validation and test CSV files."
DESCRIPTION_SOLUBILITY = "Sample-level binary classification task predicting protein solubility. Labels: 1 = soluble, 0 = insoluble. Features are k-mer (k=3) frequency vectors computed from single amino-acid sequences. Missing feature values are not imputed."
LABEL = "Proteomics"
SUB_LABELS = ["protein-seq"]
KMER_K = 3
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_CACHE_DIR = "data/raw/proteinea"
DEFAULT_UNIFIED_DIR = "data/unified_format"


def _build_kmer_vocab(k: int) -> list[str]:
    return ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]


def _sequence_to_kmer_freq(seq: str, k: int, vocab_index: dict) -> np.ndarray:
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    seq = str(seq)
    counts = Counter(
        (
            seq[i : i + k]
            for i in range(len(seq) - k + 1)
            if all((c in set(AMINO_ACIDS) for c in seq[i : i + k]))
        )
    )
    total = sum(counts.values())
    if total > 0:
        for kmer, count in counts.items():
            if kmer in vocab_index:
                vec[vocab_index[kmer]] = count / total
    return vec


def build_kmer_features(sequences_list: list[list[str]], k: int = KMER_K) -> np.ndarray:
    vocab = _build_kmer_vocab(k)
    vocab_index = {kmer: i for i, kmer in enumerate(vocab)}
    per_col_matrices = []
    for col_seqs in sequences_list:
        mat = np.stack(
            [_sequence_to_kmer_freq(seq, k, vocab_index) for seq in col_seqs], axis=0
        )
        per_col_matrices.append(mat)
    return np.concatenate(per_col_matrices, axis=1)


def _download_csv(url: str, local_path: str) -> None:
    if os.path.exists(local_path):
        print(f"    Already cached: {local_path}")
        return
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"    Downloading {url} → {local_path} …")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)


def build_metadata(name, source, description, task, X):
    return {
        "name": name,
        "source": source,
        "description": description,
        "label": LABEL,
        "sub_labels": SUB_LABELS,
        "task": task,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "cat_idx": [],
    }


def process_solubility(cache_dir: str, unified_dir: str) -> None:
    name = "proteinea_solubility"
    print(f"\n[{name}]")
    urls = {
        "train": "https://huggingface.co/datasets/proteinea/solubility/resolve/main/solubility_training.csv?download=true",
        "val": "https://huggingface.co/datasets/proteinea/solubility/resolve/main/solubility_validation.csv?download=true",
        "test": "https://huggingface.co/datasets/proteinea/solubility/resolve/main/solubility_testing.csv?download=true",
    }
    dfs = {}
    for split, url in urls.items():
        local = os.path.join(cache_dir, f"solubility_{split}.csv")
        _download_csv(url, local)
        dfs[split] = pd.read_csv(local)
    for split, df in dfs.items():
        print(f"  {split}: {len(df)} rows  columns: {list(df.columns)}")
    for split in dfs:
        dfs[split] = dfs[split].dropna(subset=["sequences", "labels"])
    n_train = len(dfs["train"])
    n_val = len(dfs["val"])
    n_test = len(dfs["test"])
    all_seqs = (
        dfs["train"]["sequences"].tolist()
        + dfs["val"]["sequences"].tolist()
        + dfs["test"]["sequences"].tolist()
    )
    all_labels = pd.concat(
        [dfs["train"]["labels"], dfs["val"]["labels"], dfs["test"]["labels"]]
    ).values
    X = build_kmer_features([all_seqs], k=KMER_K)
    y = all_labels.astype(np.float32)
    split_indices = {
        "train": list(range(n_train)),
        "val": list(range(n_train, n_train + n_val)),
        "test": list(range(n_train + n_val, n_train + n_val + n_test)),
    }
    meta = build_metadata(name, SOURCE_SOLUBILITY, DESCRIPTION_SOLUBILITY, "binary", X)
    out_dir = os.path.join(unified_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X.npy", X)
    np.save(f"{out_dir}/y.npy", y)
    np.save(f"{out_dir}/split_indices.npy", split_indices)
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    print(
        f"  Saved → {out_dir}\n    X={X.shape}  train={n_train}  val={n_val}  test={n_test}"
    )


def main():
    process_solubility(DEFAULT_CACHE_DIR, DEFAULT_UNIFIED_DIR)


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Curate proteinea benchmark data").parse_args()
    main()
