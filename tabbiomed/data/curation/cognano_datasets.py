from pathlib import Path
from tabbiomed.data.catalog import dataset_info
from .paths import CONFIG_DIR
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

FIX_ROOT = Path.cwd()
UNIFIED_DIR = str(FIX_ROOT / "data/unified_format")
AVIDA_RAW_DIR = str(FIX_ROOT / "data/raw/AVIDA")
CDHIT_WORK_DIR = AVIDA_RAW_DIR + "/cdhit"
SARSCOV2_TRAIN_CSV = os.path.join(AVIDA_RAW_DIR, "AVIDa-SARS-CoV-2", "train.csv")
SARSCOV2_TEST_CSV = os.path.join(AVIDA_RAW_DIR, "AVIDa-SARS-CoV-2", "test.csv")
REPOS = {
    "hIL6": "cognano/AVIDa-hIL6",
    "hTNFa": "cognano/AVIDa-hTNFa",
    "SARS-CoV-2": "cognano/AVIDa-SARS-CoV-2",
}
N_MAX_HTNFA = 20000
N_TFIDF_FEATURES = 5000
KMER_K = 3
SARSCOV2_CAPS = {"3mer": {"train": 10000, "val": 10000, "test": 10000}}
def _default_esm2_device() -> str:
    override = os.environ.get("TBM_ESM2_DEVICE")
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


ESM2_MODEL = "esm2_t12_35M_UR50D"
ESM2_DEVICE = _default_esm2_device()
ESM2_BATCH_SZ = 16
RANDOM_SEED = 42
CDHIT_C = 0.8
CDHIT_AS = 0.8
CDHIT_G = 1
SARSCOV2_VAL_FAMILY_C = 0.8
VHH_LEADER = "MKYLLPTAAAGLLLLAAQPAMA"
VHH_TAIL = "HHHHHH"
SPLIT_TRAIN = 0.7
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15
HTNFA_ASSIGNMENT = "label_balanced"
LOCAL_SEARCH_ITERS = 6000
AA_TOKENS = list("ACDEFGHIKLMNPQRSTVWY")
PAD_IDX = 0
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_TOKENS)}
VOCAB_SIZE = len(AA_TOKENS) + 1
SOURCE = "COGNANO AVIDa datasets (HuggingFace: cognano/AVIDa-hIL6, cognano/AVIDa-hTNFa, cognano/AVIDa-SARS-CoV-2). VHH (nanobody) sequences with binary binding labels against antigen variants."
LABEL = "Proteomics"
SUB_LABELS = ["Antigenomics"]
_HTNFA_ASSIGN_DESC = {
    "balanced": "with a balanced scheme (families larger than half the val/test size go to train, the rest in seeded random order)",
    "label_balanced": "by a label-balanced local search (partition sizes and val/test positive rates matched to the overall data)",
}
_DESC_HTNFA = (
    "Binary classification: predict VHH nanobody binding to hTNFa. Single antigen ,  antigen features omitted. {feat_desc}. Split: CD-HIT families (identity 0.8, coverage 0.8, on the VHH core without the constant leader and His-tag) assigned to train/val/test 70/15/15 "
    + _HTNFA_ASSIGN_DESC[HTNFA_ASSIGNMENT]
    + "; VHH families never span splits."
)
_DESC_HIL6 = "Binary classification: predict VHH nanobody binding to one of 31 hIL6 antigen variants (wild type + 30 alanine mutants). {feat_desc}. Split: VHH-family-disjoint ,  VHH cores clustered with CD-HIT (identity 0.9, coverage 0.8), families assigned to train/val/test (about 70/15/15 of VHHs) by a label-balanced search, and validation/test VHHs with >= 0.9 identity to any training VHH removed; all 31 antigens occur in every split. Per-split sample caps apply."
_DESC_SARSCOV2 = "Binary classification: predict VHH nanobody binding to 13 antigens (12 SARS-CoV-2 spike constructs plus HCoV-OC43 as a control) under 17 antigen labels: Alpha, Beta, Delta and Omicron were screened by both bead panning (purified ectodomain, '_bead') and cell panning (full-length membrane spike, '_cell'), and bead/cell constructs have different antigen sequences. {feat_desc}. Split: COGNANO official train/test, which splits by alpaca (Alpaca P train, Alpaca C test); validation (up to 7k samples) is carved from the Alpaca P training set as whole VHH families (CD-HIT identity 0.8 on the VHH core), so no validation VHH occurs in train. After subsampling, exact (VHH sequence, antigen sequence) pairs found in test are removed from training/validation without using labels; test samples are preserved."
FEAT_DESCS = {
    "3mer_vhh": f"Features: TF-IDF of amino-acid {KMER_K}-mers on VHH sequence only (vocabulary and IDF fitted on training rows)",
    "esm2_vhh": f"Features: mean-pooled ESM-2 ({ESM2_MODEL}) embeddings of VHH sequence",
    "onehot_vhh": f"Features: padded integer AA token IDs of VHH sequence (vocab size {VOCAB_SIZE}, 0=PAD); padded/truncated to the maximum training VHH length",
    "3mer": f"Features: [TF-IDF(VHH {KMER_K}-mers) | TF-IDF(Ag {KMER_K}-mers)] hstacked (vocabularies and IDF fitted on training rows)",
    "esm2_paired": f"Features: [mean_pool(ESM-2(VHH)) | mean_pool(ESM-2(Ag))] ({ESM2_MODEL})",
    "onehot_paired": f"Features: padded integer AA token IDs ,  VHH padded/truncated to the maximum training VHH length, Ag padded/truncated to the maximum training Ag length, then concatenated; vocab size {VOCAB_SIZE} (0=PAD). All columns are categorical (cat_idx covers all).",
}


def _tokenize(seq: str, max_len: int) -> list:
    ids = [AA_TO_IDX.get(aa.upper(), PAD_IDX) for aa in seq]
    ids = ids[:max_len]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


def _encode_onehot_single(
    seqs: list[str], train_idx: list[int]
) -> tuple[np.ndarray, list[int]]:
    max_len = max((len(seqs[i]) for i in train_idx))
    X = np.array([_tokenize(s, max_len) for s in seqs], dtype=np.int32)
    return (X, list(range(X.shape[1])))


def _encode_onehot_paired(
    vhh_seqs: list[str], ag_seqs: list[str], train_idx: list[int]
) -> tuple[np.ndarray, list[int]]:
    max_vhh = max((len(vhh_seqs[i]) for i in train_idx))
    max_ag = max((len(ag_seqs[i]) for i in train_idx))
    X_vhh = np.array([_tokenize(s, max_vhh) for s in vhh_seqs], dtype=np.int32)
    X_ag = np.array([_tokenize(s, max_ag) for s in ag_seqs], dtype=np.int32)
    X = np.hstack([X_vhh, X_ag])
    return (X, list(range(X.shape[1])))


def _kmers(seq: str, k: int = KMER_K) -> str:
    if not seq:
        return ""
    return " ".join((seq[i : i + k] for i in range(len(seq) - k + 1)))


def _tfidf_fit_transform_train(
    texts: list[str], train_idx: list[int], max_features: int = N_TFIDF_FEATURES
) -> np.ndarray:
    vec = TfidfVectorizer(max_features=max_features)
    vec.fit([texts[i] for i in train_idx])
    return vec.transform(texts).toarray().astype(np.float32)


def _tfidf_encode(
    seqs: list[str], train_idx: list[int], max_features: int = N_TFIDF_FEATURES
) -> np.ndarray:
    texts = [_kmers(s) for s in seqs]
    return _tfidf_fit_transform_train(texts, train_idx, max_features)


def _encode_3mer_sepconcat(
    vhh_seqs: list[str],
    ag_seqs: list[str],
    train_idx: list[int],
    max_features: int = N_TFIDF_FEATURES,
) -> np.ndarray:
    X_vhh = _tfidf_encode(vhh_seqs, train_idx, max_features)
    X_ag = _tfidf_encode(ag_seqs, train_idx, max_features)
    return np.hstack([X_vhh, X_ag])


def _encode_esm2_pool(sequences: list[str], cache_path: str = None) -> np.ndarray:
    import torch
    import protenc

    encoder = protenc.get_encoder(ESM2_MODEL, device=ESM2_DEVICE)
    n, all_embeds = (len(sequences), [])
    for start in range(0, n, ESM2_BATCH_SZ):
        chunk_idx = start // ESM2_BATCH_SZ
        chunk = sequences[start : start + ESM2_BATCH_SZ]
        if cache_path is not None:
            chunk_file = f"{cache_path}_chunk{chunk_idx:05d}.npy"
            if os.path.exists(chunk_file):
                cached = np.load(chunk_file)
                if len(cached) == len(chunk):
                    all_embeds.append(cached)
                    print(
                        f"    chunk {chunk_idx} ({start}-{start + len(chunk) - 1}): from cache"
                    )
                    continue
                else:
                    print(
                        f"    chunk {chunk_idx} size mismatch (cached {len(cached)} != req {len(chunk)}); re-encoding …"
                    )
        embeds = np.stack(
            [e.mean(0) for e in encoder(chunk, return_format="numpy")], axis=0
        ).astype(np.float32)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.save(chunk_file, embeds)
        all_embeds.append(embeds)
        print(f"    chunk {chunk_idx} ({start}-{start + len(chunk) - 1}/{n}) done")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(all_embeds, axis=0)


def _load_antigen_df(hf_path: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(hf_path, data_files="antigen_sequences.csv", split="train")
    return pd.DataFrame(ds)


def _load_and_merge(hf_path: str, csv_source, n_max: int = None) -> pd.DataFrame:
    if isinstance(csv_source, str) and csv_source.startswith("file://"):
        df_main = pd.read_csv(csv_source[len("file://") :])
    else:
        from datasets import load_dataset

        split = f"train[:{n_max}]" if n_max else "train"
        ds = load_dataset(hf_path, data_files=csv_source, split=split)
        df_main = pd.DataFrame(ds)
    df_ag = _load_antigen_df(hf_path)
    return df_main.merge(df_ag, on="Ag_label", how="left")


def _vhh_core(seq: str) -> str:
    if seq.startswith(VHH_LEADER):
        seq = seq[len(VHH_LEADER) :]
    if seq.endswith(VHH_TAIL):
        seq = seq[: -len(VHH_TAIL)]
    return seq


def _cdhit_groups(seqs, c, name):
    import re
    import subprocess

    if c < 0.7:
        raise ValueError("The configured word size requires identity at least 0.7")
    work = Path(CDHIT_WORK_DIR) / name
    work.mkdir(parents=True, exist_ok=True)
    fasta, prefix = (work / "in.fasta", work / "clusters")
    fasta.write_text(
        "".join((f">{i}\n{sequence}\n" for i, sequence in enumerate(seqs)))
    )
    subprocess.run(
        [
            "cd-hit",
            "-i",
            str(fasta),
            "-o",
            str(prefix),
            "-c",
            str(c),
            "-aS",
            str(CDHIT_AS),
            "-G",
            str(CDHIT_G),
            "-d",
            "0",
            "-sc",
            "1",
            "-T",
            "8",
            "-M",
            "16000",
        ],
        check=True,
    )
    groups = np.full(len(seqs), -1, dtype=np.int64)
    cluster = -1
    for line in prefix.with_suffix(".clstr").read_text().splitlines():
        if line.startswith(">Cluster "):
            cluster = int(line.split()[-1])
        else:
            match = re.search(">(\\d+)\\.\\.\\.", line)
            if match is None or cluster < 0:
                raise ValueError("Invalid CD-HIT cluster record")
            groups[int(match[1])] = cluster
    if (groups < 0).any():
        raise ValueError("CD-HIT did not assign every input sequence")
    return groups


def _members(group_ids: np.ndarray) -> list:
    order = np.argsort(group_ids, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(group_ids[order])) + 1)


def _assign_groups_balanced(group_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)
    groups, n = (_members(group_ids), len(group_ids))
    n_val, n_test = (SPLIT_VAL * n, SPLIT_TEST * n)
    big = [g for g in groups if len(g) > n_val / 2 or len(g) > n_test / 2]
    small = [g for g in groups if not (len(g) > n_val / 2 or len(g) > n_test / 2)]
    ordered = [big[i] for i in rng.permutation(len(big))] + [
        small[i] for i in rng.permutation(len(small))
    ]
    part, size = (np.zeros(n, dtype=np.int64), [0, 0, 0])
    caps = [SPLIT_TRAIN * n, SPLIT_VAL * n]
    for g in ordered:
        k = (
            0
            if size[0] + len(g) <= caps[0]
            else 1
            if size[1] + len(g) <= caps[1]
            else 2
        )
        part[g] = k
        size[k] += len(g)
    return part


def _assign_groups_label_balanced(
    group_ids: np.ndarray,
    label_rate: np.ndarray,
    seed: int,
    iters: int = LOCAL_SEARCH_ITERS,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups, n = (_members(group_ids), len(group_ids))
    K = len(groups)
    size = np.array([len(g) for g in groups], dtype=float)
    pos = np.array([label_rate[g].sum() for g in groups])
    rate = label_rate.mean()
    target = np.array([SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]) * n

    def loss(a):
        total = 0.0
        for k in range(3):
            m = a == k
            sz = size[m].sum()
            total += abs(sz - target[k]) / n
            if k > 0:
                total += abs(pos[m].sum() / sz - rate) / rate if sz > 0 else 10.0
        return total

    a, filled = (np.zeros(K, dtype=int), np.zeros(3))
    for i in np.argsort(-size + rng.random(K) * 0.001):
        k = int(np.argmax(target - filled))
        a[i] = k
        filled[k] += size[i]
    best = loss(a)
    for _ in range(iters):
        b = a.copy()
        if rng.random() < 0.5:
            b[rng.integers(K)] = rng.integers(3)
        else:
            i, j = rng.integers(K, size=2)
            b[i], b[j] = (b[j], b[i])
        lb = loss(b)
        if lb <= best:
            a, best = (b, lb)
    part = np.zeros(n, dtype=np.int64)
    for k, g in zip(a, groups):
        part[g] = k
    return part


def _cluster_split(df: pd.DataFrame, name: str) -> dict:
    cores = [_vhh_core(s) for s in df["VHH_sequence"]]
    groups = _cdhit_groups(cores, CDHIT_C, f"{name}_families")
    if HTNFA_ASSIGNMENT == "balanced":
        part = _assign_groups_balanced(groups, RANDOM_SEED)
    elif HTNFA_ASSIGNMENT == "label_balanced":
        part = _assign_groups_label_balanced(
            groups, df["label"].to_numpy().astype(float), RANDOM_SEED
        )
    else:
        raise ValueError(f"unknown HTNFA_ASSIGNMENT {HTNFA_ASSIGNMENT!r}")
    train_idx, val_idx, test_idx = (
        np.flatnonzero(part == k).tolist() for k in range(3)
    )
    if not (train_idx and val_idx and test_idx):
        raise ValueError(
            f"{name}: cluster split left a partition empty (train:{len(train_idx)} val:{len(val_idx)} test:{len(test_idx)})"
        )
    y = df["label"].to_numpy()
    print(f"  Split ,  train:{len(train_idx)} val:{len(val_idx)} test:{len(test_idx)}")
    print(
        "  positive rate ,  "
        + "  ".join(
            (
                f"{k} {y[i].mean():.3f}"
                for k, i in (("train", train_idx), ("val", val_idx), ("test", test_idx))
            )
        )
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _subsample_stratified(
    df: pd.DataFrame, idx: list, n_max: int, rng: np.random.Generator
) -> list:
    if len(idx) <= n_max:
        return idx
    sub = df.loc[idx].copy()
    result = sub.groupby(["Ag_label", "label"], group_keys=False).apply(
        lambda g: g.sample(
            n=max(1, round(n_max * len(g) / len(sub))),
            random_state=int(rng.integers(1000000)),
        ),
        include_groups=False,
    )
    result = result.sample(frac=1, random_state=int(rng.integers(1000000)))
    return result.index[:n_max].tolist()


def _split_sarscov2() -> tuple[pd.DataFrame, dict]:
    hf_path = REPOS["SARS-CoV-2"]
    df_ag = _load_antigen_df(hf_path)
    df_tr_full = pd.read_csv(SARSCOV2_TRAIN_CSV).merge(df_ag, on="Ag_label", how="left")
    df_te = pd.read_csv(SARSCOV2_TEST_CSV).merge(df_ag, on="Ag_label", how="left")
    n_val_max = min(7000, len(df_tr_full))
    vhh_id, uniq = pd.factorize(df_tr_full["VHH_sequence"])
    families = _cdhit_groups(
        [_vhh_core(s) for s in uniq], SARSCOV2_VAL_FAMILY_C, "SARS-CoV-2_val_families"
    )
    row_family = families[vhh_id]
    family_rows = np.bincount(row_family)
    rng = np.random.default_rng(RANDOM_SEED)
    chosen, total = ([], 0)
    for g in rng.permutation(len(family_rows)):
        if total + family_rows[g] <= n_val_max:
            chosen.append(g)
            total += int(family_rows[g])
    val_mask = np.isin(row_family, chosen)
    df_tr = df_tr_full[~val_mask].reset_index(drop=True)
    df_val = df_tr_full[val_mask].reset_index(drop=True)
    if not set(df_tr["VHH_sequence"]).isdisjoint(df_val["VHH_sequence"]):
        raise ValueError("SARS-CoV-2: validation VHHs also occur in train")
    print(
        f"  validation: {len(chosen)} families, positive rate {df_val['label'].mean():.3f} (train {df_tr['label'].mean():.3f})"
    )
    df = pd.concat([df_tr, df_val, df_te], ignore_index=True)
    n_tr, n_val, n_te = (len(df_tr), len(df_val), len(df_te))
    split = {
        "train": list(range(n_tr)),
        "val": list(range(n_tr, n_tr + n_val)),
        "test": list(range(n_tr + n_val, n_tr + n_val + n_te)),
    }
    print(f"  SARS-CoV-2 split ,  train:{n_tr} val:{n_val} test:{n_te}")
    return (df, split)


def _exclude_test_pair_overlap(
    df: pd.DataFrame, split: dict
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    pairs = list(zip(df["VHH_sequence"], df["Ag_sequence"]))
    test_pairs = {pairs[i] for i in split["test"]}
    kept = {
        role: [i for i in split[role] if pairs[i] not in test_pairs]
        for role in ("train", "val")
    }
    kept["test"] = list(split["test"])
    removed = {role: len(split[role]) - len(kept[role]) for role in ("train", "val")}
    if not all(kept.values()):
        raise ValueError("SARS-CoV-2 pair exclusion left an empty partition")
    print(
        f"  Excluded exact test pairs ,  train:{removed['train']} val:{removed['val']}; test unchanged"
    )
    return _get_subset_data(df, df["label"].to_numpy(dtype=np.int64), kept)


def build_metadata(name, description, X, task, cat_idx=None, num_classes=None):
    meta = {
        "name": name,
        "source": SOURCE,
        "description": description,
        "label": LABEL,
        "sub_labels": SUB_LABELS,
        "task": task,
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "cat_idx": cat_idx if cat_idx is not None else [],
    }
    if num_classes is not None:
        meta["num_classes"] = num_classes
    return meta


def save_dataset(X, y, split_indices, metadata, out_dir):
    dataset_info(metadata["name"])
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X.npy", X)
    np.save(f"{out_dir}/y.npy", y)
    np.save(f"{out_dir}/split_indices.npy", split_indices)
    with open(f"{out_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    si = split_indices
    print(
        f"  Saved → {out_dir}  X={X.shape}  train={len(si['train'])} val={len(si['val'])} test={len(si['test'])}"
    )


def _get_subset_data(
    df: pd.DataFrame, y: np.ndarray, split: dict
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    tr, va, te = (split["train"], split["val"], split["test"])
    all_idx = tr + va + te
    sub_df = df.iloc[all_idx].reset_index(drop=True)
    sub_y = y[all_idx]
    n_tr, n_va, n_te = (len(tr), len(va), len(te))
    new_split = {
        "train": list(range(n_tr)),
        "val": list(range(n_tr, n_tr + n_va)),
        "test": list(range(n_tr + n_va, n_tr + n_va + n_te)),
    }
    return (sub_df, sub_y, new_split)


ALL_DATASETS = [
    "avida_htnfa_3mer",
    "avida_htnfa_esm2",
    "avida_htnfa_onehot",
    "avida_hil6_onehot",
    "avida_sarscov2_3mer",
]


def _build_htnfa(wanted: set) -> list:
    print("\n[hTNFa]")
    hf_path = REPOS["hTNFa"]
    df = _load_and_merge(hf_path, "AVIDa-hTNFa.csv", n_max=N_MAX_HTNFA)
    df = df.reset_index(drop=True)
    split = _cluster_split(df, "hTNFa")
    y = df["label"].values.astype(np.int64)
    vhh = df["VHH_sequence"].tolist()
    results = []
    if "avida_htnfa_3mer" in wanted:
        print("  encoding: 3-mer TF-IDF (VHH only)")
        X = _tfidf_encode(vhh, split["train"])
        desc = _DESC_HTNFA.format(feat_desc=FEAT_DESCS["3mer_vhh"])
        results.append(("avida_htnfa_3mer", X, y, split, [], desc))
    if "avida_htnfa_esm2" in wanted:
        print("  encoding: ESM-2 (VHH only)")
        cache_base = Path(
            os.environ.get("TBM_DATA_CACHE", Path.home() / ".cache/tabbiomed")
        ).expanduser()
        cache = str(cache_base / "esm2_cognano" / "htnfa_vhh")
        X = _encode_esm2_pool(vhh, cache_path=cache)
        desc = _DESC_HTNFA.format(feat_desc=FEAT_DESCS["esm2_vhh"])
        results.append(("avida_htnfa_esm2", X, y, split, [], desc))
    if "avida_htnfa_onehot" in wanted:
        print("  encoding: one-hot token IDs (VHH only)")
        X, cat_idx = _encode_onehot_single(vhh, split["train"])
        desc = _DESC_HTNFA.format(feat_desc=FEAT_DESCS["onehot_vhh"])
        results.append(("avida_htnfa_onehot", X, y, split, cat_idx, desc))
    return results


def _build_hil6(wanted):
    import hashlib

    reference = json.loads((CONFIG_DIR / "avida_hil6_samples.json").read_text())
    raw = Path(AVIDA_RAW_DIR) / "AVIDa-hIL6"
    for filename, expected in reference["source_sha256"].items():
        if hashlib.sha256((raw / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"AVIDa hIL6 source differs: {filename}")
    df = pd.read_csv(raw / "AVIDa-hIL6.csv").merge(
        pd.read_csv(raw / "antigen_sequences.csv"), on="Ag_label", how="left"
    )
    sub, y, split = _get_subset_data(
        df, df["label"].to_numpy(dtype=np.int64), reference["source_rows"]
    )
    X, cat_idx = _encode_onehot_paired(
        sub["VHH_sequence"].tolist(), sub["Ag_sequence"].tolist(), split["train"]
    )
    description = _DESC_HIL6.format(feat_desc=FEAT_DESCS["onehot_paired"])
    return [("avida_hil6_onehot", X, y, split, cat_idx, description)]


def _build_sarscov2(wanted: set) -> list:
    print("\n[SARS-CoV-2]")
    df, full_split = _split_sarscov2()
    y_full = df["label"].values.astype(np.int64)
    results = []
    rng = np.random.default_rng(RANDOM_SEED)

    def _get_ready_data(enc_key: str):
        caps = SARSCOV2_CAPS[enc_key]
        split = {
            k: _subsample_stratified(df, full_split[k], caps[k], rng)
            if caps[k] is not None
            else full_split[k]
            for k in ("train", "val", "test")
        }
        sub_df, _, sub_split = _get_subset_data(df, y_full, split)
        return _exclude_test_pair_overlap(sub_df, sub_split)

    sub_df, y, split = _get_ready_data("3mer")
    if "avida_sarscov2_3mer" in wanted:
        print("  encoding: 3-mer")
        X = _encode_3mer_sepconcat(
            sub_df["VHH_sequence"].tolist(),
            sub_df["Ag_sequence"].tolist(),
            split["train"],
        )
        desc = _DESC_SARSCOV2.format(feat_desc=FEAT_DESCS["3mer"])
        results.append(("avida_sarscov2_3mer", X, y, split, [], desc))
    return results


def _collect_all(wanted: set = None) -> list:
    wanted = set(ALL_DATASETS) if wanted is None else set(wanted)
    results = []
    for prefix, builder in (
        ("avida_htnfa_", _build_htnfa),
        ("avida_hil6_", _build_hil6),
        ("avida_sarscov2_", _build_sarscov2),
    ):
        if any((name.startswith(prefix) for name in wanted)):
            results += builder(wanted)
    return results


def main(wanted: set = None):
    all_datasets = _collect_all(wanted)
    for name, X, y, split, cat_idx, desc in all_datasets:
        meta = build_metadata(
            name=name,
            description=desc,
            X=X,
            task="binary",
            cat_idx=cat_idx,
            num_classes=2,
        )
        save_dataset(X, y, split, meta, os.path.join(UNIFIED_DIR, name))
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ALL_DATASETS,
        default=None,
        help="Only build these datasets (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=UNIFIED_DIR,
        help="Base output directory (default: %(default)s)",
    )
    args = parser.parse_args()
    UNIFIED_DIR = str(args.output_dir)
    main(args.datasets)
