from pathlib import Path
import argparse
import glob
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

REPO_ROOT = Path.cwd()
RAW_DIR = REPO_ROOT / "data/raw/insta_deep_ai/true-cds-protein-tasks/data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/unified_format"
TRANSLATIONS_DIR = (
    REPO_ROOT / "data/raw/insta_deep_ai/true-cds-protein-tasks/true_cds_translations"
)
ESM_CACHE_DIR = REPO_ROOT / "data/raw/esm2_cache"
SPLITS = [("train", "train"), ("validation", "val"), ("test", "test")]
GENETIC_CODE = 1
SOURCE_MPP = "InstaDeepAI/true-cds-protein-tasks (Hugging Face). Original data from the Thermostability Atlas (mass spectrometry-based proteomics). Splits follow the 'mixed' split strategy from FLIP: sequences clustered at 20% identity, 80% of clusters assigned to train and 20% to test."
SOURCE_BETALACTAMASE = "InstaDeepAI/true-cds-protein-tasks (Hugging Face). Original data from Firnberg et al. ,  systematic fitness landscape of all single codon mutations in the TEM-1 Beta-lactamase gene synthesized in E. coli. Fitness measured as a weighted average of allele counts over ampicillin-concentration plates, normalised by WT fitness."
SOURCE_FLUORESCENCE = "InstaDeepAI/true-cds-protein-tasks (Hugging Face). Original data from Sarkisyan et al. ,  experimental study of the avGFP fitness landscape via random mutagenesis synthesized in E. coli. Split follows TAPE/PEER: train on sequences with ≤3 mutations from parent GFP, test on sequences with ≥4 mutations."
TASK_DESCRIPTIONS = {
    "melting_point": "Sequence-level regression task predicting the melting temperature of proteins. Data originates from the Thermostability Atlas, compiled via a mass spectrometry-based proteomic approach. Splits follow the 'mixed' strategy from FLIP: sequences are clustered at 20% identity, with 80% of clusters assigned to train and 20% to test, avoiding over-emphasis of large clusters.",
    "beta_lactamase_unique": "Sequence-level regression task predicting the fitness of all single codon substitutions in the TEM-1 Beta-lactamase gene. Labels measure the ability to confer ampicillin resistance. The Unique dataset contains a random maximal subset of non-degenerate coding sequences (one CDS per protein), using raw CDS fitness values.",
    "fluorescence": "Sequence-level regression task predicting the log-fluorescence of higher-order mutant green fluorescent protein (avGFP) sequences. The library was generated via random mutagenesis of the wildtype sequence. Training is restricted to sequences with three or fewer mutations from parent GFP sequences; the test set contains sequences with four or more mutations, following the TAPE and PEER benchmarks. A random maximal subset of non-degenerate coding sequences was selected.",
}
TRANSLATION_NOTE = "The source provides coding DNA sequences (CDS); they were translated with the standard genetic code before featurization, and nonsense mutants (CDS with a premature in-frame stop codon) were removed from all splits."
ENCODING_DESCRIPTIONS = {
    "esm2": "Features: mean-pooled ESM-2 (esm2_t12_35M_UR50D) embeddings of the translated protein.",
    "3mer": "Features: TF-IDF of amino-acid 3-mers of the translated sequence; vocabulary (3-mers in >= 2 training sequences, top 8,000), an extra '<unk>' term for all other 3-mers, and IDF weights are fitted on the training split.",
}
TASK_SOURCES = {
    "melting_point": SOURCE_MPP,
    "beta_lactamase_unique": SOURCE_BETALACTAMASE,
    "fluorescence": SOURCE_FLUORESCENCE,
}
NATURAL_CDS = {"melting_point"}
DATASETS = {
    "true_betalactamase_unique": ("beta_lactamase_unique", "esm2"),
    "true_fluorescence": ("fluorescence", "3mer"),
    "true_melting_point": ("melting_point", "esm2"),
}
LABEL = "Proteomics"
SUB_LABELS = ["Protein sequence", "Translated CDS"]
ESM2_MODEL = "esm2_t12_35M_UR50D"
ESM2_DEVICE = "cuda"
KMER_K = 3
TFIDF_MAX_FEATS = 8000
TFIDF_MIN_DF = 2
UNK_KMER = "<unk>"


def load_task(task: str) -> pd.DataFrame:
    parts = []
    for hf_split, split in SPLITS:
        files = sorted(glob.glob(str(RAW_DIR / task / f"{hf_split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet for {task}/{hf_split} under {RAW_DIR}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        parts.append(df.assign(split=split))
    return pd.concat(parts, ignore_index=True)


def translate_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    from Bio.Seq import Seq

    cds = df["sequence"].str.upper()
    bad = ~cds.str.fullmatch("[ACGT]+") | (cds.str.len() % 3 != 0)
    if bad.any():
        raise ValueError(
            f"{task}: {int(bad.sum())} CDS with non-ACGT characters or length not divisible by 3"
        )
    if task in NATURAL_CDS:
        protein = [str(Seq(s).translate(table=GENETIC_CODE, cds=True)) for s in cds]
    else:
        protein = [str(Seq(s).translate(table=GENETIC_CODE)) for s in cds]
        if all((p.endswith("*") for p in protein)):
            protein = [p[:-1] for p in protein]
    out = df.copy()
    out["protein"] = protein
    _validate_translation(out, task)
    nonsense = out["protein"].str.contains("*", regex=False)
    counts = pd.DataFrame(
        {
            "source": out["split"].value_counts(),
            "removed": out.loc[nonsense, "split"].value_counts(),
        }
    )
    counts = counts.reindex([s for _, s in SPLITS]).fillna(0).astype(int)
    counts["kept"] = counts["source"] - counts["removed"]
    counts["kept_ratio"] = (counts["kept"] / counts["kept"].sum()).round(3)
    print(
        f"  {task}: removed {int(nonsense.sum())} nonsense mutants\n"
        + counts.to_string()
    )
    return out[~nonsense].reset_index(drop=True)


def _consensus(seqs: pd.Series) -> str:
    lengths = seqs.str.len()
    if lengths.nunique() != 1:
        raise ValueError("consensus needs equal-length sequences")
    cols = zip(*seqs.tolist())
    return "".join((Counter(c).most_common(1)[0][0] for c in cols))


def _validate_translation(df: pd.DataFrame, task: str) -> None:
    if task in NATURAL_CDS:
        assert not df["protein"].str.contains("\\*").any()
        assert df["protein"].str.startswith("M").all()
        print(
            f"  {task}: {len(df)} natural CDS translated (valid start, single terminal stop, no internal stops); lengths {df.protein.str.len().min()}-{df.protein.str.len().max()} aa"
        )
        return
    ref_cds = _consensus(df["sequence"])
    codon_diffs = df["sequence"].map(
        lambda s: sum((s[i : i + 3] != ref_cds[i : i + 3] for i in range(0, len(s), 3)))
    )
    ref_prot = _consensus(df["protein"])
    aa_diffs = df["protein"].map(lambda p: sum((a != b for a, b in zip(p, ref_prot))))
    print(
        f"  {task}: reference protein ({len(ref_prot)} aa) {ref_prot[:30]}...{ref_prot[-15:]}"
    )
    print(
        f"  {task}: premature stops {int(df.protein.str.contains('[*]').sum())}, codon differences to consensus max {codon_diffs.max()}"
    )
    if task.startswith("beta_lactamase"):
        assert codon_diffs.max() <= 1, (
            "beta-lactamase CDS should be single-codon variants"
        )
    if task == "fluorescence":
        by_split = aa_diffs.groupby(df["split"]).agg(["min", "max"])
        print(
            "  fluorescence amino-acid mutations vs reference by split:\n"
            + by_split.to_string()
        )
        assert by_split.loc["train", "max"] <= 3 and by_split.loc["val", "max"] <= 3
        assert by_split.loc["test", "min"] >= 4, (
            "test split should contain >= 4 mutations"
        )


def _kmer_doc(seq: str, k: int = KMER_K) -> list[str]:
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def encode_3mer(
    proteins: list[str], train_idx: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    docs = [_kmer_doc(p) for p in proteins]
    counter = CountVectorizer(
        analyzer=lambda d: d,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATS,
        lowercase=False,
    )
    counter.fit([docs[i] for i in train_idx])
    vocab = set(counter.vocabulary_)
    docs = [[t if t in vocab else UNK_KMER for t in d] for d in docs]
    terms = sorted(vocab) + [UNK_KMER]
    tfidf = TfidfVectorizer(analyzer=lambda d: d, vocabulary=terms, lowercase=False)
    tfidf.fit([docs[i] for i in train_idx])
    X = tfidf.transform(docs).toarray().astype(np.float32)
    unk_train = float((X[train_idx, -1] > 0).mean())
    print(
        f"    3-mer vocabulary {len(vocab)} (+<unk>); training rows with <unk>: {unk_train:.1%}"
    )
    return (X, [f"3mer_{t}" for t in terms])


def encode_esm2(proteins: list[str], cache_name: str) -> np.ndarray:
    unique = sorted(set(proteins))
    ESM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = ESM_CACHE_DIR / f"{cache_name}_{ESM2_MODEL}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        lookup = dict(zip(z["sequences"], z["embeddings"]))
        if all((s in lookup for s in unique)):
            print(f"    ESM-2 embeddings loaded from {cache}")
            return np.stack([lookup[p] for p in proteins]).astype(np.float32)
    import torch
    import protenc

    encoder = protenc.get_encoder(ESM2_MODEL, device=ESM2_DEVICE)
    embeds = {}
    for i, emb in enumerate(
        encoder(unique, average_sequence=True, return_format="numpy")
    ):
        embeds[unique[i]] = emb.astype(np.float32)
        if (i + 1) % 2000 == 0:
            print(f"    ESM-2: {i + 1}/{len(unique)}", flush=True)
            torch.cuda.empty_cache()
    emb_matrix = np.stack([embeds[s] for s in unique])
    np.savez_compressed(
        cache, sequences=np.array(unique, dtype=object), embeddings=emb_matrix
    )
    lookup = dict(zip(unique, emb_matrix))
    return np.stack([lookup[p] for p in proteins]).astype(np.float32)


def build_dataset(
    name: str, task: str, encoding: str, df: pd.DataFrame, output_base: Path
) -> None:
    print(f"\n[{name}] task={task} encoding={encoding}")
    split = df["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    cat_idx = []
    if encoding == "esm2":
        X = encode_esm2(df["protein"].tolist(), cache_name=task)
        feature_names = [f"esm2_{i}" for i in range(X.shape[1])]
    elif encoding == "3mer":
        X, feature_names = encode_3mer(df["protein"].tolist(), train_idx)
    else:
        raise ValueError(encoding)
    y = df["label"].to_numpy(dtype=np.float32)
    split_indices = {s: np.where(split == s)[0].tolist() for _, s in SPLITS}
    meta = {
        "name": name,
        "source": TASK_SOURCES[task],
        "description": f"{TASK_DESCRIPTIONS[task]} {TRANSLATION_NOTE} {ENCODING_DESCRIPTIONS[encoding]}",
        "label": LABEL,
        "sub_labels": SUB_LABELS,
        "task": "regression",
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "cat_idx": cat_idx,
    }
    out_dir = output_base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "split_indices.npy", split_indices)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(
        f"  Saved → {out_dir}  X={X.shape}  "
        + "  ".join((f"{k}={len(v)}" for k, v in split_indices.items()))
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--esm_device",
        type=str,
        help="Device to use for ESM embeddings",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    args = parser.parse_args()
    global ESM2_DEVICE
    ESM2_DEVICE = args.esm_device
    import torch

    if ESM2_DEVICE == "cuda" and (not torch.cuda.is_available()):
        print("CUDA not available, switching to CPU.")
        ESM2_DEVICE = "cpu"
    translated = {}
    for name in args.datasets:
        task, encoding = DATASETS[name]
        if task not in translated:
            print(f"\nTranslating {task} ...")
            translated[task] = translate_task(load_task(task), task)
            TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
            translated[task][["split", "sequence", "protein", "label"]].to_csv(
                TRANSLATIONS_DIR / f"{task}.csv.gz", index=False
            )
        build_dataset(name, task, encoding, translated[task], args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
