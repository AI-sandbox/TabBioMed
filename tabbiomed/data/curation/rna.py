from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import itertools
import json
import numpy as np
import pandas as pd
from scipy.io import arff

REPO_ROOT = Path.cwd()
RAW_DIR = REPO_ROOT / "data" / "raw" / "rna_hf"
DATASETS_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "unified_format"
DEFAULT_REFERENCE = DATASETS_DIR / "raw" / "hg38" / "hg38.fa"
BASES = "ACGT"
RC = str.maketrans("ACGTN", "TGCAN")
LOCAL_HALF_WINDOW = 32
KMER_K = 3
KMER_HALF_WINDOWS = (256,)
EXCLUDED_CHROMOSOMES = ("chrX", "chrY")
VAL_CHROMOSOMES = ("chr5", "chr6", "chr11")
TEST_CHROMOSOMES = ("chr1", "chr4", "chr21")
VEP_SOURCE = "InstaDeepAI/genomics-long-range-benchmark (Hugging Face), task variant_effect_pathogenic (ClinVar, coding SNVs): ClinVar pathogenic variants vs gnomAD v3.1.2 common variants (MAF > 5%, AN >= 25,000) (Kao et al., 2024). Sequence context from the UCSC hg38 (GRCh38) reference genome."
VEP_DESCRIPTION = "Classify coding single-nucleotide variants as pathogenic (ClinVar, label 1) or common (gnomAD MAF > 5%, label 0) from DNA sequence context only. Variants are strand-canonicalised (reverse-complemented when the reference allele is a purine, so it is always C or T). Features: GRCh38 reference bases at offsets -32..+32 around the variant (the central base is the reference allele) and the alternative allele (66 categorical), plus canonical 3-mer frequencies (k-mer and reverse complement merged) of the 513-bp reference window centred on the variant (32 numeric). Allele-frequency fields, annotation fields (VEP consequence, ClinVar IDs/review status) and genomic coordinates are excluded because they define or reveal the label. chrX/chrY variants (all pathogenic by construction) and two variants listed with both labels are removed. Split by whole chromosomes: test = {test_chroms}; validation = {val_chroms}; train = remaining autosomes."


def save_dataset(X, y, metadata, output_base, feature_names=None, split_indices=None):
    dataset_info(metadata["name"])
    output_dir = Path(output_base) / metadata["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to {output_dir}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Task: {metadata['task']}, Classes: {metadata.get('num_classes', 'N/A')}")
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    if split_indices is not None:
        np.save(output_dir / "split_indices.npy", split_indices)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if feature_names is not None:
        with open(output_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)


def process_bioresponse(output_base):
    candidates = [
        RAW_DIR / "bioresponse.arff",
        RAW_DIR / "dataset_4134_bioresponse.arff",
        RAW_DIR / "phpSSK7iA.arff",
        RAW_DIR / "phpSSK7iA (1).arff",
    ]
    filepath = next((p for p in candidates if p.is_file()), None)
    if filepath is None:
        arffs = sorted(RAW_DIR.glob("*.arff")) if RAW_DIR.is_dir() else []
        filepath = arffs[0] if len(arffs) == 1 else candidates[-1]
    dataset_name = "bioresponse"
    print(f"\nProcessing: {filepath.name}")
    data, _ = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    target_col = "target"
    y = df[target_col].apply(lambda x: int(x.decode("utf-8"))).values
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    n_classes = len(np.unique(y))
    task = "binary" if n_classes == 2 else "classification"
    metadata = {
        "name": dataset_name,
        "task": task,
        "cat_idx": [],
        "source": "OpenML/UCI",
        "description": "Bioresponse dataset: predict biological response to molecules based on chemical properties",
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "num_classes": int(n_classes),
    }
    feature_names = [col for col in df.columns if col != target_col]
    save_dataset(X, y, metadata, output_base, feature_names=feature_names)


def _fetch_windows(df: pd.DataFrame, reference: Path, half: int) -> list[str]:
    import pysam

    fasta = pysam.FastaFile(str(reference))
    seqs = []
    for chrom, pos in zip(df["CHROM"], df["POS"]):
        start, end = (pos - 1 - half, pos + half)
        chrom_len = fasta.get_reference_length(chrom)
        seq = fasta.fetch(chrom, max(start, 0), min(end, chrom_len)).upper()
        seq = "N" * max(0, -start) + seq + "N" * max(0, end - chrom_len)
        seqs.append(seq)
    return seqs


def _revcomp(seq: str) -> str:
    return seq.translate(RC)[::-1]


def _canonical_kmers(k: int = KMER_K) -> list[str]:
    return sorted(
        {
            min(km, _revcomp(km))
            for km in map("".join, itertools.product(BASES, repeat=k))
        }
    )


def _kmer_frequencies(
    seqs: list[str], center: int, half: int, k: int = KMER_K
) -> np.ndarray:
    index = {km: i for i, km in enumerate(_canonical_kmers(k))}
    vocab = {
        km: index[min(km, _revcomp(km))]
        for km in map("".join, itertools.product(BASES, repeat=k))
    }
    X = np.zeros((len(seqs), len(index)), dtype=np.float32)
    for r, s in enumerate(seqs):
        w = s[center - half : center + half + 1]
        for i in range(len(w) - k + 1):
            j = vocab.get(w[i : i + k])
            if j is not None:
                X[r, j] += 1.0
        total = X[r].sum()
        if total > 0:
            X[r] /= total
    return X


def process_vep_pathogenic_coding(output_base, reference: Path):
    filepath = RAW_DIR / "vep_pathogenic_coding.csv"
    dataset_name = "vep_pathogenic_coding"
    print(f"\nProcessing: {filepath.name}")
    df = pd.read_csv(filepath, low_memory=False)
    excluded = df["CHROM"].isin(EXCLUDED_CHROMOSOMES)
    df = df[~excluded].reset_index(drop=True)
    print(
        f"  Dropping {int(excluded.sum())} rows on {', '.join(EXCLUDED_CHROMOSOMES)} (all pathogenic by construction)"
    )
    key = ["CHROM", "POS", "REF", "ALT"]
    conflicting = df.groupby(key)["INT_LABEL"].transform("nunique") > 1
    print(
        f"  Dropping {int(conflicting.sum())} rows of variants listed with both labels: {df.loc[conflicting, key].drop_duplicates().astype(str).agg(':'.join, axis=1).tolist()}"
    )
    df = df[~conflicting].reset_index(drop=True)
    assert not df.duplicated(key).any(), "unexpected duplicate variants"
    assert df["REF"].str.len().eq(1).all() and df["ALT"].str.len().eq(1).all(), (
        "SNVs expected"
    )
    max_half = max(KMER_HALF_WINDOWS + (LOCAL_HALF_WINDOW,))
    seqs = _fetch_windows(df, reference, max_half)
    ref_match = np.array([s[max_half] == r for s, r in zip(seqs, df["REF"])])
    if not ref_match.all():
        raise ValueError(
            f"{int((~ref_match).sum())} REF alleles do not match {reference}"
        )
    print(f"  All {len(df)} REF alleles match the GRCh38 reference")
    flip = df["REF"].isin(["A", "G"]).to_numpy()
    seqs = [_revcomp(s) if f else s for s, f in zip(seqs, flip)]
    alt = np.array(
        [a.translate(RC) if f else a for a, f in zip(df["ALT"], flip)], dtype=object
    )
    assert all((s[max_half] in "CT" for s in seqs)), "canonical REF must be C or T"
    print(f"  Reverse-complemented {int(flip.sum())} purine-REF variants")
    offsets = range(-LOCAL_HALF_WINDOW, LOCAL_HALF_WINDOW + 1)
    local = pd.DataFrame(
        {
            f"ref_{o:+d}" if o else "ref_0": [s[max_half + o] for s in seqs]
            for o in offsets
        }
    )
    local["alt"] = alt
    kmer_blocks, kmer_names = ([], [])
    kmers = _canonical_kmers()
    for half in KMER_HALF_WINDOWS:
        kmer_blocks.append(_kmer_frequencies(seqs, max_half, half))
        kmer_names += [f"k{KMER_K}_w{2 * half + 1}_{km}" for km in kmers]
    X = np.empty((len(df), local.shape[1] + len(kmer_names)), dtype=object)
    X[:, : local.shape[1]] = local.to_numpy(dtype=object)
    X[:, local.shape[1] :] = np.hstack(kmer_blocks).astype(np.float64)
    feature_names = list(local.columns) + kmer_names
    cat_idx = list(range(local.shape[1]))
    y = df["INT_LABEL"].to_numpy(dtype=np.int64)
    is_test = df["CHROM"].isin(TEST_CHROMOSOMES).to_numpy()
    is_val = df["CHROM"].isin(VAL_CHROMOSOMES).to_numpy()
    split_indices = {
        "train": np.where(~is_test & ~is_val)[0].tolist(),
        "val": np.where(is_val)[0].tolist(),
        "test": np.where(is_test)[0].tolist(),
    }
    for name, idx in split_indices.items():
        print(
            f"  {name:5s}: {len(idx):6d} rows ({len(idx) / len(df):.1%}), pathogenic {y[idx].mean():.3f}"
        )
    print(
        f"  validation chromosomes: {list(VAL_CHROMOSOMES)}; test chromosomes: {list(TEST_CHROMOSOMES)}"
    )
    metadata = {
        "name": dataset_name,
        "task": "binary",
        "cat_idx": cat_idx,
        "source": VEP_SOURCE,
        "description": VEP_DESCRIPTION.format(
            test_chroms=", ".join(TEST_CHROMOSOMES),
            val_chroms=", ".join(VAL_CHROMOSOMES),
        ),
        "label": "genomics",
        "num_samples": int(X.shape[0]),
        "num_features": int(X.shape[1]),
        "num_classes": 2,
        "label_mapping": {"Common": 0, "Pathogenic": 1},
    }
    save_dataset(
        X,
        y,
        metadata,
        output_base,
        feature_names=feature_names,
        split_indices=split_indices,
    )
    return


def main():
    print("=" * 60)
    print("RNA_HF Dataset Unification")
    print("=" * 60)
    process_bioresponse(DEFAULT_OUTPUT_DIR)
    process_vep_pathogenic_coding(DEFAULT_OUTPUT_DIR, DEFAULT_REFERENCE)
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Curate rna benchmark data").parse_args()
    main()
