"""Dataset inventory figures."""

from __future__ import annotations
from tabbiomed.analysis.paths import PLAN, PROCESSED, OUTPUT as _OUTPUT_ROOT
from tabbiomed.analysis.datasets import load_benchmark
from tabbiomed.analysis.datasets import dataset_directory
from tabbiomed.analysis.common import analysis_dataset_metadata
import argparse
import json
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    MaxNLocator,
    MultipleLocator,
    NullLocator,
)
import numpy as np
import pandas as pd

BASE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 22
TITLE_FONT_SIZE = 24
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 16
LEGEND_TITLE_FONT_SIZE = 17
ANNOTATION_FONT_SIZE = 18
SOURCE_XTICK_FONT_SIZE = 17
D1_AXIS_LABEL_FONT_SIZE = 27
D1_TITLE_FONT_SIZE = 29
D1_TICK_FONT_SIZE = 23
D1_TASK_LABEL_FONT_SIZE = 21
D1_ANNOTATION_FONT_SIZE = 22
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 700,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_TITLE_FONT_SIZE,
        "figure.titlesize": TITLE_FONT_SIZE,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "text.usetex": False,
        "legend.frameon": False,
    }
)
TASK_ORDER = ["binary", "classification", "regression", "multiregression"]
TASK_LABELS = {
    "binary": "Binary",
    "classification": "Multiclass",
    "regression": "Regression",
    "multiregression": "Multiregression",
}
TASK_MARKERS = {
    "binary": "o",
    "classification": "s",
    "regression": "^",
    "multiregression": "P",
}
DEFAULT_TASK_MARKER = "X"
TASK_COLORS = {
    "binary": "#008577",
    "classification": "#7B3294",
    "regression": "#E76F51",
    "multiregression": "#A6BD39",
}
TASK_BAR_COLORS = TASK_COLORS
DOMAIN_LABEL_DISPLAY = {
    "clinical & healthcare": "Clinical & Healthcare",
    "drug discovery": "Drug Discovery",
    "genomics": "Genomics",
    "metabolomics": "Metabolomics",
    "proteomics": "Proteomics",
    "single-cell": "Single-cell",
    "systems biology": "Systems Biology",
    "transcriptomics": "Transcriptomics",
}
DOMAIN_COLORS = {
    "Clinical & Healthcare": "#1b9e77",
    "Drug Discovery": "#d95f02",
    "Genomics": "#7570b3",
    "Metabolomics": "#e7298a",
    "Proteomics": "#66a61e",
    "Single-cell": "#e6ab02",
    "Systems Biology": "#a6761d",
    "Transcriptomics": "#666666",
}
DEFAULT_SOURCE_MAP = {
    "ALLAML": "scikit-feature",
    "CLL_SUB_111": "scikit-feature",
    "GLIOMA": "scikit-feature",
    "GLI_85": "scikit-feature",
    "Prostate_GE": "scikit-feature",
    "SMK_CAN_187": "scikit-feature",
    "lung": "scikit-feature",
    "PatientCare": "PatientCare",
    "true_betalactamase_unique": "Firnberg TEM-1",
    "true_fluorescence": "Sarkisyan GFP",
    "true_melting_point": "Meltome Atlas",
    "proteinea_solubility": "DeepSol",
    "vep_pathogenic_coding": "ClinVar/gnomAD",
    "sc_gastrulation_stage": "Gastrulation Atlas",
    "metabric_grade": "METABRIC",
    "MTBLS136": "MetaboLights",
    "MTBLS161": "MetaboLights",
    "MTBLS404": "MetaboLights",
    "ST000369": "Metabolomics Workbench",
    "ST000496": "Metabolomics Workbench",
    "ST001000": "Metabolomics Workbench",
    "ST001047": "Metabolomics Workbench",
    "MultiOmics_GS-BRCA": "TCGA",
    "MultiOmics_GS-COAD": "TCGA",
    "MultiOmics_GS-GBM": "TCGA",
    "MultiOmics_GS-LGG": "TCGA",
    "MultiOmics_GS-OV": "TCGA",
    "sc_muris_age": "Tabula Muris",
    "sc_muris_celltype": "Tabula Muris",
    "sc_muraro_celltype": "Muraro Pancreas",
    "CDC_Diabetes_Health_Indicators": "CDC BRFSS",
    "National_Health_and_Nutrition_Health_Survey": "NHANES",
}
SOURCE_PROVENANCE = {
    "PatientCare": "EHR Dataset for Patient Treatment Classification, Mujiono Sadikin (2020), version 1. Mendeley Data, DOI: 10.17632/7kv3rctx7m.1.",
    "MTBLS136": "MetaboLights accession MTBLS136, processed via CIMCB/MetabComparisonBinaryML.",
    "MTBLS161": "MetaboLights accession MTBLS161, processed via CIMCB/MetabComparisonBinaryML.",
    "MTBLS404": "MetaboLights accession MTBLS404, processed via CIMCB/MetabComparisonBinaryML.",
    "ST000369": "Metabolomics Workbench study ST000369, processed via CIMCB/MetabComparisonBinaryML.",
    "ST000496": "Metabolomics Workbench study ST000496, processed via CIMCB/MetabComparisonBinaryML.",
    "ST001000": "Metabolomics Workbench study ST001000, processed via CIMCB/MetabComparisonBinaryML.",
    "ST001047": "Metabolomics Workbench study ST001047, processed via CIMCB/MetabComparisonBinaryML.",
    "MultiOmics_GS-BRCA": "Biological source: TCGA BRCA cohort, curated through MLOmics (DOI: 10.1038/s41597-025-05235-x). Curated release: TabPFN-Wide MLOmics GS-BRCA dataset",
    "MultiOmics_GS-COAD": "Biological source: TCGA COAD cohort, curated through MLOmics (DOI: 10.1038/s41597-025-05235-x). Curated release: TabPFN-Wide MLOmics GS-COAD dataset",
    "MultiOmics_GS-GBM": "Biological source: TCGA GBM cohort, curated through MLOmics (DOI: 10.1038/s41597-025-05235-x). Curated release: TabPFN-Wide MLOmics GS-GBM dataset",
    "MultiOmics_GS-LGG": "Biological source: TCGA LGG cohort, curated through MLOmics (DOI: 10.1038/s41597-025-05235-x). Curated release: MLOmics GS-LGG with TCGA/Xena HumanMethylation450",
    "MultiOmics_GS-OV": "Biological source: TCGA OV cohort, curated through MLOmics (DOI: 10.1038/s41597-025-05235-x). Curated release: TabPFN-Wide MLOmics GS-OV dataset",
    "sc_muris_age": "Biological source: Tabula Muris Senis kidney Smart-seq2, Tabula Muris Consortium (2020), DOI: 10.1038/s41586-020-2496-1. Curated release: Tabula Muris Consortium (2020), Tabula Muris Senis kidney Smart-seq2 via scUnified",
    "sc_muris_celltype": "Biological source: Tabula Muris Senis kidney Smart-seq2, Tabula Muris Consortium (2020), DOI: 10.1038/s41586-020-2496-1. Curated release: Tabula Muris Consortium (2020), Tabula Muris Senis kidney Smart-seq2 via scUnified",
    "sc_muraro_celltype": "Biological source: Muraro human pancreas atlas (2016), GEO GSE85241, DOI: 10.1016/j.cels.2016.09.002. Curated release: Muraro et al. (2016), human pancreas CEL-Seq2 via scUnified",
    "CDC_Diabetes_Health_Indicators": "Biological source: CDC Behavioral Risk Factor Surveillance System (BRFSS), diabetes health-indicator subset. Curated release: Talent Benchmark Dataset - CDC_Diabetes_Health_Indicators",
    "National_Health_and_Nutrition_Health_Survey": "Biological source: CDC/NCHS NHANES 2013-2014, age-prediction subset (UCI 887). Curated release: Talent Benchmark Dataset - National_Health_and_Nutrition_Health_Survey",
    "true_betalactamase_unique": "Biological source: Firnberg et al. (2014), TEM-1 beta-lactamase mutational fitness measurements. Curated release: InstaDeepAI/true-cds-protein-tasks (Hugging Face). Original data from Firnberg et al. — systematic fitness landscape of all single codon mutations in the TEM-1 Beta-lactamase gene synthesized in E. coli. Fitness measured as a weighted average of allele counts over ampicillin-concentration plates, normalised by WT fitness.",
    "true_fluorescence": "Biological source: Sarkisyan et al. (2016), avGFP mutational fluorescence measurements. Curated release: InstaDeepAI/true-cds-protein-tasks (Hugging Face). Original data from Sarkisyan et al. — experimental study of the avGFP fitness landscape via random mutagenesis synthesized in E. coli. Split follows TAPE/PEER: train on sequences with ≤3 mutations from parent GFP, test on sequences with ≥4 mutations.",
    "true_melting_point": "Biological source: Meltome Atlas, Jarzab et al. (2020), DOI: 10.1038/s41592-020-0801-4. Curated release: InstaDeepAI/true-cds-protein-tasks (Hugging Face). Original data from the Thermostability Atlas (mass spectrometry-based proteomics). Splits follow the 'mixed' split strategy from FLIP: sequences clustered at 20% identity, 80% of clusters assigned to train and 20% to test.",
    "vep_pathogenic_coding": "Biological source: ClinVar pathogenic variants and gnomAD v3.1.2 common variants. Curated release: InstaDeepAI/genomics-long-range-benchmark (Hugging Face), task variant_effect_pathogenic (ClinVar, coding SNVs): ClinVar pathogenic variants vs gnomAD v3.1.2 common variants (MAF > 5%, AN >= 25,000) (Kao et al., 2024). Sequence context from the UCSC hg38 (GRCh38) reference genome.",
}
SOURCE_LABEL_DISPLAY = {"Metabolomics Workbench": "Metab. Workbench"}
SOURCE_BAR_COLOR = "#4C78A8"
EDGE_COLOR = "0.20"
_RE_I = re.IGNORECASE
_SOURCE_RULES: list[tuple[str, list[str | re.Pattern[str]]]] = [
    ("GEO", ["geo_gds", "geo_gse", "ncbi.nlm.nih.gov/geo", "gene expression omnibus"]),
    ("HNOCA", ["hnoca", "human neural organoid cell atlas"]),
    ("CPTAC", ["cptac", "clinical proteomic tumor analysis consortium"]),
    (
        "1KG+HGDP",
        ["1000 genomes", "1000genomes", "1kg+hgdp", "1kg + hgdp", "hgdp", "1kg"],
    ),
    ("OpenML", ["openml"]),
    ("TALENT", [re.compile("\\btalent\\b", _RE_I)]),
    ("PMLB", [re.compile("\\bpmlb\\b", _RE_I)]),
    ("cBioPortal", ["cbioportal", "metabric"]),
    ("AVIDA", ["avida"]),
    (
        "UCI repo",
        [
            "ics.uci.edu",
            "uci repository",
            "uci machine learning repository",
            "uci ml repository",
            "uci dataset",
            re.compile("\\buci\\b", _RE_I),
        ],
    ),
    ("scikit-feature", ["scikit-feature", "scikit feature"]),
    ("MoleculeNet", ["moleculenet", "molecule net"]),
    ("MLOmics", ["mlomics", "ml-omics", "ml omics"]),
    ("scVelo", ["scvelo", "sc.velo", "sc_velo"]),
    ("squidpy", ["squidpy"]),
    (
        "Hugging Face",
        ["hugging face", "huggingface", "hugging-face", "hf.co", "huggingface.co"],
    ),
    (
        "VEP",
        [re.compile("\\bvep\\b", _RE_I), "variant effect predictor", "ensembl-vep"],
    ),
    (
        "PhysioNet",
        [
            "physionet",
            "vitaldb",
            "iaccd",
            "mimic-ii",
            "mimic ii",
            "mimic_ii",
            "mimic-iii",
            "mimic iii",
            "mimic_iii",
            "mimic-iv",
            "mimic iv",
            re.compile("\\bmusic\\b", _RE_I),
        ],
    ),
    ("Mendeley", ["mendeley"]),
    ("UCEC", [re.compile("\\bucec\\b", _RE_I), "uterine corpus endometrial carcinoma"]),
    ("TabPFN-VOCS", ["tabpfn-vocs"]),
    ("CIMCB", ["cimcb", "cimbc"]),
    ("MetaboLights", ["metabolights"]),
    ("scUnified", ["scunified"]),
    ("Tahoe", ["tahoe"]),
]


def load_include_datasets(spec):
    if spec is None:
        return None
    path = Path(spec).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = "dataset" if "dataset" in df.columns else df.columns[0]
            return frozenset(df[col].dropna().astype(str).str.strip())
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return frozenset(str(x).strip() for x in payload if str(x).strip())
        return frozenset(payload["datasets"])
    text = str(spec).strip()
    if "," in text or not path.suffix:
        names = [part.strip() for part in text.split(",") if part.strip()]
        if names:
            return frozenset(names)
    raise FileNotFoundError(f"Include specification not found: {spec}")


def compute_missingness_pct(X: np.ndarray) -> float:
    (n_rows, n_cols) = X.shape
    total = n_rows * n_cols
    if total == 0:
        return 0.0
    missing = 0
    rows_per_chunk = max(1, 8000000 // n_cols)
    for start in range(0, n_rows, rows_per_chunk):
        chunk = X[start : start + rows_per_chunk]
        if np.issubdtype(X.dtype, np.number):
            missing += int(np.count_nonzero(np.isnan(chunk)))
        else:
            mask = pd.isna(chunk)
            missing += int(np.count_nonzero(mask))
            missing += int(np.count_nonzero(chunk[~mask] == ""))
    return 100.0 * missing / total


def load_ready_metadata(
    processed_path: Path, compute_missing: bool, include: frozenset[str] | None = None
) -> pd.DataFrame:
    plan = load_benchmark(root=processed_path)["datasets"]
    selected = sorted(plan if include is None else include)
    rows = []
    for name in selected:
        item = dataset_directory(name, processed_path, arrays=compute_missing)
        if name in plan:
            meta = analysis_dataset_metadata(
                name, plan[name], processed_path, verify_arrays=compute_missing
            )
        else:
            meta = json.loads((item / "metadata.json").read_text())
        task = meta["task"]
        label = str(meta.get("label") or "Custom").strip()
        num_samples = meta["num_samples"]
        num_features = meta["num_features"]
        n_cat = len(meta.get("cat_idx", []))
        cat_ratio = n_cat / num_features
        imbalance_ratio = meta.get("imbalance_ratio")
        missingness_pct = np.nan
        if compute_missing:
            try:
                X = np.load(item / "X.npy", mmap_mode="r", allow_pickle=False)
            except ValueError:
                X = np.load(item / "X.npy", allow_pickle=True)
            missingness_pct = compute_missingness_pct(X)
            del X
        balanced = (
            "Balanced"
            if imbalance_ratio is None or imbalance_ratio <= 2.0
            else "Imbalanced"
        )
        num_classes = meta.get("num_classes")
        src = meta.get("source")
        source_raw = src.strip() if isinstance(src, str) else ""
        if DEFAULT_SOURCE_MAP.get(name) == "scikit-feature":
            source_raw = "scikit-feature"
        source_raw = SOURCE_PROVENANCE.get(name, source_raw)
        rows.append(
            {
                "dataset": name,
                "task": task,
                "label": label,
                "num_samples": num_samples,
                "num_features": num_features,
                "num_classes": num_classes,
                "cat_ratio": cat_ratio,
                "imbalance_ratio": imbalance_ratio,
                "missingness_pct": missingness_pct,
                "balanced": balanced,
                "source_raw": source_raw,
            }
        )
    return pd.DataFrame(rows)


def normalize_domain_key(label: object) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return ""
    s = str(label).strip()
    if not s:
        return ""
    s = re.sub("\\s+", " ", s)
    return s.lower()


def display_domain_label(label: object) -> str:
    key = normalize_domain_key(label)
    if not key:
        return ""
    return DOMAIN_LABEL_DISPLAY.get(key, key.title())


def explain_source_classification(source_raw: str) -> tuple[str, str]:
    if not isinstance(source_raw, str) or not source_raw.strip():
        return ("Other", "empty or missing metadata['source']")
    lowered = source_raw.lower()
    for canonical, patterns in _SOURCE_RULES:
        for p in patterns:
            if isinstance(p, re.Pattern):
                m = p.search(source_raw)
                if m:
                    return (canonical, f"regex {p.pattern!r} matched {m.group()!r}")
            elif p in lowered:
                return (canonical, f"substring {p!r}")
    return ("Other", "no rule matched")


def classify_source(source_raw: str) -> str:
    return explain_source_classification(source_raw)[0]


def load_source_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Source map must be a JSON object")
    return {str(k): str(v) for (k, v) in data.items()}


def resolve_source(
    dataset_name: str, source_raw: str, source_map: dict[str, str]
) -> str:
    if source_map and dataset_name in source_map:
        return source_map[dataset_name]
    return DEFAULT_SOURCE_MAP.get(dataset_name) or classify_source(source_raw)


def prepare_frame(
    processed_path: Path,
    include: frozenset[str],
    compute_missing: bool,
    source_map: dict[str, str],
) -> pd.DataFrame:
    df = load_ready_metadata(
        processed_path, compute_missing=compute_missing, include=include
    )
    if df.empty:
        return df
    df = df[df["dataset"].isin(include)].copy()
    df = df[df["label"] != "Unlabeled"].copy()
    df["source_raw"] = df["source_raw"].fillna("").astype(str)
    df["source"] = [
        resolve_source(str(r["dataset"]), str(r["source_raw"]), source_map)
        for (_, r) in df.iterrows()
    ]
    df["num_samples"] = pd.to_numeric(df["num_samples"], errors="coerce")
    df["num_features"] = pd.to_numeric(df["num_features"], errors="coerce")
    df["num_classes"] = pd.to_numeric(df["num_classes"], errors="coerce")
    df["cat_ratio"] = pd.to_numeric(df["cat_ratio"], errors="coerce").fillna(0.0)
    df["cat_pct"] = (df["cat_ratio"] * 100.0).clip(0, 100)
    df["missingness_pct"] = pd.to_numeric(df["missingness_pct"], errors="coerce")
    return df


def inventory_to_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ns = pd.to_numeric(df["num_samples"], errors="coerce")
    nf = pd.to_numeric(df["num_features"], errors="coerce")
    log_s = np.log10(np.maximum(1.0, ns.fillna(1.0).astype(float)))
    log_f = np.log10(np.maximum(1.0, nf.fillna(1.0).astype(float)))
    cr = df["cat_ratio"].fillna(0.0).astype(float)
    pct_num = (1.0 - cr) * 100.0
    pct_cat = cr * 100.0
    miss = pd.to_numeric(df["missingness_pct"], errors="coerce")
    task = df["task"].astype(str)
    nc = pd.to_numeric(df["num_classes"], errors="coerce")
    num_classes = np.where(task == "classification", nc, np.nan)
    dom_disp = df["label"].map(display_domain_label)
    dom_key = dom_disp.map(normalize_domain_key)
    return pd.DataFrame(
        {
            "dataset": df["dataset"].astype(str),
            "task": task,
            "domain_key": dom_key,
            "domain_display": dom_disp,
            "log_samples": log_s,
            "log_features": log_f,
            "pct_numerical": pct_num,
            "pct_categorical": pct_cat,
            "pct_missingness": miss,
            "num_classes": num_classes,
        }
    )


def boxed_axis(ax: Axes, labelsize: float = TICK_FONT_SIZE) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("0.25")
    ax.tick_params(axis="both", labelsize=labelsize, width=1.1)


def _domain_color_map(domains: list[str]) -> dict[str, str]:
    ordered = [
        d
        for d in DOMAIN_COLORS
        if normalize_domain_key(d) in {normalize_domain_key(x) for x in domains}
    ]
    extra = sorted(
        (
            d
            for d in domains
            if d
            and normalize_domain_key(d)
            not in {normalize_domain_key(x) for x in DOMAIN_COLORS}
        )
    )
    final = ordered + extra
    fallback_colors = ["#999999", "#8dd3c7", "#bebada", "#fb8072"]
    return {
        d: DOMAIN_COLORS.get(d, fallback_colors[i % len(fallback_colors)])
        for (i, d) in enumerate(final)
    }


def _bin_edges(series: pd.Series, col: str, n_bins: int) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if v.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    (vmin, vmax) = (float(np.min(v)), float(np.max(v)))
    if col == "num_classes":
        lo = int(np.floor(np.min(v)))
        hi = int(np.ceil(np.max(v)))
        if lo == hi:
            return np.array([lo - 0.5, hi + 0.5])
        return np.arange(lo - 0.5, hi + 1.5, 1.0)
    if vmin == vmax:
        eps = 1e-06 if vmin == 0 else abs(vmin) * 1e-06
        return np.array([vmin - eps, vmax + eps])
    if col in ("pct_numerical", "pct_categorical"):
        return np.linspace(0.0, 100.0, n_bins + 1)
    if col == "pct_missingness":
        return np.linspace(vmin, vmax, n_bins + 1)
    return np.linspace(vmin, vmax, n_bins + 1)


def save_both_formats(
    fig: plt.Figure, out_path: Path, *, extra_artists: list | None = None
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    kwargs = {"bbox_inches": "tight", "pad_inches": 0.18}
    if extra_artists:
        kwargs["bbox_extra_artists"] = extra_artists
    fig.savefig(stem.with_suffix(".pdf"), **kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **kwargs)
    print(f"Saved: {stem.with_suffix('.pdf')}")
    print(f"Saved: {stem.with_suffix('.png')}")


def _stacked_count_bars(
    ax: Axes,
    df_work: pd.DataFrame,
    col: str,
    *,
    color_by: str,
    bin_edges: np.ndarray,
    task_cmap: dict[str, str],
    domain_keys_order: list[str],
    dom_cmap: dict[str, str],
    key_to_display: dict[str, str],
    title: str,
    xlabel: str,
) -> None:
    widths = np.diff(bin_edges)
    x_left = bin_edges[:-1]
    n_bins = len(widths)
    if color_by == "task":
        groups = [t for t in TASK_ORDER if (df_work["task"].astype(str) == t).any()]
        bottom = np.zeros(n_bins)
        for t in groups:
            vals = pd.to_numeric(
                df_work.loc[df_work["task"].astype(str) == t, col], errors="coerce"
            ).dropna()
            if vals.empty:
                continue
            (counts, _) = np.histogram(vals.to_numpy(dtype=float), bins=bin_edges)
            ax.bar(
                x_left,
                counts,
                width=widths,
                bottom=bottom,
                align="edge",
                color=task_cmap[t],
                edgecolor=EDGE_COLOR,
                linewidth=0.45,
                label=TASK_LABELS.get(t, t),
            )
            bottom += counts
        ax.set_ylim(0, max(1.0, float(bottom.max())) * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Dataset Count")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold")
        boxed_axis(ax)
        return
    groups = [
        k
        for k in domain_keys_order
        if k and (df_work["domain_key"].astype(str) == k).any()
    ]
    bottom = np.zeros(n_bins)
    for k in groups:
        vals = pd.to_numeric(
            df_work.loc[df_work["domain_key"].astype(str) == k, col], errors="coerce"
        ).dropna()
        if vals.empty:
            continue
        (counts, _) = np.histogram(vals.to_numpy(dtype=float), bins=bin_edges)
        ax.bar(
            x_left,
            counts,
            width=widths,
            bottom=bottom,
            align="edge",
            color=dom_cmap.get(k, "#999999"),
            edgecolor=EDGE_COLOR,
            linewidth=0.45,
            label=key_to_display.get(k, k),
        )
        bottom += counts
    ax.set_ylim(0, max(1.0, float(bottom.max())) * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Dataset Count")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")
    boxed_axis(ax)


def plot_six_panels_by_task(df: pd.DataFrame, out_path: Path, *, n_bins: int) -> None:
    if df.empty:
        raise ValueError("empty dataframe")
    task_cmap = TASK_BAR_COLORS
    keys = [str(x) for x in df["domain_key"].astype(str)]
    display = [str(x) for x in df["domain_display"].astype(str)]
    uniq_keys: list[str] = []
    key_to_display: dict[str, str] = {}
    for k, d in zip(keys, display):
        if k not in key_to_display:
            key_to_display[k] = d
        if k and k not in uniq_keys:
            uniq_keys.append(k)
    dom_cmap = _domain_color_map([key_to_display.get(k, k) for k in uniq_keys])
    dom_cmap_keyed = {
        k: dom_cmap.get(key_to_display.get(k, k), "#999999") for k in uniq_keys
    }
    legend_colors = [
        task_cmap[t] for t in TASK_ORDER if (df["task"].astype(str) == t).any()
    ]
    legend_labels = [
        TASK_LABELS.get(t, t) for t in TASK_ORDER if (df["task"].astype(str) == t).any()
    ]
    titles = [
        "Number of Samples",
        "Number of Features",
        "Numerical Features",
        "Categorical Features",
        "Missingness",
        "Number of Classes (Multiclass)",
    ]
    xlabels = [
        "$\\log_{10}(n)$, $n \\geq 1$",
        "$\\log_{10}(p)$, $p \\geq 1$",
        "Numerical Features (%)",
        "Categorical Features (%)",
        "Missing Values (%)",
        "Number of Classes",
    ]
    (fig, axes) = plt.subplots(2, 3, figsize=(16.0, 10.0))
    axes_flat = axes.ravel()
    ycols = [
        "log_samples",
        "log_features",
        "pct_numerical",
        "pct_categorical",
        "pct_missingness",
        "num_classes",
    ]
    for ax, col, ttl, xlab in zip(axes_flat, ycols, titles, xlabels):
        if col == "num_classes":
            df_work = df[df["task"] == "classification"].copy()
            if df_work.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No multiclass classification",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=BASE_FONT_SIZE,
                )
                ax.set_title(ttl, fontweight="bold")
                ax.set_ylabel("Dataset Count")
                boxed_axis(ax)
                continue
        elif col == "pct_missingness":
            df_work = df[np.isfinite(pd.to_numeric(df[col], errors="coerce"))].copy()
            if df_work.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Missingness unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=BASE_FONT_SIZE,
                )
                ax.set_title(ttl, fontweight="bold")
                ax.set_ylabel("Dataset Count")
                boxed_axis(ax)
                continue
        else:
            df_work = df
        edges = _bin_edges(df_work[col], col, n_bins)
        _stacked_count_bars(
            ax,
            df_work,
            col,
            color_by="task",
            bin_edges=edges,
            task_cmap=task_cmap,
            domain_keys_order=uniq_keys,
            dom_cmap=dom_cmap_keyed,
            key_to_display=key_to_display,
            title=ttl,
            xlabel=xlab,
        )
        if col == "num_classes":
            ax.xaxis.set_major_locator(MultipleLocator(5))
        if col in {"pct_numerical", "pct_categorical", "pct_missingness"}:
            max_count = max((p.get_y() + p.get_height() for p in ax.patches))
            ax.set_yscale("log", nonpositive="clip")
            ax.set_ylim(0.8, max(85.0, max_count * 1.15))
            ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 80]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel("Dataset Count (log scale)")
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=c, edgecolor=EDGE_COLOR, linewidth=0.45, label=lab
        )
        for (c, lab) in zip(legend_colors, legend_labels)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 8),
        fontsize=BASE_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.tight_layout(rect=[0, 0.085, 1, 0.98])
    save_both_formats(fig, out_path)
    plt.close(fig)


def plot_d1(df: pd.DataFrame, out_path: Path) -> None:
    extra_artists: list = []
    (fig, axes) = plt.subplots(2, 2, figsize=(18.5, 12.0))
    ax = axes[0, 0]
    task_counts = df["task"].apply(lambda t: t if t in TASK_ORDER else "other")
    counts = [int((task_counts == t).sum()) for t in TASK_ORDER]
    x = np.arange(len(TASK_ORDER)) * 3.0
    bar_colors = [TASK_COLORS[t] for t in TASK_ORDER]
    bars = ax.bar(
        x, counts, width=1.08, color=bar_colors, edgecolor=EDGE_COLOR, linewidth=0.9
    )
    ax.set_xticks(x)
    ax.set_xticklabels([])
    for position, task in zip(x, TASK_ORDER):
        marker_box = DrawingArea(14, 16, 0, 0)
        marker_box.add_artist(
            mlines.Line2D(
                [7],
                [8],
                marker=TASK_MARKERS.get(task, DEFAULT_TASK_MARKER),
                linestyle="None",
                markersize=9,
                color="0.35",
            )
        )
        label_box = TextArea(
            TASK_LABELS.get(task, task), textprops={"fontsize": D1_TASK_LABEL_FONT_SIZE}
        )
        key = HPacker(children=[marker_box, label_box], align="center", pad=0, sep=4)
        annotation = AnnotationBbox(
            key,
            (position, 0),
            xycoords=ax.get_xaxis_transform(),
            xybox=(0, -9),
            boxcoords="offset points",
            box_alignment=(0.5, 1),
            frameon=False,
            annotation_clip=False,
        )
        ax.add_artist(annotation)
        extra_artists.append(annotation)
    ax.set_xlim(-0.9, x[-1] + 0.9)
    ax.set_ylabel("Number of Datasets", fontsize=D1_AXIS_LABEL_FONT_SIZE)
    ax.set_title("Datasets per Task Type", fontweight="bold", fontsize=D1_TITLE_FONT_SIZE)
    ymax = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.15, ymax * 0.015),
                str(c),
                ha="center",
                va="bottom",
                fontsize=D1_ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )
    ax.set_ylim(0, ymax * 1.15)
    boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
    ax = axes[0, 1]
    sub = df.dropna(subset=["num_samples", "num_features"]).copy()
    sub = sub[(sub["num_samples"] > 0) & (sub["num_features"] > 0)].copy()
    if not sub.empty:
        sub = sub.assign(domain_plot=sub["label"].map(display_domain_label))
        labels_sorted = [
            d for d in DOMAIN_COLORS if d in set(sub["domain_plot"].astype(str))
        ]
        extra_domains = sorted(
            (
                d
                for d in set(sub["domain_plot"].astype(str))
                if d not in DOMAIN_COLORS and d
            )
        )
        labels_sorted = labels_sorted + extra_domains
        fallback_colors = ["#999999", "#8dd3c7", "#bebada", "#fb8072"]
        label_to_color = {
            lab: DOMAIN_COLORS.get(lab, fallback_colors[i % len(fallback_colors)])
            for (i, lab) in enumerate(labels_sorted)
        }
        for task in TASK_ORDER:
            marker = TASK_MARKERS.get(task, DEFAULT_TASK_MARKER)
            tsub = sub[sub["task"] == task]
            for lab in labels_sorted:
                pts = tsub[tsub["domain_plot"] == lab]
                if pts.empty:
                    continue
                ax.scatter(
                    pts["num_samples"],
                    pts["num_features"],
                    c=[label_to_color[lab]],
                    marker=marker,
                    s=96 if marker == "P" else 88,
                    alpha=0.9,
                    edgecolors=EDGE_COLOR,
                    linewidths=0.7,
                    label="_nolegend_",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Samples (log scale)", fontsize=D1_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel("Number of Features (log scale)", fontsize=D1_AXIS_LABEL_FONT_SIZE)
        ax.set_title(
            "Samples vs. Features", fontweight="bold", fontsize=D1_TITLE_FONT_SIZE
        )
        boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
    else:
        ax.text(
            0.5,
            0.5,
            "No sample/feature data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=BASE_FONT_SIZE,
        )
        boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
    ax = axes[1, 0]
    label_counts = (
        df["label"].map(display_domain_label).value_counts().sort_values(ascending=True)
    )
    if not label_counts.empty:
        y = np.arange(len(label_counts))
        domain_bar_colors = [
            DOMAIN_COLORS.get(str(label), "#999999") for label in label_counts.index
        ]
        bars = ax.barh(
            y,
            label_counts.values,
            height=0.62,
            color=domain_bar_colors,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([label for label in label_counts.index])
        ax.set_xlabel("Number of Datasets", fontsize=D1_AXIS_LABEL_FONT_SIZE)
        ax.set_title("Datasets per Domain", fontweight="bold", fontsize=D1_TITLE_FONT_SIZE)
        xmax = max(label_counts.values) if len(label_counts.values) else 1
        ax.set_xlim(0, xmax * 1.12)
        for bar, c in zip(bars, label_counts.values):
            ax.text(
                bar.get_width() + max(0.15, xmax * 0.012),
                bar.get_y() + bar.get_height() / 2,
                str(int(c)),
                va="center",
                fontsize=D1_ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )
        boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
    else:
        ax.set_title("Datasets per Domain", fontweight="bold")
        boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
    ax = axes[1, 1]
    src_ordered = (
        df["source"]
        .value_counts()
        .sort_index()
        .sort_values(ascending=False, kind="stable")
    )
    if not src_ordered.empty:
        x2 = np.arange(len(src_ordered))
        bars = ax.bar(
            x2,
            src_ordered.values.astype(int),
            width=0.55,
            color=SOURCE_BAR_COLOR,
            edgecolor=EDGE_COLOR,
            linewidth=0.8,
        )
        ax.set_xticks(x2)
        ax.set_xticklabels(
            [SOURCE_LABEL_DISPLAY.get(source, source) for source in src_ordered.index],
            rotation=75,
            ha="right",
            rotation_mode="anchor",
            fontsize=SOURCE_XTICK_FONT_SIZE,
        )
        ax.set_ylabel("Number of Datasets", fontsize=D1_AXIS_LABEL_FONT_SIZE)
        ax.set_title("Datasets per Source", fontweight="bold", fontsize=D1_TITLE_FONT_SIZE)
        ymax = max(src_ordered.values) if len(src_ordered.values) else 1
        for bar, c in zip(bars, src_ordered.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.003,
                str(int(c)),
                ha="center",
                va="bottom",
                fontsize=D1_ANNOTATION_FONT_SIZE,
                fontweight="bold",
            )
        ax.set_ylim(0, ymax * 1.18)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
        ax.tick_params(axis="x", labelsize=SOURCE_XTICK_FONT_SIZE)
    else:
        ax.set_title("Datasets per Source", fontweight="bold")
        boxed_axis(ax, labelsize=D1_TICK_FONT_SIZE)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    save_both_formats(fig, out_path, extra_artists=extra_artists)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dataset characteristics for selected benchmark tasks."
    )
    p.add_argument(
        "--include",
        default=PLAN,
        help="Benchmark JSON profile, CSV dataset list, or comma-separated dataset names",
    )
    p.add_argument("--processed-path", type=Path, default=PROCESSED)
    p.add_argument(
        "--output-dir", "-o", type=Path, default=_OUTPUT_ROOT / "dataset_inventory"
    )
    p.add_argument(
        "--no-missingness-scan",
        action="store_true",
        help="Skip loading X.npy for missingness",
    )
    p.add_argument(
        "--inventory-csv",
        type=Path,
        default=None,
        help="Save the inventory used to generate both figures (defaults to <output-dir>/dataset_summary_data.csv).",
    )
    p.add_argument(
        "--source-map",
        type=Path,
        default=None,
        help="Optional JSON overrides dataset→source bucket",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=12,
        metavar="N",
        help="Bin count for continuous metrics in the six-panel plot.",
    )
    args = p.parse_args()
    include = load_include_datasets(args.include)
    source_map = load_source_map(args.source_map)
    df = prepare_frame(
        args.processed_path,
        include,
        compute_missing=not args.no_missingness_scan,
        source_map=source_map,
    )
    if df.empty:
        raise SystemExit(
            "No datasets after filtering include list + non-Unlabeled labels. Check processed metadata and include file."
        )
    missing_inc = sorted(include - set(df["dataset"].astype(str)))
    if missing_inc:
        raise SystemExit(
            f"{len(missing_inc)} included dataset(s) missing READY metadata or Unlabeled: {missing_inc}"
        )
    if not args.no_missingness_scan and df["missingness_pct"].isna().any():
        missing_scan = df.loc[df["missingness_pct"].isna(), "dataset"].tolist()
        raise SystemExit(f"Missingness scan failed for: {missing_scan}")
    out_dir = Path(args.output_dir)
    inventory_csv = (
        Path(args.inventory_csv)
        if args.inventory_csv is not None
        else out_dir / "dataset_summary_data.csv"
    )
    inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(inventory_csv, index=False)
    print(f"Saved inventory: {inventory_csv} ({len(df)} datasets)")
    plot_d1(df, out_dir / "dataset_inventory_overview_d1")
    summary_df = inventory_to_summary_metrics(df)
    n_bins = max(3, int(args.bins))
    plot_six_panels_by_task(
        summary_df, out_dir / "dataset_inventory_six_panels_by_task", n_bins=n_bins
    )


if __name__ == "__main__":
    main()
