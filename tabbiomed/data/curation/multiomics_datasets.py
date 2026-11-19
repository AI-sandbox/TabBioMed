from pathlib import Path
from tabbiomed.data.catalog import dataset_info
import numpy as np
import pandas as pd
import json
import argparse

DEFAULT_BASE_DIR = Path("data/raw/MultiOmics_Datasets")
EHR_LOCAL_MULTIOMICS = Path("data/raw/MultiOmics_Datasets")


def find_labels_file(top_dir: Path, dataset_name: str = "") -> Path | None:
    top_dir = Path(top_dir)
    if not top_dir.is_dir():
        return None
    if dataset_name.startswith("GS-"):
        prefix = dataset_name[3:]
        preferred = top_dir / f"{prefix}_label_num.csv"
        if preferred.exists():
            return preferred
    labels_path = top_dir / "labels.csv"
    if labels_path.exists():
        return labels_path
    for f in top_dir.glob("*.csv"):
        if "label" in f.stem.lower():
            return f
    return None


OMICS_SUFFIXES = {"CNV": "_CNV", "Methy": "_Methy", "miRNA": "_miRNA", "mRNA": "_mRNA"}


def get_omics_file_suffix(path: Path) -> str | None:
    stem = path.stem.upper()
    for key, suffix in OMICS_SUFFIXES.items():
        if key.upper() in stem:
            return suffix
    return None


def get_omics_csvs_with_suffix(
    top_dir: Path, labels_path: Path | None
) -> list[tuple[Path, str]]:
    top_dir = Path(top_dir)
    labels_path = Path(labels_path) if labels_path else None
    order = ["_CNV", "_Methy", "_miRNA", "_mRNA"]
    out = []
    for f in top_dir.glob("*.csv"):
        if labels_path and f.resolve() == Path(labels_path).resolve():
            continue
        suf = get_omics_file_suffix(f)
        if suf is not None:
            out.append((f, suf))
    out.sort(key=lambda x: order.index(x[1]) if x[1] in order else 999)
    return out


def infer_task(y: pd.Series) -> str:
    y_clean = y.dropna()
    if len(y_clean) == 0:
        return "regression"
    n_unique = y_clean.nunique()
    if n_unique == 2:
        return "binary"
    try:
        y_numeric = pd.to_numeric(y_clean, errors="coerce")
        if y_numeric.isna().all():
            return "classification"
        if n_unique <= 50 and y_numeric.notna().all():
            return "classification"
        return "regression"
    except Exception:
        return "classification"


def process_one_dataset(
    dataset_name: str, base_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict, list[str]]:
    dataset_info("MultiOmics_" + dataset_name)
    top_dir = base_dir / dataset_name / "Top"
    if not top_dir.is_dir():
        raise FileNotFoundError(f"Top directory not found: {top_dir}")
    labels_path = find_labels_file(top_dir, dataset_name)
    if labels_path is None:
        raise FileNotFoundError(
            f"No labels file found in {top_dir} (expected {{name}}_label_num.csv or *label*.csv)"
        )
    omics_list = get_omics_csvs_with_suffix(top_dir, labels_path)
    if len(omics_list) == 0:
        raise ValueError(
            f"No omics CSV files found in {top_dir} (need CNV, Methy, miRNA, mRNA)"
        )
    dfs = []
    for p, suffix in omics_list:
        if dataset_name == "GS-LGG" and suffix == "_Methy":
            from tabbiomed.data.curation.lgg_methylation import (
                load_verified_methylation,
            )

            df = load_verified_methylation(top_dir.parent / "Verified")
        else:
            df = pd.read_csv(p, index_col=0)
        sample_like = [str(c) for c in df.columns[: min(5, len(df.columns))]]
        if any(
            ("TCGA" in c or (c.count(".") >= 2 and c[0].isalpha()) for c in sample_like)
        ):
            df = df.T
        df.index = df.index.astype(str).str.strip()
        df.columns = [str(c).strip() + suffix for c in df.columns]
        dfs.append(df)
    df_final = dfs[0]
    for df in dfs[1:]:
        df_final = df_final.join(df, how="inner")
    labels_df = pd.read_csv(labels_path)
    target_candidates = [
        c for c in labels_df.columns if c.lower() in ("label", "target", "y", "class")
    ]
    target_col = target_candidates[0] if target_candidates else labels_df.columns[-1]
    id_candidates = [c for c in labels_df.columns if c != target_col]
    if id_candidates:
        label_id_col = id_candidates[0]
        labels_df[label_id_col] = labels_df[label_id_col].astype(str).str.strip()
        labels_indexed = labels_df.set_index(label_id_col)[[target_col]]
        labels_indexed = labels_indexed[~labels_indexed.index.duplicated(keep="first")]
        overlap = df_final.index.intersection(labels_indexed.index)
        if len(overlap) == 0:
            raise ValueError(
                f"No matching sample IDs between omics modalities and {labels_path}"
            )
        df_final = df_final.join(labels_indexed, how="inner")
    elif len(labels_df) == len(df_final):
        df_final = df_final.copy()
        df_final[target_col] = labels_df[target_col].to_numpy()
    else:
        raise ValueError(
            f"Labels table {labels_path} has no sample ID column and row count {len(labels_df)} != {len(df_final)}"
        )
    df_final = df_final.reset_index()
    id_col = df_final.columns[0]
    y_raw = df_final[target_col]
    valid_mask = y_raw.notna() & (y_raw.astype(str).str.strip() != "")
    df_final = df_final.loc[valid_mask].reset_index(drop=True)
    y_series = df_final[target_col].astype(str).str.strip()
    X_combined = df_final.drop(columns=[id_col, target_col])
    task = infer_task(y_series)
    if task in ("binary", "classification"):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y_series.astype(str))
        y_out = y_enc.astype(np.int64)
        n_classes = len(le.classes_)
        if task == "binary" and n_classes != 2:
            task = "classification"
    else:
        y_out = pd.to_numeric(y_series, errors="coerce").values.astype(np.float32)
        n_classes = None
    X_combined = X_combined.loc[:, X_combined.nunique() > 1]
    for c in X_combined.columns:
        X_combined[c] = pd.to_numeric(X_combined[c], errors="coerce")
    X_combined = X_combined.dropna(axis=1, how="all")
    X_out = X_combined.values.astype(np.float32)
    metadata = {
        "name": f"MultiOmics_{dataset_name}",
        "task": task,
        "cat_idx": [],
        "num_samples": int(X_out.shape[0]),
        "num_features": int(X_out.shape[1]),
        "source": f"TabPFN-Wide MLOmics {dataset_name} dataset",
        "description": f"MultiOmics dataset: {dataset_name} (Top/ omics + labels)",
    }
    if n_classes is not None:
        metadata["num_classes"] = n_classes
    feature_names = [str(c) for c in X_combined.columns]
    if dataset_name == "GS-LGG":
        from tabbiomed.data.curation.lgg_methylation import (
            source_metadata,
            verify_values,
        )

        verify_values(X_out, y_out)
        metadata.update(source_metadata())
        counters = {}
        for i, name in enumerate(feature_names):
            modality = name.rsplit("_", 1)[-1]
            if modality != "Methy":
                number = counters.get(modality, 0)
                feature_names[i] = f"source_feature_{number:05d}_{modality}"
                counters[modality] = number + 1
    return (X_out, y_out, metadata, feature_names)


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict,
    unified_dir: Path,
    feature_names: list[str] | None = None,
) -> None:
    dataset_info(metadata["name"])
    out_dir = unified_dir / metadata["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    if metadata["name"] == "MultiOmics_GS-BRCA":
        from tabbiomed.data.curation.talent_datasets import _make_talent_split_indices

        np.save(
            out_dir / "split_indices.npy",
            _make_talent_split_indices(y, "classification"),
        )
    if metadata["name"] == "MultiOmics_GS-LGG":
        from .lgg_methylation import retained_splits

        np.save(out_dir / "split_indices.npy", retained_splits())
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    if feature_names is not None:
        with open(out_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
    print(f"  Saved to {out_dir}")


def discover_datasets(base_dir: Path) -> list[str]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    return [
        d.name
        for d in base_dir.iterdir()
        if d.is_dir()
        and d.name in {"GS-BRCA", "GS-GBM", "GS-COAD", "GS-OV", "GS-LGG"}
        and (d / "Top").is_dir()
    ]


def run_all(
    base_dir: Path,
    dataset_names: list[str] | None,
    unified_dir: Path,
    force: bool,
    *,
    raise_on_failure: bool = False,
) -> int:
    base_dir = Path(base_dir)
    unified_dir = Path(unified_dir)
    if dataset_names is None:
        dataset_names = discover_datasets(base_dir)
        if not dataset_names:
            print(f"No datasets found under {base_dir} (expected subfolders with Top/)")
            if raise_on_failure:
                raise SystemExit(1)
            return 0
        print(f"Discovered {len(dataset_names)} datasets: {dataset_names}")
    success = 0
    failures = 0
    for name in dataset_names:
        dataset_info("MultiOmics_" + name)
        out_dir = unified_dir / f"MultiOmics_{name}"
        if out_dir.exists() and (not force):
            print(f"  Skip (exists): MultiOmics_{name}")
            success += 1
            continue
        try:
            X, y, metadata, feature_names = process_one_dataset(name, base_dir)
            save_dataset(X, y, metadata, unified_dir, feature_names=feature_names)
            if name == "GS-LGG":
                from tabbiomed.data.curation.lgg_methylation import (
                    registry,
                    retained_splits,
                )

                manifest, _ = registry()
                output = unified_dir / metadata["name"]
                np.save(output / "split_indices.npy", retained_splits())
                pd.DataFrame(
                    {"row_idx": range(len(y)), "sample_id": manifest["sample_ids"]}
                ).to_csv(output / "sample_metadata.csv", index=False)
            success += 1
        except Exception as e:
            failures += 1
            print(f"  Error processing {name}: {e}")
            import traceback

            traceback.print_exc()
    if raise_on_failure and failures:
        raise SystemExit(1)
    return success


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MultiOmics datasets (Top/ with CSVs + labels.csv) to unified format"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help=f"Base directory (default: {DEFAULT_BASE_DIR}; or datasets/raw/MultiOmics_Datasets)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        help="Dataset name(s) to process (e.g. GS-OV). If omitted and --all not set, processes all discovered.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all discovered datasets under base_dir",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing unified outputs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/unified_format",
        help="Output directory for unified datasets",
    )
    args = parser.parse_args()
    if not args.all and (not args.dataset):
        args.all = True
    return args


def main():
    args = parse_args()
    base_dir = Path(args.base_dir) if args.base_dir else DEFAULT_BASE_DIR
    if not base_dir.exists():
        alt = EHR_LOCAL_MULTIOMICS
        if alt.exists():
            base_dir = alt
            print(f"Using base_dir: {base_dir}")
        else:
            print(f"Base directory not found: {base_dir} (nor {alt})")
            raise SystemExit(1)
    unified_dir = Path(args.output)
    dataset_list = None if args.all else args.dataset or []
    n = run_all(
        base_dir, dataset_list, unified_dir, args.force, raise_on_failure=True
    )
    print(f"Done: {n} dataset(s) written to {unified_dir}")


if __name__ == "__main__":
    main()
