from pathlib import Path
from tabbiomed.data.catalog import dataset_names
import argparse
import json
import re
from urllib.request import urlopen
import numpy as np
import pandas as pd
from scipy.io import arff
from . import lymphography_datasets as lymphography
from . import pima_datasets as pima

OPENML_ID_TO_DATASET_NAME: dict[str, str] = {
    "10": "lymphography",
    "999": "audiology",
    "1464": "blood-transfusion-service",
    "1480": "ilpd_patient_data",
    "1510": "wisconsin-breast-cancer",
}
OPENML_ID_TO_RELATION: dict[str, str] = {
    "1480": "Indian Liver Patient Dataset",
    "1510": "Breast Cancer Wisconsin (Diagnostic), WDBC",
}
WDBC_URL = "https://www.openml.org/data/v1/download/1592318/wdbc.arff"
WDBC_DESCRIPTION = "Breast Cancer Wisconsin (Diagnostic), WDBC (OpenML ID 1510, version 1), also distributed by sklearn.datasets.load_breast_cancer. 569 samples with 30 continuous cell-nucleus measurements extracted from digitized fine-needle aspirate images; no missing values. Binary diagnosis: 0 = benign (357), 1 = malignant (212)."
AUDIOLOGY_TARGET_MAPPING = {"0": "other source diagnoses (N)", "1": "cochlear_age (P)"}
AUDIOLOGY_DESCRIPTION = "OpenML Audiology ID 999, version 2: binary classification of the original cochlear_age diagnosis (P, encoded 1) versus all other source diagnoses (N, encoded 0). The P/N mapping is verified against the row-aligned original multiclass release, OpenML ID 7. The negative class includes other hearing disorders and normal ears. Exact duplicate feature rows with agreeing labels are collapsed before splitting; missing categorical measurements are preserved. Evaluation uses five stratified outer folds and three-fold inner cross-validation for hyperparameter selection."


def deduplicate_audiology(X_df, y):
    pairs = X_df.assign(__audiology_target=np.asarray(y))
    if len(pairs.drop_duplicates()) != len(X_df.drop_duplicates()):
        raise ValueError("Audiology has identical features with conflicting labels")
    keep = ~X_df.duplicated(keep="first")
    return (X_df.loc[keep].reset_index(drop=True), np.asarray(y)[keep.to_numpy()])


def deduplicate_dropping_conflicts(X_df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Collapse exact duplicate rows and drop feature patterns with conflicting labels."""
    feat_cols = list(X_df.columns)
    combined = X_df.copy()
    combined["__target__"] = np.asarray(y)
    dedup = combined.drop_duplicates(subset=[*feat_cols, "__target__"], keep="first")
    conflict_mask = dedup.duplicated(subset=feat_cols, keep=False)
    cleaned = dedup.loc[~conflict_mask].reset_index(drop=True)
    y_clean = cleaned["__target__"].to_numpy(dtype=np.asarray(y).dtype)
    return cleaned.drop(columns=["__target__"]), y_clean


def verify_audiology_mapping(binary_path, multiclass_path):
    binary, _ = load_arff_to_dataframe(Path(binary_path))
    original, _ = load_arff_to_dataframe(Path(multiclass_path))
    binary = replace_question_mark_with_nan(convert_objects_to_strings(binary))
    original = replace_question_mark_with_nan(convert_objects_to_strings(original))
    pd.testing.assert_frame_equal(binary.iloc[:, :-1], original.iloc[:, :-1])
    if not np.array_equal(
        binary["binaryClass"].eq("P"), original["class"].eq("cochlear_age")
    ):
        raise ValueError(
            "Audiology P does not match the original cochlear_age diagnosis"
        )
    return {
        "raw_rows": len(binary),
        "positive_rows": int(binary["binaryClass"].eq("P").sum()),
        "mapping": AUDIOLOGY_TARGET_MAPPING,
    }


def download_wdbc(input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / "dataset_1510_wdbc.arff"
    with urlopen(WDBC_URL, timeout=60) as response:
        payload = response.read()
    from io import StringIO

    data, _ = arff.loadarff(StringIO(payload.decode("utf-8")))
    if len(data) != 569 or len(data.dtype.names) != 31:
        raise ValueError("Expected the 569-row, 30-feature WDBC release")
    path.write_bytes(payload)
    return path


def get_relation_name(arff_path: Path) -> str:
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith("@relation"):
                name = line.split(maxsplit=1)[-1].strip().strip("'\"")
                return name.replace(" ", "_")
    stem = arff_path.stem
    return stem.replace(" ", "_")


def get_openml_id_from_filename(filename: str) -> str:
    m = re.search("dataset_(\\d+)", filename, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search("(\\d+)", filename)
    if m:
        return m.group(1)
    return "unknown"


def load_arff_to_dataframe(arff_path: Path) -> tuple[pd.DataFrame, object]:
    data, meta = arff.loadarff(str(arff_path))
    df = pd.DataFrame(data)
    return (df, meta)


def convert_objects_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.kind == "O":
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8")
                if isinstance(x, bytes)
                else str(x)
                if pd.notna(x)
                else x
            )
    return df


def replace_question_mark_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "string":
            df[col] = df[col].replace("?", np.nan)
        elif df[col].dtype.kind in ("f", "i", "u"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_target_column(df: pd.DataFrame, meta, target_col: str | None) -> str:
    if target_col and target_col in df.columns:
        return target_col
    if meta is not None and hasattr(meta, "_attr_names"):
        return meta._attr_names[-1]
    return df.columns[-1]


def encode_target(y_series: pd.Series) -> tuple[np.ndarray, str]:
    from sklearn.preprocessing import LabelEncoder

    valid = y_series.notna()
    if not valid.all():
        y_series = y_series.loc[valid]
    y_flat = y_series.astype(str)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_flat)
    n_classes = len(le.classes_)
    if n_classes == 2:
        task = "binary"
    else:
        task = "classification"
    return (y_enc.astype(np.int64), task, valid)


def infer_cat_idx(X_df: pd.DataFrame) -> list[int]:
    cat_idx = []
    for i, col in enumerate(X_df.columns):
        if X_df[col].dtype == object or X_df[col].dtype.name == "string":
            cat_idx.append(i)
        elif X_df[col].dtype.name == "category":
            cat_idx.append(i)
    return cat_idx


def process_one_arff(
    arff_path: Path,
    output_base: Path,
    target_col: str | None = None,
    force: bool = False,
    *,
    errors: list[str] | None = None,
) -> bool:
    arff_path = Path(arff_path)
    if not arff_path.suffix.lower() == ".arff":
        return False
    relation_name = get_relation_name(arff_path)
    openml_id = get_openml_id_from_filename(arff_path.name)
    if openml_id == "7":
        print(
            f"  Skip {arff_path.name}: OpenML ID 7 is the multiclass reference for audiology (ID 999)"
        )
        return False
    if openml_id == "15" or relation_name.lower() in {
        "breast-w",
        "wisconsin-breast-cancer",
    }:
        raise ValueError(
            "Wisconsin uses Diagnostic / OpenML 1510. Remove the Original OpenML-15 ARFF and use --download-wdbc."
        )
    if openml_id in OPENML_ID_TO_DATASET_NAME:
        dataset_name = OPENML_ID_TO_DATASET_NAME[openml_id]
    else:
        dataset_name = relation_name or f"OpenML_{openml_id}"
        dataset_name = (
            re.sub("[^\\w\\-]", "_", dataset_name).strip("_") or f"OpenML_{openml_id}"
        )
    if dataset_name not in dataset_names():
        print(f"  Skip {dataset_name}: outside the benchmark")
        return False
    output_dir = output_base / dataset_name
    if output_dir.exists() and (not force):
        print(f"  Skip (exists): {dataset_name}")
        return True
    print(f"  Processing: {arff_path.name} -> {dataset_name}")
    try:
        df, meta = load_arff_to_dataframe(arff_path)
    except Exception as e:
        print(f"  Error loading {arff_path}: {e}")
        if errors is not None:
            errors.append(arff_path.name)
        return False
    df = convert_objects_to_strings(df)
    df = replace_question_mark_with_nan(df)
    all_null_cols = df.columns[df.isna().all()].tolist()
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
        print(f"    Dropped all-null columns: {all_null_cols}")
    target_name = get_target_column(df, meta, target_col)
    if target_name not in df.columns:
        print(f"  Error: target column '{target_name}' not in {df.columns.tolist()}")
        if errors is not None:
            errors.append(arff_path.name)
        return False
    y_series = df[target_name]
    X_df = df.drop(columns=[target_name])
    if openml_id == "10":
        if target_name != "class":
            raise ValueError("Lymphography requires the source diagnosis class target")
        X_df, y_series, source_rows = lymphography.select_cohort(
            X_df, y_series, arff_path
        )
    y_enc, task, valid = encode_target(y_series)
    X_df = X_df.loc[valid].reset_index(drop=True)
    if openml_id == "999":
        if target_name != "binaryClass" or set(y_series.loc[valid].astype(str)) != {
            "N",
            "P",
        }:
            raise ValueError(
                "Audiology ID 999 requires its source binaryClass N/P target"
            )
        for candidate in arff_path.parent.glob("*.arff"):
            if candidate != arff_path and get_openml_id_from_filename(candidate.name) == "7":
                verify_audiology_mapping(arff_path, candidate)
                break
        X_df, y_enc = deduplicate_audiology(X_df, y_enc)
    if openml_id == "1464":
        for col in list(X_df.columns):
            if col.lower() in ("monetary", "v3"):
                freq_cols = [
                    c for c in X_df.columns if c.lower() in ("frequency", "v2")
                ]
                if freq_cols and np.allclose(
                    pd.to_numeric(X_df[col], errors="coerce"),
                    250.0 * pd.to_numeric(X_df[freq_cols[0]], errors="coerce"),
                    equal_nan=True,
                ):
                    X_df = X_df.drop(columns=[col])
                    break
        X_df, y_enc = deduplicate_dropping_conflicts(X_df, y_enc)
    X_arr = X_df.to_numpy(dtype=object, copy=True)
    for i in range(X_arr.shape[1]):
        col = X_df.iloc[:, i]
        if col.dtype.kind in ("i", "u", "f"):
            X_arr[:, i] = pd.to_numeric(col, errors="coerce").values
    cat_idx = infer_cat_idx(X_df)
    if openml_id == "37":
        if X_df.columns.tolist() != pima.FEATURES:
            raise ValueError("Unexpected Pima feature order")
        X_arr = pima.repair_values(X_arr)
    if openml_id == "1510":
        from sklearn.datasets import load_breast_cancer

        X_arr = X_df.to_numpy(dtype=np.float64)
        if X_arr.shape != (569, 30) or cat_idx or (not np.isfinite(X_arr).all()):
            raise ValueError(
                "WDBC must have 569 rows and 30 complete continuous features"
            )
        if set(y_series.astype(str)) != {"1", "2"}:
            raise ValueError("Expected OpenML WDBC labels 1=benign, 2=malignant")
        canonical = load_breast_cancer()
        if not np.allclose(
            X_arr, canonical.data, rtol=0, atol=5e-07
        ) or not np.array_equal(y_enc, 1 - canonical.target):
            raise ValueError(
                "OpenML WDBC does not match the sklearn Diagnostic dataset"
            )
        X_arr = canonical.data.copy()
        X_df.columns = canonical.feature_names
    num_classes = (
        int(np.max(y_enc)) + 1
        if task == "classification"
        else 2
        if task == "binary"
        else 1
    )
    metadata = {
        "name": dataset_name,
        "task": task,
        "cat_idx": cat_idx,
        "num_samples": int(X_arr.shape[0]),
        "num_features": int(X_arr.shape[1]),
        "num_classes": num_classes if task != "regression" else None,
        "source": f"OpenML_Dataset_ID_{openml_id}",
        "description": f"OpenML dataset (ID {openml_id}), relation: {OPENML_ID_TO_RELATION.get(openml_id, relation_name)}",
    }
    if metadata["num_classes"] is None:
        del metadata["num_classes"]
    if openml_id == "37":
        metadata.update(pima.metadata_fields())
    if openml_id == "1510":
        metadata.update(
            description=WDBC_DESCRIPTION,
            url="https://www.openml.org/d/1510",
            label="clinical & healthcare",
            target_mapping={"0": "benign", "1": "malignant"},
        )
    if openml_id == "999":
        metadata.update(
            description=AUDIOLOGY_DESCRIPTION,
            url="https://www.openml.org/d/999",
            label="clinical & healthcare",
            target_mapping=AUDIOLOGY_TARGET_MAPPING,
            split_strategy="cv",
        )
    if openml_id == "10":
        lymphography.verify_values(X_arr, y_enc)
        metadata.update(lymphography.task_metadata())
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X_arr, allow_pickle=True)
    np.save(output_dir / "y.npy", y_enc)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(X_df.columns.tolist(), f, indent=2)
    if openml_id == "10":
        pd.DataFrame(
            {
                "row_idx": range(len(y_enc)),
                "source_row_idx": source_rows,
                "source_diagnosis": y_series,
            }
        ).to_csv(output_dir / "sample_metadata.csv", index=False)
    if openml_id in {"10", "1510", "999"}:
        (output_dir / "split_indices.npy").unlink(missing_ok=True)
    print(
        f"    Saved to {output_dir} (task={task}, n={metadata['num_samples']}, cat_idx={len(cat_idx)})"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unify OpenML .arff datasets into TabBioMed format."
    )
    parser.add_argument(
        "--download-wdbc",
        action="store_true",
        help="Download Wisconsin Diagnostic (OpenML 1510) into --input before ingestion.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/OpenML_datasets",
        help="Input directory containing .arff files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/unified_format",
        help="Output base directory for unified datasets",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column name (default: last column from .arff)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing unified datasets"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Process only .arff files whose name contains this string (e.g. sick). Omit to process all.",
    )
    args = parser.parse_args()
    input_dir = Path(args.input)
    output_base = Path(args.output)
    if args.download_wdbc:
        download_wdbc(input_dir)
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)
    arff_files = sorted(
        (
            p
            for p in input_dir.glob("*.arff")
            if get_openml_id_from_filename(p.name) != "15"
        )
    )
    if not arff_files:
        print(f"No .arff files in {input_dir}")
        raise SystemExit(1)
    if args.dataset:
        name_lower = args.dataset.lower()
        arff_files = [
            p
            for p in arff_files
            if name_lower in p.stem.lower()
            or name_lower
            == OPENML_ID_TO_DATASET_NAME.get(get_openml_id_from_filename(p.name))
        ]
        if not arff_files:
            print(f"No .arff files matching '{args.dataset}' in {input_dir}")
            raise SystemExit(1)
        print(f"Filtered to {len(arff_files)} .arff file(s) matching '{args.dataset}'")
    else:
        print(f"Found {len(arff_files)} .arff files in {input_dir}")
    success = 0
    errors: list[str] = []
    for path in arff_files:
        if process_one_arff(
            path,
            output_base,
            target_col=args.target_col,
            force=args.force,
            errors=errors,
        ):
            success += 1
    print(f"Done: {success}/{len(arff_files)} datasets written to {output_base}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
