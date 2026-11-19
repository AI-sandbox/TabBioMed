"""Reference raw-source curation, separate from official benchmark downloads."""

from importlib import import_module
from pathlib import Path
import os
import runpy
import sys

SOURCES = {
    "avida": "cognano_datasets",
    "cptac": "cptac_datasets",
    "diabetes": "diabetes_datasets",
    "gastrulation": "gastrulation_datasets",
    "genomics": "genomics",
    "heart-disease": "heartdisease_uci_datasets",
    "hnoca": "hnoca_datasets",
    "iaccd": "iaccd_datasets",
    "metabric": "metabric_datasets",
    "lgg-methylation": "lgg_methylation",
    "metabolomics": "metabolomics_datasets",
    "moleculenet": "moleculenet_datasets",
    "multiomics": "multiomics_datasets",
    "music": "music_datasets",
    "openml": "openml_datasets",
    "parkinson-speech": "parkinson_speech_measures",
    "patientcare": "patientcare_datasets",
    "pmlb": "pmlb_datasets",
    "proteinea": "proteinea_datasets",
    "rna": "rna",
    "scikit-feature": "scikit_feature",
    "singlecell": "singlecell_datasets",
    "tahoe": "tahoe_datasets",
    "talent": "talent_datasets",
    "true-protein": "true_protein_tasks_datasets",
    "vitaldb": "vitaldb_datasets",
    "vocs": "tabpfn_vocs_datasets",
}


FIXED_LAYOUT_SOURCES = (
    "patientcare",
    "rna",
    "vitaldb",
    "heart-disease",
    "proteinea",
    "iaccd",
    "scikit-feature",
    "tahoe",
    "genomics",
)


def curate(source, arguments=(), workspace="."):
    if source not in SOURCES:
        raise ValueError(f"Unknown data source: {source}")
    if source in FIXED_LAYOUT_SOURCES:
        import argparse

        argparse.ArgumentParser(
            description=f"Curate {source} benchmark data from data/raw in the workspace"
        ).parse_args(arguments)
    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    previous, previous_argv = Path.cwd(), sys.argv
    try:
        os.chdir(workspace)
        module = "tabbiomed.data.curation." + SOURCES[source]
        sys.argv = [module, *arguments]
        if source == "singlecell":
            import argparse

            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--input", type=Path, default=Path("data/raw/scUnified")
            )
            parser.add_argument(
                "--output", type=Path, default=Path("data/unified_format")
            )
            implementation = import_module(module)
            parser.add_argument(
                "--dataset",
                choices=implementation.NAMES,
                nargs="+",
                default=implementation.NAMES,
            )
            args = parser.parse_args(arguments)
            for name in args.dataset:
                implementation.ingest(args.input, args.output, name)
        else:
            runpy.run_module(module, run_name="__main__")
    finally:
        os.chdir(previous)
        sys.argv = previous_argv
