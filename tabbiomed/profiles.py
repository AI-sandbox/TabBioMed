"""Benchmark experiment profiles."""

from copy import deepcopy
from pathlib import Path

PROFILES = ("main", "pca500")
PCA_MODELS = ("TabPFNv2.5", "RealTabPFNv2.5", "TabICLv2")
PCA_MIN_FEATURES = 2000
PCA_SEARCH_SPACES = (
    Path(__file__).resolve().parent / "configs/model_search_space_pca500"
)


def validate_profile(profile, model, metadata):
    if profile not in PROFILES:
        raise ValueError(f"Unknown experiment profile: {profile}")
    if profile == "pca500":
        if model not in PCA_MODELS:
            raise ValueError(f"PCA-500 supports only {', '.join(PCA_MODELS)}")
        if metadata["num_features"] <= PCA_MIN_FEATURES:
            raise ValueError("PCA-500 requires a dataset with more than 2,000 features")


def profile_config(config, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown experiment profile: {profile}")
    config = deepcopy(config)
    if profile == "pca500":
        config["search_space_dir"] = str(PCA_SEARCH_SPACES)
    return config


def default_output(profile):
    return Path("results/raw_pca500" if profile == "pca500" else "results/raw")


def profile_scope(profile):
    return "pca_500" if profile == "pca500" else "main"
