KNOWN_OVERLAPS = {
    "iLTM": (
        "ALLAML",
        "Cardiovascular-Disease-dataset",
        "Diabetes_UCI",
        "GLI_85",
        "HeartDisease_UCI",
        "ilpd_patient_data",
        "pima_diabetes",
        "sick",
        "spectf",
        "thyroid",
        "wisconsin-breast-cancer",
    ),
    "HyperFast": (
        "QSAR_biodegradation",
        "bioresponse",
        "blood-transfusion-service",
        "dna",
        "ilpd_patient_data",
        "sick",
    ),
    "TabDPT": ("Diabetes_UCI",),
    "RealTabPFNv2.5": ("Cardiovascular-Disease-dataset",),
}


def pretraining_overlaps():
    groups = {model: list(datasets) for model, datasets in KNOWN_OVERLAPS.items()}
    groups["all"] = sorted(set().union(*KNOWN_OVERLAPS.values()))
    return groups
