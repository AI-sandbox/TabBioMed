from __future__ import annotations
import math

QUADRANT_LABELS = [
    "Q1 (Low samples, Low features)",
    "Q2 (Low samples, High features)",
    "Q3 (High samples, Low features)",
    "Q4 (High samples, High features)",
]


def assign_quadrant(
    num_samples: float, num_features: float, med_log_s: float, med_log_f: float
) -> str:
    log_s = math.log10(max(1.0, float(num_samples)))
    log_f = math.log10(max(1.0, float(num_features)))
    if log_s < med_log_s and log_f < med_log_f:
        return QUADRANT_LABELS[0]
    if log_s < med_log_s and log_f >= med_log_f:
        return QUADRANT_LABELS[1]
    if log_s >= med_log_s and log_f < med_log_f:
        return QUADRANT_LABELS[2]
    return QUADRANT_LABELS[3]
