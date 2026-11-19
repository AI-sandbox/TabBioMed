"""Shared wall-clock eligibility checks."""

import math


def completed_within_deadline(completed_seconds, deadline_seconds):
    return (
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in (completed_seconds, deadline_seconds)
        )
        and completed_seconds <= deadline_seconds
    )
