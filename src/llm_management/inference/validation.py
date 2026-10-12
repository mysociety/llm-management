"""
Probability validation shared by binary correspondence classifiers.
"""

from __future__ import annotations

import math

from .errors import ClassifierOutputError


def validate_probability(score: float) -> None:
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ClassifierOutputError("Invalid classifier score")


def binary_positive_probability(rows: list[list[float]]) -> float:
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ClassifierOutputError("Invalid binary classifier dimensions")
    for score in rows[0]:
        validate_probability(score)
    if not math.isclose(sum(rows[0]), 1.0, abs_tol=1e-5):
        raise ClassifierOutputError("Invalid binary probability sum")
    return rows[0][1]
