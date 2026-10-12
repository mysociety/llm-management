"""
Automatic-response classification results.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AutomaticResult(BaseModel):
    is_automatic: bool
    stage: Literal["logistic", "deberta", "none"]
    unresolved: bool = False
    reason: str | None = None
    logistic_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    encoder_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


def unresolved(
    stage: Literal["logistic", "deberta", "none"],
    reason: str,
    logistic_score: float | None = None,
) -> AutomaticResult:
    return AutomaticResult(
        is_automatic=False,
        stage=stage,
        unresolved=True,
        reason=reason,
        logistic_score=logistic_score,
    )
