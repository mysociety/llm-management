"""
SAR classification results and conservative failure contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SARResult(BaseModel):
    is_sar: bool
    stage: Literal["logistic", "deberta"]
    status: Literal["complete", "unclear"]
    reason: (
        Literal[
            "logistic_unavailable",
            "deberta_unavailable",
            "deberta_busy",
            "deberta_input_unsupported",
            "invalid_score",
            "inference_failed",
        ]
        | None
    ) = None
    logistic_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    deberta_score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
