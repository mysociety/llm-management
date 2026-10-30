"""
Validated operational limits captured when a classifier runtime is assembled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ..settings import Settings


class ClassifierExecutionPolicy(BaseModel):
    """
    Immutable admission and batching limits shared by SAR and Automatic execution.

    This describes scheduling only. Feature decision rules, input preparation and
    failure results remain in their respective pipelines. Existing runtimes retain
    these values even if process settings change after assembly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(ge=1, le=64)
    batch_delay_seconds: float = Field(ge=0, le=1)
    batch_max_pending: int = Field(ge=1, le=4096)
    batch_timeout_seconds: float = Field(gt=0)
    logistic_max_pending: int = Field(ge=1)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """
        Capture explicitly supplied deployment settings at the assembly boundary.
        """
        return cls(
            batch_size=settings.deberta_batch_size,
            batch_delay_seconds=settings.deberta_batch_delay_ms / 1000,
            batch_max_pending=settings.deberta_queue_size,
            batch_timeout_seconds=settings.deberta_queue_timeout_seconds,
            logistic_max_pending=settings.logistic_max_pending,
        )
