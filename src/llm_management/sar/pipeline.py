"""Local personal-records moderation independent of HTTP.

Only successful negative classification clears a SAR flag. Failed inference
retains the flag for moderator review.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..inference.batching import AsyncClassifierBatcher
from ..inference.errors import (
    ClassifierBusy,
    ClassifierOutputError,
    ClassifierUnavailable,
)
from ..inference.logistic_v1 import normalize_text_v1
from ..inference.tasks import ClassifierTasks
from ..inference.validation import binary_positive_probability, validate_probability
from ..settings import settings
from .resources import deberta_detector, logistic_detector
from .schemas import SARResult

if TYPE_CHECKING:
    from ..inference.classifiers import AuthenticatedSequenceClassifier
    from .resources import LogisticDetector


logger = logging.getLogger(__name__)


def _detect_sar(
    raw_text: str,
    fast: bool = False,
    *,
    logistic_provider: Callable[[], LogisticDetector] | None = None,
    encoder_provider: Callable[[], AuthenticatedSequenceClassifier] | None = None,
) -> SARResult:
    """
    Apply the logistic gate and optionally confirm positives with the encoder.

    Negative logistic decisions and fast mode return immediately. Otherwise normalize
    the text, run DeBERTa and compare its validated positive probability with 0.5.
    Resolve injected providers or default owners only when that stage is required.

    Only successful negative classification clears the SAR flag. Busy, unsupported,
    unavailable or invalid inference returns is_sar=True with status='unclear' and a
    reason, retaining a valid logistic score when available. Log reasons without
    correspondence or arbitrary exception messages.
    """
    score = None
    stage = "logistic"
    try:
        prediction = (
            logistic_provider if logistic_provider is not None else logistic_detector
        )().predict(raw_text)
        score = prediction.positive_score
        try:
            validate_probability(score)
        except ClassifierOutputError:
            score = None
            raise
        if not prediction.result or fast:
            return SARResult(
                is_sar=prediction.result,
                stage="logistic",
                status="complete",
                logistic_score=score,
            )
        stage = "deberta"
        rows = (
            encoder_provider if encoder_provider is not None else deberta_detector
        )().classify([normalize_text_v1(raw_text)])
        positive = binary_positive_probability(rows)
        return SARResult(
            is_sar=positive >= 0.5,
            stage="deberta",
            status="complete",
            logistic_score=score,
            deberta_score=positive,
        )
    except ClassifierBusy:
        reason = "deberta_busy"
    except ClassifierOutputError:
        reason = "invalid_score"
    except ClassifierUnavailable:
        reason = (
            "logistic_unavailable" if stage == "logistic" else "deberta_unavailable"
        )
    except ValueError:
        reason = (
            "deberta_input_unsupported"
            if stage == "deberta"
            else "logistic_unavailable"
        )
    except Exception:
        reason = "inference_failed"
    # Never log correspondence or exception messages that may contain input/credentials.
    logger.warning("SAR moderation unclear: %s", reason)
    return SARResult(
        is_sar=True, stage=stage, status="unclear", reason=reason, logistic_score=score
    )


async def detect_sar(raw_text: str, fast: bool = False) -> SARResult:
    """
    Classify through the process-local default assembly for direct callers.
    """
    from ..bootstrap import default_application

    return await default_application().sar.detect(raw_text, fast)


class SARClassifier:
    """
    Application-owned providers for the SAR moderation pipeline.
    """

    def __init__(
        self,
        logistic_provider: Callable[[], LogisticDetector],
        encoder_provider: Callable[[], AuthenticatedSequenceClassifier],
    ) -> None:
        """
        Bind cold stage providers without constructing or loading their models.
        """
        self.logistic_provider = logistic_provider
        self.encoder_provider = encoder_provider
        self.batcher = AsyncClassifierBatcher(
            lambda texts: self.encoder_provider().classify(texts),
            batch_size=settings.deberta_batch_size,
            delay_seconds=settings.deberta_batch_delay_ms / 1000,
            max_pending=settings.deberta_queue_size,
            timeout_seconds=settings.deberta_queue_timeout_seconds,
        )

        self.gate_tasks = ClassifierTasks[SARResult]("Logistic")

    async def close(self) -> None:
        """
        Stop new gates, close encoder admission, then drain accepted logistic work.

        Closing the encoder first preserves the existing shutdown policy: queued
        encoder requests fail, and late positive gates receive the feature's normal
        encoder-unavailable result. Always drain gates, even if encoder cleanup fails.
        """
        self.gate_tasks.stop_accepting()
        try:
            await self.batcher.close()
        finally:
            await self.gate_tasks.close()

    async def detect(self, raw_text: str, fast: bool = False) -> SARResult:
        """
        Run logistic gating in a thread, then await batched encoder confirmation.
        """
        pending = self.gate_tasks.submit(
            lambda: asyncio.to_thread(
                _detect_sar,
                raw_text,
                True,
                logistic_provider=self.logistic_provider,
                encoder_provider=self.encoder_provider,
            ),
            max_pending=settings.logistic_max_pending,
        )
        result = await pending
        if fast or result.status != "complete" or not result.is_sar:
            return result
        try:
            row = await self.batcher.predict(normalize_text_v1(raw_text))
            probability = binary_positive_probability([row])
            return SARResult(
                is_sar=probability >= 0.5,
                stage="deberta",
                status="complete",
                logistic_score=result.logistic_score,
                deberta_score=probability,
            )
        except ClassifierBusy:
            reason = "deberta_busy"
        except ClassifierOutputError:
            reason = "invalid_score"
        except ClassifierUnavailable:
            reason = "deberta_unavailable"
        except ValueError:
            reason = "deberta_input_unsupported"
        except Exception:
            reason = "inference_failed"
        return SARResult(
            is_sar=True,
            stage="deberta",
            status="unclear",
            reason=reason,
            logistic_score=result.logistic_score,
        )
