"""Local personal-records moderation independent of HTTP.

Only successful negative classification clears a SAR flag. Failed inference
retains the flag for moderator review.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..inference.errors import (
    ClassifierBusy,
    ClassifierOutputError,
    ClassifierUnavailable,
)
from ..inference.logistic_v1 import normalize_text_v1
from ..inference.validation import binary_positive_probability, validate_probability
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

    async def detect(self, raw_text: str, fast: bool = False) -> SARResult:
        """
        Run the moderation cascade in a worker thread using this application's owners.

        fast=True skips encoder confirmation. Cancelling the async caller does not stop
        the running thread; model owners retain their resource leases until work ends.
        Inference failures are represented by an unclear result that retains the flag.
        """
        return await asyncio.to_thread(
            _detect_sar,
            raw_text,
            fast,
            logistic_provider=self.logistic_provider,
            encoder_provider=self.encoder_provider,
        )
