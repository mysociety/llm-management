"""AutoDetector adapted from fast-auto-classifier's supplied inference.py.

Use service-owned shared models, its authenticated CPU backend, and the existing
Unicode compatibility adaptation. Results describe classification only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Protocol

from ..inference.batching import AsyncClassifierBatcher
from ..inference.errors import (
    ClassifierBusy,
    ClassifierInputTooLong,
    ClassifierOutputError,
    ClassifierUnavailable,
)
from ..inference.logistic_v1 import BinaryPredictionV1, normalize_text_v1
from ..inference.validation import binary_positive_probability, validate_probability
from ..settings import settings
from .schemas import AutomaticResult, unresolved

if TYPE_CHECKING:
    from .resources import LogisticDetector


class Encoder(Protocol):
    """
    Describe the probability-row interface required by encoder confirmation.
    """

    def classify(self, texts: list[str]) -> list[list[float]]:
        """
        Return one ordered authored/automatic probability row per input text.
        """
        ...


class AutoDetector:
    """
    Combine a typed logistic predictor with an optional encoder controller.
    """

    def __init__(
        self,
        logistic: Callable[[str], BinaryPredictionV1],
        encoder: Encoder | None = None,
        mode: Literal["cascade", "logistic"] = "cascade",
    ) -> None:
        """
        Bind a logistic predictor and optional encoder for the selected policy.

        Cascade mode requires an encoder; logistic mode can run without one. Construction
        does not invoke either stage or load model weights.
        """
        if mode not in ("cascade", "logistic"):
            raise ValueError("Unsupported automatic detector mode")
        if mode == "cascade" and encoder is None:
            raise ValueError("Cascade mode requires an encoder")
        self.logistic = logistic
        self.encoder = encoder
        self.mode = mode

    def classify(self, body: str) -> AutomaticResult:
        """
        Classify one body, returning an unresolved result when a decision cannot finish.

        Empty normalized text is unresolved. Negative logistic decisions and logistic mode
        return immediately; positive cascade decisions require encoder confirmation at
        probability 0.5. Validate scores and record the stage and reason of failures.
        An unresolved result's False flag must not be read as a successful authored decision.
        """
        if not normalize_text_v1(body):
            return unresolved("none", "empty_body")
        try:
            cpu = self.logistic(body)
            validate_probability(cpu.positive_score)
        except Exception:
            return unresolved("logistic", "logistic_failure")
        if not cpu.result or self.mode == "logistic":
            return AutomaticResult(
                is_automatic=cpu.result,
                stage="logistic",
                logistic_score=cpu.positive_score,
            )
        try:
            assert self.encoder is not None
            rows = self.encoder.classify([normalize_text_v1(body)])
            probability = binary_positive_probability(rows)
            return AutomaticResult(
                is_automatic=probability >= 0.5,
                stage="deberta",
                logistic_score=cpu.positive_score,
                encoder_score=probability,
            )
        except ClassifierBusy:
            reason = "encoder_busy"
        except ClassifierUnavailable:
            reason = "encoder_unavailable"
        except ClassifierOutputError:
            reason = "invalid_score"
        except ClassifierInputTooLong:
            reason = "overlong_body"
        except Exception:
            reason = "encoder_failure"
        return unresolved("deberta", reason, cpu.positive_score)


async def detect_automatic(body: str, fast: bool = False) -> AutomaticResult:
    """
    Classify through the process-local default assembly for direct callers.
    """
    from ..bootstrap import default_application

    return await default_application().automatic.detect(body, fast)


class AutomaticClassifier:
    """
    Application-owned admission capacity and cold model providers.
    """

    def __init__(
        self,
        logistic_provider: Callable[[], LogisticDetector],
        encoder_provider: Callable[[], Encoder],
        *,
        max_concurrency: int,
    ) -> None:
        """
        Bind cold providers and create this application's bounded admission capacity.
        """
        self.logistic_provider = logistic_provider
        self.encoder_provider = encoder_provider
        self.capacity = asyncio.Semaphore(max_concurrency)
        self.batcher = AsyncClassifierBatcher(
            lambda texts: self.encoder_provider().classify(texts),
            batch_size=settings.deberta_batch_size,
            delay_seconds=settings.deberta_batch_delay_ms / 1000,
            max_pending=settings.deberta_queue_size,
            timeout_seconds=settings.deberta_queue_timeout_seconds,
        )
        self._gates: set[asyncio.Task[AutomaticResult]] = set()

    def detector(self, fast: bool) -> AutoDetector:
        """
        Build a per-call policy adapter without downloading model weights.

        Fast mode selects the standalone logistic policy and never resolves the encoder
        provider. Normal mode selects the cascade gate and obtains the cold encoder owner.
        """
        return AutoDetector(
            lambda body: self.logistic_provider().predict(body, fast=fast),
            self.encoder_provider() if not fast else None,
            mode="logistic" if fast else "cascade",
        )

    def classify(self, body: str, fast: bool) -> AutomaticResult:
        """
        Run synchronous classification and represent setup failures as unresolved.
        """
        try:
            return self.detector(fast).classify(body)
        except Exception:
            return unresolved("none", "inference_failed")

    def classify_gate(self, body: str, fast: bool) -> AutomaticResult:
        """Apply the selected logistic policy without resolving the encoder."""
        return AutoDetector(
            lambda text: self.logistic_provider().predict(text, fast=fast),
            mode="logistic",
        ).classify(body)

    async def _gate(self, body: str, fast: bool) -> AutomaticResult:
        async with self.capacity:
            return await asyncio.to_thread(self.classify_gate, body, fast)

    async def close(self) -> None:
        await self.batcher.close()
        if self._gates:
            await asyncio.gather(*self._gates, return_exceptions=True)

    async def detect(self, body: str, fast: bool = False) -> AutomaticResult:
        """Await logistic capacity, then batch only positive cascade decisions."""
        task = asyncio.create_task(self._gate(body, fast))
        self._gates.add(task)
        task.add_done_callback(self._gates.discard)
        try:
            result = await asyncio.shield(task)
        except Exception:
            return unresolved("none", "inference_failed")
        if fast or result.unresolved or not result.is_automatic:
            return result
        try:
            row = await self.batcher.predict(normalize_text_v1(body))
            probability = binary_positive_probability([row])
            return AutomaticResult(
                is_automatic=probability >= 0.5,
                stage="deberta",
                logistic_score=result.logistic_score,
                encoder_score=probability,
            )
        except ClassifierBusy:
            reason = "encoder_busy"
        except ClassifierUnavailable:
            reason = "encoder_unavailable"
        except ClassifierOutputError:
            reason = "invalid_score"
        except ClassifierInputTooLong:
            reason = "overlong_body"
        except Exception:
            reason = "encoder_failure"
        return unresolved("deberta", reason, result.logistic_score)
