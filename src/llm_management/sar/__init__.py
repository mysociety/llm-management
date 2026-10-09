"""Local personal-records moderation, independent of FOI extraction and HTTP.

logistic_v1.py is adapted from mySociety/logistic-sar-detector-v1
at ade6333a06995cbde835648d758d8b092900bc52 (MIT). Model artifacts remain
in the Hugging Face cache, under their upstream licenses. The runtime Unicode
version check is intentionally relaxed; preparation uses the host Python tables.
"""

import asyncio
from functools import lru_cache
import logging
import math
from typing import Literal

from pydantic import BaseModel, Field

from ..errors import ClassifierBusy, ClassifierOutputError, ClassifierUnavailable
from ..inference import LocalSequenceClassifier
from ..local_resources import LocalResource, local_resources
from ..settings import settings
from ..deployments import get_catalog, LogisticDeployment, SARClassifierDeployment
from .logistic_v1 import load_model_v1, normalize_text_v1, predict_binary

LOGISTIC_THRESHOLD = 0.0018603434604847564
# Expected training metadata, not a runtime Unicode-version requirement.
UNICODE_VERSION = "16.0.0"
logger = logging.getLogger(__name__)


class SARResult(BaseModel):
    is_sar: bool
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


class LogisticDetector:
    def __init__(self, config: LogisticDeployment | None = None):
        catalog = get_catalog()
        self.config = config or catalog.local.get(
            catalog.require_sar().logistic, LogisticDeployment
        )
        self.checkpoint = catalog.model[self.config.model_ref]
        self._model = None
        self.resource = local_resources.register(
            LocalResource(self.config.slug, self.load, self.unload)
        )

    def load(self):
        with self.resource.use():
            if self._model is None:
                from huggingface_hub import hf_hub_download
                from pathlib import Path

                path = hf_hub_download(
                    self.checkpoint.repo,
                    self.config.artifact,
                    revision=self.checkpoint.revision,
                    token=settings.huggingface_token or None,
                    cache_dir=settings.classifier_cache_dir,
                )
                model = load_model_v1(Path(path))
                if (
                    model.true_label != "sar"
                    or model.false_label != "not-sar"
                    or model.probability_threshold != LOGISTIC_THRESHOLD
                    or not model.inclusive
                    or model.logistic.features.unicode_version != UNICODE_VERSION
                ):
                    raise ClassifierUnavailable(
                        "Unexpected SAR logistic artifact policy"
                    )
                self._model = model
            self.resource.mark_ready()
            return self._model

    def unload(self):
        self._model = None

    def predict(self, raw_text: str):
        with self.resource.use():
            return predict_binary(raw_text, self.load())


class SARSequenceClassifier(LocalSequenceClassifier):
    def _tokenize(self, texts):
        with self._load_lock:
            if self._tokenizer is None:
                try:
                    from huggingface_hub import snapshot_download
                    from transformers import AutoTokenizer

                    # Like the Granite tokenizer, fetch with explicit authentication
                    # then load locally to avoid unauthenticated v4 metadata probes.
                    snapshot = snapshot_download(
                        self.model_name,
                        revision=self.revision,
                        token=self.token,
                        cache_dir=self.cache_dir,
                        allow_patterns=[
                            "config.json",
                            "tokenizer*",
                            "special_tokens_map.json",
                        ],
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        snapshot,
                        local_files_only=True,
                        fix_mistral_regex=False,
                        # v5 serialized a list; v4 expects a mapping. PAD/CLS/SEP
                        # already have their standard named roles in this artifact.
                        extra_special_tokens={},
                    )
                except Exception as exc:
                    raise ClassifierUnavailable("Could not load SAR tokenizer") from exc
        return super()._tokenize(texts)


@lru_cache(maxsize=None)
def _logistic_detector(slug: str):
    return LogisticDetector(get_catalog().local.get(slug, LogisticDeployment))


def logistic_detector(config: LogisticDeployment | None = None):
    return _logistic_detector(
        config.slug if config else get_catalog().require_sar().logistic
    )


@lru_cache(maxsize=None)
def _deberta_detector(slug: str):
    catalog = get_catalog()
    config = catalog.local.get(slug, SARClassifierDeployment)
    checkpoint = catalog.model[config.model_ref]
    return SARSequenceClassifier(
        model=checkpoint.repo,
        revision=checkpoint.revision,
        labels=("not-sar", "sar"),
        max_tokens=config.max_tokens,
        max_units=1,
        batch_size=1,
        threads=settings.cpu_inference_threads,
        token=settings.huggingface_token or None,
        cache_dir=settings.classifier_cache_dir,
        model_kwargs={"dtype": "float32"},
        resource_name=config.slug,
    )


def deberta_detector(config: SARClassifierDeployment | None = None):
    return _deberta_detector(
        config.slug if config else get_catalog().require_sar().classifier
    )


def _detect_sar(raw_text: str) -> SARResult:
    score = None
    stage = "logistic"
    try:
        prediction = logistic_detector().predict(raw_text)
        score = prediction.positive_score
        if not math.isfinite(score) or not 0 <= score <= 1:
            score = None
            raise ClassifierOutputError("Invalid logistic score")
        if not prediction.result:
            return SARResult(is_sar=False, status="complete", logistic_score=score)
        stage = "deberta"
        rows = deberta_detector().classify([normalize_text_v1(raw_text)])
        if (
            len(rows) != 1
            or len(rows[0]) != 2
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in rows[0])
            or not math.isclose(sum(rows[0]), 1.0, abs_tol=1e-5)
        ):
            raise ClassifierOutputError("Invalid DeBERTa scores")
        positive = rows[0][1]
        return SARResult(
            is_sar=positive >= 0.5,
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
    return SARResult(is_sar=True, status="unclear", reason=reason, logistic_score=score)


async def detect_sar(raw_text: str) -> SARResult:
    """Run blocking work off the event loop; worker leases survive cancellation."""
    return await asyncio.to_thread(_detect_sar, raw_text)
