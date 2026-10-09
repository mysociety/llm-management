"""Public async FOI operations, usable from HTTP routes or batch callers.

extract_questions: segmentation -> backend inference -> reconstruction.
process_information_request: extraction -> bounded topic classification.
Deployment access is injected so this module never imports the HTTP server.
"""

import asyncio
import logging

import httpx

from ..errors import ClassifierOutputError, ClassifierUnavailable
from ..models import LLMManagementError
from ..settings import settings
from ..sanitization import presidio
from . import backends, question_extractor
from .backends import DeploymentAccess
from .question_slice import build_extraction_result, contextual_inputs, segment_request
from .schemas import ExtractionBackend, InformationRequestResult, QuestionSliceResult

logger = logging.getLogger(__name__)


class PipelineInputError(ValueError):
    """Input exceeds a supported model budget or is otherwise invalid."""


class PipelineUnavailable(RuntimeError):
    """A required model, tokenizer or deployment is unavailable."""


class PipelineOutputError(RuntimeError):
    """A model request failed or produced invalid/incomplete output."""


async def _extract_questions(
    request_text: str,
    *,
    deployments: DeploymentAccess,
    backend: ExtractionBackend = "cpu",
) -> tuple[QuestionSliceResult, QuestionSliceResult, str]:
    try:
        units = segment_request(request_text)
        clean_request, sanitized_units, model_inputs = await asyncio.to_thread(
            presidio.sanitize_request,
            request_text,
            [unit.text for unit in units],
            contextual_inputs(units),
        )
        clean_units = [
            unit.model_copy(update={"text": text})
            for unit, text in zip(units, sanitized_units, strict=True)
        ]
        classified = await backends.classify_question_units(
            model_inputs, backend=backend, deployments=deployments
        )
        original = build_extraction_result(
            units,
            classified.probabilities,
            backend=backend,
            model=classified.model,
            revision=classified.revision,
        )
        clean = build_extraction_result(
            clean_units,
            classified.probabilities,
            backend=backend,
            model=classified.model,
            revision=classified.revision,
        )
        return original, clean, clean_request
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable(str(exc)) from exc
    except (ClassifierOutputError, httpx.HTTPError) as exc:
        logger.warning("QuestionSlice upstream failure: %s", type(exc).__name__)
        raise PipelineOutputError(
            "Classifier returned an invalid response or is unreachable"
        ) from exc
    except LLMManagementError as exc:
        raise PipelineUnavailable("Classifier deployment could not be started") from exc
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc


async def extract_questions(
    request_text: str,
    *,
    deployments: DeploymentAccess,
    backend: ExtractionBackend = "cpu",
) -> QuestionSliceResult:
    original, _, _ = await _extract_questions(
        request_text, deployments=deployments, backend=backend
    )
    return original


async def process_information_request(
    request_text: str,
    *,
    deployments: DeploymentAccess,
    backend: ExtractionBackend = "cpu",
) -> InformationRequestResult:
    extraction, clean_extraction, clean_request = await _extract_questions(
        request_text, backend=backend, deployments=deployments
    )
    result = InformationRequestResult(
        **extraction.model_dump(),
        request_text=request_text,
        classification=None,
        classification_model=None,
    )
    if not extraction.questions:
        return result
    try:
        payload = await asyncio.to_thread(
            question_extractor.prepare_topic_request,
            clean_request,
            clean_extraction.questions,
        )
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc
    except Exception as exc:
        logger.warning("Question topic tokenizer unavailable: %s", type(exc).__name__)
        raise PipelineUnavailable("Question topic tokenizer unavailable") from exc
    slug = settings.foi_topic_deployment
    cfg = deployments.get_config(slug)
    if cfg.model != settings.foi_topic_model:
        raise PipelineUnavailable(
            "FOI classification requires the configured fine-tuned topic checkpoint"
        )
    try:
        cfg, state = await deployments.ensure_running(slug)
        result.classification = await question_extractor.classify_topics(
            payload=payload,
            model=cfg.model,
            deployment_url=state.deployment_url,
            api_key=state.api_key,
            question_ids=[q.question_id for q in extraction.questions],
        )
        result.classification_model = cfg.model
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable(str(exc)) from exc
    except LLMManagementError as exc:
        raise PipelineUnavailable(
            "Question topic deployment could not be started"
        ) from exc
    except question_extractor.TopicOutputError as exc:
        raise PipelineOutputError(str(exc)) from exc
    except httpx.HTTPError as exc:
        logger.warning("Topic classification failure: %s", type(exc).__name__)
        raise PipelineOutputError(
            "Question topic classification failed or returned invalid output"
        ) from exc
    finally:
        deployments.touch(slug)
    return result
