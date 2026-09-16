"""Public async FOI operations, usable from HTTP routes or batch callers.

extract_questions: segmentation -> backend inference -> reconstruction.
process_information_request: extraction -> bounded Granite classification.
Deployment access is injected so this module never imports the HTTP server.
"""

import asyncio
import logging

import httpx

from ..errors import ClassifierOutputError, ClassifierUnavailable
from ..models import LLMManagementError
from ..settings import settings
from . import backends, granite
from .backends import DeploymentAccess
from .model_spec import GRANITE_MERGED
from .question_slice import build_extraction_result, contextual_inputs, segment_request
from .schemas import ExtractionBackend, InformationRequestResult, QuestionSliceResult

logger = logging.getLogger(__name__)


class PipelineInputError(ValueError):
    """Input exceeds a supported model budget or is otherwise invalid."""


class PipelineUnavailable(RuntimeError):
    """A required model, tokenizer or deployment is unavailable."""


class PipelineOutputError(RuntimeError):
    """A model request failed or produced invalid/incomplete output."""


async def extract_questions(
    request_text: str,
    *,
    deployments: DeploymentAccess,
    backend: ExtractionBackend = "cpu",
) -> QuestionSliceResult:
    try:
        units = segment_request(request_text)
        classified = await backends.classify_question_units(
            contextual_inputs(units), backend=backend, deployments=deployments
        )
        return build_extraction_result(
            units,
            classified.probabilities,
            backend=backend,
            model=classified.model,
            revision=classified.revision,
        )
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


async def process_information_request(
    request_text: str,
    *,
    deployments: DeploymentAccess,
    backend: ExtractionBackend = "cpu",
) -> InformationRequestResult:
    extraction = await extract_questions(
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
            granite.prepare_topic_request, request_text, extraction.questions
        )
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc
    except Exception as exc:
        logger.warning("Granite tokenizer unavailable: %s", type(exc).__name__)
        raise PipelineUnavailable("Granite tokenizer unavailable") from exc
    slug = settings.foi_topic_deployment
    cfg = deployments.get_config(slug)
    if cfg.model != GRANITE_MERGED:
        raise PipelineUnavailable(
            "FOI classification requires the merged fine-tuned Granite checkpoint"
        )
    try:
        cfg, state = await deployments.ensure_running(slug)
        result.classification = await granite.classify_topics(
            payload=payload,
            model=cfg.model,
            deployment_url=state.deployment_url,
            api_key=state.api_key,
            question_ids=[q.question_id for q in extraction.questions],
        )
        result.classification_model = cfg.model
    except LLMManagementError as exc:
        raise PipelineUnavailable(
            "Fine-tuned Granite deployment could not be started"
        ) from exc
    except granite.TopicOutputError as exc:
        raise PipelineOutputError(str(exc)) from exc
    except httpx.HTTPError as exc:
        logger.warning("Granite classification failure: %s", type(exc).__name__)
        raise PipelineOutputError(
            "Fine-tuned Granite classification failed or returned invalid output"
        ) from exc
    finally:
        deployments.touch(slug)
    return result
