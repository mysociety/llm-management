"""Public async FOI operations, usable from HTTP routes or batch callers.

extract_questions: segmentation -> backend inference -> reconstruction.
process_information_request: extraction -> topic classification -> category matching.
Deployment access is injected so this module never imports the HTTP server.
"""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

import httpx2

from ..deployments.errors import LLMManagementError
from ..deployments.local_config import TopicTokenizerDeployment
from ..deployments.types import DeploymentInference
from ..inference.errors import ClassifierOutputError, ClassifierUnavailable
from . import backends, category_classification, topic_classification
from .category_schemas import CategoryMetadata
from .question_slice import build_extraction_result, contextual_inputs, segment_request
from .sanitization import sanitize_request
from .schemas import ExtractionBackend, InformationRequestResult, QuestionSliceResult

logger = logging.getLogger(__name__)


class PipelineInputError(ValueError):
    """
    Input exceeds a supported model budget or is otherwise invalid.
    """


class PipelineUnavailable(RuntimeError):
    """
    A required model, tokenizer or deployment is unavailable.
    """


class PipelineOutputError(RuntimeError):
    """
    A model request failed or produced invalid/incomplete output.
    """


class QuestionExtractionResult(NamedTuple):
    """
    Original extraction and sanitized text for downstream classification.
    """

    original: QuestionSliceResult
    sanitized: QuestionSliceResult
    sanitized_request: str


async def _extract_questions(
    request_text: str,
    *,
    deployments: DeploymentInference,
    resources: backends.FOIResources,
    backend: ExtractionBackend = "cpu",
) -> QuestionExtractionResult:
    """
    Classify sanitized semantic units and reconstruct both views of the request.

    Original text is retained for caller-facing diagnostics; the matching
    sanitized extraction is used for downstream topic classification. Both
    views use the same predictions and question IDs. Sanitization runs in a
    worker thread before either local or remote classifier inference.

    Injected resources supply app-owned model providers. Translate input,
    availability and upstream failures into the corresponding pipeline errors
    shared by HTTP and batch callers.
    """
    try:
        units = segment_request(request_text)
        sanitized = await asyncio.to_thread(
            sanitize_request,
            resources.sanitizer(),
            request_text,
            [unit.text for unit in units],
            contextual_inputs(units),
        )
        clean_units = [
            unit.model_copy(update={"text": text})
            for unit, text in zip(units, sanitized.units, strict=True)
        ]
        classified = await backends.classify_question_units(
            sanitized.model_inputs,
            backend=backend,
            deployments=deployments,
            resources=resources,
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
        return QuestionExtractionResult(
            original=original, sanitized=clean, sanitized_request=sanitized.request_text
        )
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable(str(exc)) from exc
    except (ClassifierOutputError, httpx2.HTTPError) as exc:
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
    deployments: DeploymentInference,
    resources: backends.FOIResources,
    backend: ExtractionBackend = "cpu",
) -> QuestionSliceResult:
    """
    Extract questions and reconstruction diagnostics without classifying topics.

    Return original request text in the result while submitting sanitized text
    to the selected CPU or remote backend using the injected resources.
    Deployment access is required for the remote backend; this function does not
    start a topic-model deployment.

    Raises PipelineInputError for invalid inputs, PipelineUnavailable for missing
    resources, and PipelineOutputError for failed or invalid upstream responses.
    """
    extraction = await _extract_questions(
        request_text, deployments=deployments, backend=backend, resources=resources
    )
    return extraction.original


async def process_information_request(
    request_text: str,
    *,
    deployments: DeploymentInference,
    resources: backends.FOIResources,
    backend: ExtractionBackend = "cpu",
) -> InformationRequestResult:
    """
    Extract questions, classify topics/regimes, then match FOI/EIR categories locally.

    Retain original text and extraction diagnostics in the result. Send only the
    sanitized request and questions to the topic model, after checking the
    prompt budget and configured checkpoint. Requests without extracted
    questions return immediately with no classification or topic-model startup.

    Extraction may use either backend; topic classification uses the configured
    remote deployment. Record its activity even when classification fails.
    Input, resource and upstream failures propagate as the pipeline error types.
    """
    # Extract once, reconstructing original and sanitized question text with the
    # same IDs. Return original wording to callers; use sanitized text for inference.
    extracted = await _extract_questions(
        request_text, backend=backend, deployments=deployments, resources=resources
    )
    extraction = extracted.original
    result = InformationRequestResult(
        **extraction.model_dump(),
        request_text=request_text,
        classification=None,
        classification_model=None,
    )
    if not extraction.questions:
        # Nothing remains to classify, so skip both topic and category models.
        result.category_classification = CategoryMetadata(status="no_questions")
        return result

    # Build the topic model's chat payload, sanitize the assembled messages and
    # check prompt/output budgets before starting remote compute. Tokenizer loading
    # and counting run in a worker thread so other async requests can keep running.
    try:
        payload = await asyncio.to_thread(
            topic_classification.prepare_topic_request,
            extracted.sanitized_request,
            extracted.sanitized.questions,
            catalog=resources.catalog,
            # These providers resolve app-owned resources here; preparation loads
            # and uses the tokenizer under its resource lease in the worker thread.
            sanitizer=resources.sanitizer(),
            tokenizer_guard=resources.tokenizer(),
        )
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc
    except Exception as exc:
        logger.warning("Question topic tokenizer unavailable: %s", type(exc).__name__)
        raise PipelineUnavailable("Question topic tokenizer unavailable") from exc

    # Check that the remote model matches the checkpoint used by the local
    # tokenizer and prompt limits, rather than sending this payload to another model.
    catalog = resources.catalog
    foi = catalog.require_foi()
    slug = foi.topic_deployment
    tokenizer = catalog.local.get(foi.tokenizer, TopicTokenizerDeployment)
    cfg = deployments.get_config(slug)
    if cfg.model != catalog.model[tokenizer.model_ref].repo:
        raise PipelineUnavailable(
            "FOI classification requires the configured fine-tuned topic checkpoint"
        )

    # Start/resume the remote topic model and classify the sanitized questions.
    # Expected question IDs let the response validator detect missing or reordered
    # answers; the returned topics and regimes are retained alongside extraction.
    try:
        async with deployments.use(slug) as deployment:
            cfg = deployment.config
            state = deployment.state
            result.classification = await topic_classification.classify_topics(
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
    except topic_classification.TopicOutputError as exc:
        raise PipelineOutputError(str(exc)) from exc
    except httpx2.HTTPError as exc:
        logger.warning("Topic classification failure: %s", type(exc).__name__)
        raise PipelineOutputError(
            "Question topic classification failed or returned invalid output"
        ) from exc

    # Fill each FOI/EIR question's nullable category using its generated description.
    # SAR questions stay uncategorized. Shared status/provenance goes into metadata;
    # failures preserve question descriptions, and an absent provider disables matching.
    if result.classification is not None and resources.category_matcher is not None:
        result.category_classification = (
            await category_classification.classify_categories(
                result.classification, resources.category_matcher
            )
        )
    return result
