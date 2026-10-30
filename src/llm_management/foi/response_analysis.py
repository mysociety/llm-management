"""
Sanitized FOI response analysis using the configured remote status checkpoint.
"""

from __future__ import annotations

import asyncio

from ..deployments.errors import LLMManagementError
from ..deployments.types import DeploymentInference
from ..inference.errors import ClassifierUnavailable
from ..sanitization import Sanitized, require_sanitized
from ..sanitization.contracts import TextSanitizer
from . import response_model
from .backends import FOIResources
from .pipeline import PipelineInputError, PipelineOutputError, PipelineUnavailable
from .response_schemas import ExtractionInput, ExtractionOutput, ResponseAnalysisInput
from .sanitization import sanitize_extraction


async def predict(
    observed: Sanitized[ExtractionInput],
    *,
    model: str,
    deployments: DeploymentInference,
    resources: FOIResources,
) -> ExtractionOutput:
    """
    Prepare bounded status input before ensuring and calling its remote deployment.

    Reject raw input before using the explicitly supplied application dependencies.
    Use its retained tokenizer and lease the deployment through remote inference.
    """
    clean = require_sanitized(observed)
    config = resources.catalog.require_foi()
    slug = config.response_deployment
    if slug is None or resources.response_tokenizer is None:
        raise PipelineUnavailable("FOI response analysis is not configured")
    try:
        messages = await asyncio.to_thread(
            response_model.prepare_request,
            observed,
            tokenizer=resources.response_tokenizer(),
            catalog=resources.catalog,
        )
    except ValueError as exc:
        raise PipelineInputError(str(exc)) from exc
    except Exception as exc:
        raise PipelineUnavailable(
            "FOI status tokenizer or instruction unavailable"
        ) from exc
    cfg = deployments.get_config(slug)
    if cfg.model != model:
        raise PipelineUnavailable(
            "FOI status deployment does not match the configured model"
        )
    try:
        async with deployments.use(slug) as running:
            if not running.state.api_key or not running.state.deployment_url:
                raise PipelineUnavailable(
                    "FOI status inference connection is unavailable"
                )
            output = await response_model.classify_status(
                messages,
                model=running.config.model,
                deployment_url=running.state.deployment_url,
                api_key=running.state.api_key,
                output_tokens=config.response.output_tokens,
            )
            return output.validate_against(clean)
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable("FOI status contract unavailable") from exc
    except LLMManagementError as exc:
        raise PipelineUnavailable("FOI status deployment could not be started") from exc
    except (response_model.StatusOutputError, ValueError) as exc:
        raise PipelineOutputError(
            "Status model returned invalid analysis or failed"
        ) from exc


async def analyze_response(
    body: ResponseAnalysisInput,
    *,
    sanitizer: TextSanitizer | None = None,
    deployments: DeploymentInference,
    resources: FOIResources,
) -> ExtractionOutput:
    """
    Project upstream diagnostics out, sanitize response text and validate model references.

    Preserve source metadata and question IDs in the sanitized snapshot. Caller-facing
    output follows the public v4 schema; process_states remains unassessed. Validate
    against original input as well so injected predictors obey the same contract.

    Use the supplied resources and deployment client throughout the operation.
    An explicit sanitizer can override the resource container's sanitizer.
    """
    observed = body.model_input()
    try:
        sanitized = await asyncio.to_thread(
            sanitize_extraction,
            sanitizer if sanitizer is not None else resources.sanitizer(),
            observed,
        )
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable(str(exc)) from exc
    catalog = resources.catalog
    slug = catalog.require_foi().response_deployment
    if slug is None:
        raise PipelineUnavailable("FOI response analysis is not configured")
    model = catalog.get(slug).model
    output = await predict(
        sanitized, model=model, deployments=deployments, resources=resources
    )
    try:
        output = ExtractionOutput.model_validate(output)
        return output.validate_against(observed)
    except ValueError as exc:
        raise PipelineOutputError("Response model returned invalid analysis") from exc
