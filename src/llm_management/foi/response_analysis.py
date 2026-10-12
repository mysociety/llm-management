"""
Sanitized FOI response analysis using the configured remote status checkpoint.
"""

from __future__ import annotations

import asyncio

from ..deployments.errors import LLMManagementError
from ..deployments.types import DeploymentAccess
from ..inference.errors import ClassifierUnavailable
from ..sanitization import Sanitized, presidio, require_sanitized
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
    deployments: DeploymentAccess | None = None,
    resources: FOIResources | None = None,
) -> ExtractionOutput:
    """
    Prepare bounded status input before ensuring and calling its remote deployment.

    Reject raw input before resolving any runtime dependencies. Use the application's
    retained tokenizer and deployment service, or the process-local defaults for
    direct callers. Track remote activity on every attempted startup or inference.
    """
    clean = require_sanitized(observed)
    if resources is None or deployments is None:
        from ..bootstrap import default_application

        application = default_application()
        resources = resources if resources is not None else application.foi
        service = application.deployments
        deployments = (
            deployments
            if deployments is not None
            else DeploymentAccess(
                service.get_config, service.ensure_running, service.cache.touch
            )
        )
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
        running = await deployments.ensure_running(slug)
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
    finally:
        deployments.touch(slug)


async def analyze_response(
    body: ResponseAnalysisInput,
    *,
    sanitizer: TextSanitizer | None = None,
    deployments: DeploymentAccess | None = None,
    resources: FOIResources | None = None,
) -> ExtractionOutput:
    """
    Project upstream diagnostics out, sanitize response text and validate model references.

    Preserve source metadata and question IDs in the sanitized snapshot. Caller-facing
    output follows the public v4 schema; process_states remains unassessed. Validate
    against original input as well so injected predictors obey the same contract.
    """
    observed = body.model_input()
    try:
        sanitized = await asyncio.to_thread(
            sanitize_extraction,
            sanitizer
            if sanitizer is not None
            else resources.sanitizer()
            if resources is not None
            else presidio,
            observed,
        )
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable(str(exc)) from exc
    if resources is None:
        from ..deployments.catalog import get_catalog

        catalog = get_catalog()
    else:
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
