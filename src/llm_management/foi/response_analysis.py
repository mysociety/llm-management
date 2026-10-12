"""
Response analysis boundary; inference awaits the response model checkpoint.
"""

from __future__ import annotations

import asyncio

from ..inference.errors import ClassifierUnavailable
from ..sanitization import Sanitized, presidio, require_sanitized
from ..sanitization.contracts import TextSanitizer
from .pipeline import PipelineOutputError, PipelineUnavailable
from .response_schemas import ExtractionInput, ExtractionOutput, ResponseAnalysisInput
from .sanitization import sanitize_extraction

RESPONSE_ANALYSIS_MODEL = "foi-response-analysis-placeholder"


class ResponseModelNotReady(RuntimeError):
    """
    Response model inference is not connected yet.
    """


async def predict(
    observed: Sanitized[ExtractionInput], *, model: str
) -> ExtractionOutput:
    """
    Replace with inference once the trained model and its serving contract exist.
    """
    require_sanitized(observed)
    raise ResponseModelNotReady(f"Model inference is not connected: {model}")


async def analyze_response(
    body: ResponseAnalysisInput, *, sanitizer: TextSanitizer | None = None
) -> ExtractionOutput:
    observed = body.model_input()
    try:
        sanitized = await asyncio.to_thread(
            sanitize_extraction,
            sanitizer if sanitizer is not None else presidio,
            observed,
        )
    except ClassifierUnavailable as exc:
        raise PipelineUnavailable(str(exc)) from exc
    output = await predict(sanitized, model=RESPONSE_ANALYSIS_MODEL)
    try:
        # Validate both the structure and references, including mocked/new adapters.
        output = ExtractionOutput.model_validate(output)
        return output.validate_against(observed)
    except ValueError as exc:
        raise PipelineOutputError("Response model returned invalid analysis") from exc
