"""
Integration mocks returning example responses without inference.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import HttpUrl

from ..agents.capital_city import CapitalCityResponse
from ..foi.email_finder.schemas import (
    Evidence,
    FOIFinding,
    FOIResult,
    FOIUsage,
    ModelUsage,
    PublicAuthority,
    SearchUsage,
)
from ..foi.examples import validation_examples
from ..foi.response_schemas import ExtractionOutput, ResponseAnalysisInput
from ..foi.schemas import (
    ExtractionBackend,
    ExtractQuestionsResponse,
    QuestionSliceRequest,
    QuestionSliceResult,
)
from .agents import CapitalCityRequest

router = APIRouter(prefix="/mock", tags=["Mocks"])


@router.post("/endpoint/misc/capital_city")
async def mock_capital_city(body: CapitalCityRequest) -> CapitalCityResponse:
    """Return Fake Paris for every country without running inference."""
    return CapitalCityResponse(country=body.country, city="Fake Paris")


@router.post("/endpoint/foi/extract_questions/extract")
async def mock_question_slice(
    body: QuestionSliceRequest,
    backend: ExtractionBackend = "cpu",
) -> QuestionSliceResult:
    """
    Validate extraction input and return a random gold reconstruction.

    Diagnostics are empty because no inference occurs. The model identifier
    marks this as a mock; backend is accepted and echoed without executing it.
    """
    example = random.choice(validation_examples().requests)
    return example.extraction_result(backend)


@router.post("/endpoint/foi/extract_questions")
async def mock_information_request(
    body: QuestionSliceRequest,
    backend: ExtractionBackend = "cpu",
) -> ExtractQuestionsResponse:
    """
    Return a random example's gold extraction and classification together.

    Request text belongs to the chosen example, independent of submitted text.
    """
    return ExtractQuestionsResponse.from_result(
        random.choice(validation_examples().requests).information_result(backend)
    )


@router.post("/endpoint/foi/response_status")
async def mock_response_status(body: ResponseAnalysisInput) -> ExtractionOutput:
    """
    Validate analysis input and return an unchanged random gold output.

    Output question and source IDs belong to the example, so they need not
    resolve against submitted input. This route tests response-shape handling.
    """
    return random.choice(validation_examples().responses).expected.model_copy(deep=True)


@router.post(
    "/endpoint/foi/find_email", summary="Mock a public authority's FOI email finding"
)
async def mock_find_foi_email(body: PublicAuthority) -> FOIResult:
    """
    Echo authority details with a fixed fictional finding and zero external usage.

    No credentials, finder configuration, inference or web requests are needed.
    The email and evidence are examples, independent of the submitted authority.
    """
    return FOIResult(
        authority=body,
        finding=FOIFinding(
            status="found",
            email="foi@council.example",
            email_type="foi_eir",
            evidence=Evidence(
                source_url=HttpUrl("https://council.example/foi"),
                quote="For new FOI requests, email foi@council.example.",
            ),
            explanation=(
                "Mock result for integration testing. The address and evidence "
                "are fictional; no research was performed."
            ),
        ),
        checked_at=datetime.now(timezone.utc),
        searches=[],
        pages=[],
        usage=FOIUsage(
            model=ModelUsage(
                provider="mock",
                model="mock:foi-email-finder",
                requests=0,
                input_tokens=0,
                cached_input_tokens=0,
                cache_write_tokens=0,
                output_tokens=0,
            ),
            search=SearchUsage(provider=None, requests=0, credits=0),
        ),
    )
