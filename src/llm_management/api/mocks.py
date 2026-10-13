"""
Integration mocks returning example responses without inference.
"""

from __future__ import annotations

import random

from fastapi import APIRouter

from ..agents.capital_city import CapitalCityResponse
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
