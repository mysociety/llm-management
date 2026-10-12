"""
FOI integration mocks returning bundled examples without inference.
"""

from __future__ import annotations

import random

from fastapi import APIRouter

from ..foi.examples import validation_examples
from ..foi.response_schemas import ExtractionOutput, ResponseAnalysisInput
from ..foi.schemas import (
    ExtractionBackend,
    InformationRequestResult,
    QuestionSliceRequest,
    QuestionSliceResult,
)


def build_router() -> APIRouter:
    """
    Mirror FOI input contracts while returning randomly chosen example outputs.

    Normal application authentication applies. No deployment service or local model
    is consulted; example outputs are independent of the submitted correspondence.
    """
    router = APIRouter(prefix="/mock", tags=["FOI mocks"])

    @router.post("/agents/foi_structure/extract")
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

    @router.post("/agents/foi_structure")
    async def mock_information_request(
        body: QuestionSliceRequest,
        backend: ExtractionBackend = "cpu",
    ) -> InformationRequestResult:
        """
        Return a random example's gold extraction and classification together.

        Request text belongs to the chosen example, independent of submitted text.
        """
        return random.choice(validation_examples().requests).information_result(backend)

    @router.post("/agents/foi_response_analysis")
    async def mock_response_analysis(body: ResponseAnalysisInput) -> ExtractionOutput:
        """
        Validate analysis input and return an unchanged random gold output.

        Output question and source IDs belong to the example, so they need not
        resolve against submitted input. This route tests response-shape handling.
        """
        return random.choice(validation_examples().responses).expected.model_copy(
            deep=True
        )

    return router
