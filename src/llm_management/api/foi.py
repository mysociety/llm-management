"""
HTTP contracts and error translation for FOI pipelines.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deployments.service import DeploymentService
from ..foi import backends, pipeline, response_analysis
from ..foi.response_schemas import ExtractionOutput, ResponseAnalysisInput
from ..foi.schemas import (
    ExtractionBackend,
    InformationRequestResult,
    QuestionSliceResult,
)
from ..inference.errors import ClassifierBusy


class QuestionSliceRequest(BaseModel):
    request: str = Field(min_length=1, max_length=100_000)


def build_router(
    service: DeploymentService, resources: backends.FOIResources
) -> APIRouter:
    router = APIRouter()

    def foi_deployments() -> backends.DeploymentAccess:
        """
        Bind the shared deployment lifecycle to the HTTP-independent FOI pipeline.
        """
        return backends.DeploymentAccess(
            service.get_config, service.ensure_running, service.cache.touch
        )

    @contextmanager
    def foi_http_errors() -> Iterator[None]:
        """
        Translate domain/runtime failures only at the HTTP boundary.
        """
        try:
            yield
        except ClassifierBusy as exc:
            raise HTTPException(
                status_code=429, detail=str(exc), headers={"Retry-After": "1"}
            ) from exc
        except pipeline.PipelineInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except pipeline.PipelineUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except pipeline.PipelineOutputError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @router.post("/agents/foi_structure/extract")
    async def question_slice_endpoint(
        body: QuestionSliceRequest,
        backend: ExtractionBackend = "cpu",
    ) -> QuestionSliceResult:
        """Extract source-grounded questions with either classifier backend.

        When no question starts exist, promote the first continuation locally.
        No OLMo escalation or Granite classification. CPU overload returns 429.
        """
        with foi_http_errors():
            return await pipeline.extract_questions(
                body.request,
                backend=backend,
                deployments=foi_deployments(),
                resources=resources,
            )

    @router.post("/agents/foi_structure")
    async def information_request_endpoint(
        body: QuestionSliceRequest,
        backend: ExtractionBackend = "cpu",
    ) -> InformationRequestResult:
        """
        QuestionSlice extraction followed by fine-tuned Granite topics/regimes.
        """
        with foi_http_errors():
            return await pipeline.process_information_request(
                body.request,
                backend=backend,
                deployments=foi_deployments(),
                resources=resources,
            )

    @router.post("/agents/foi_response_analysis", response_model=ExtractionOutput)
    async def foi_response_analysis_endpoint(
        body: ResponseAnalysisInput,
    ) -> ExtractionOutput:
        """
        Analyze response sources against the questions extracted from a request.
        """
        with foi_http_errors():
            try:
                return await response_analysis.analyze_response(
                    body, sanitizer=resources.sanitizer()
                )
            except response_analysis.ResponseModelNotReady as exc:
                raise HTTPException(status_code=501, detail=str(exc)) from exc

    return router
