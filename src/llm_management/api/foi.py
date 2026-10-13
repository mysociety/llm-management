"""
HTTP contracts and error translation for FOI pipelines.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import APIRouter, HTTPException

from ..foi import backends, pipeline, response_analysis
from ..foi.response_schemas import ExtractionOutput, ResponseAnalysisInput
from ..foi.schemas import (
    ExtractionBackend,
    ExtractQuestionsResponse,
    QuestionSliceRequest,
    QuestionSliceResult,
)
from ..inference.errors import ClassifierBusy
from .dependencies import FOIResourcesDependency, ServiceDependency

router = APIRouter(tags=["FOI endpoints"])


def foi_deployments(service: ServiceDependency) -> backends.DeploymentAccess:
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


@router.post(
    "/endpoint/foi/extract_questions/extract",
    summary="Extract question structure (debugging)",
)
async def question_slice_endpoint(
    service: ServiceDependency,
    resources: FOIResourcesDependency,
    body: QuestionSliceRequest,
    backend: ExtractionBackend = "cpu",
) -> QuestionSliceResult:
    """Returns just the question structure, primarily for internal debugging.

    This omits FOI/EIR/SAR status and topic classification.

    Relevant deployment group: `group_foi_pipeline_cpu` for `backend=cpu`
    (the default), or `group_foi_pipeline` for `backend=exoscale`.
    These groups warm the full pipeline, including classification.
    """
    with foi_http_errors():
        return await pipeline.extract_questions(
            body.body,
            backend=backend,
            deployments=foi_deployments(service),
            resources=resources,
        )


@router.post("/endpoint/foi/extract_questions", summary="Extract questions")
async def information_request_endpoint(
    service: ServiceDependency,
    resources: FOIResourcesDependency,
    body: QuestionSliceRequest,
    backend: ExtractionBackend = "cpu",
) -> ExtractQuestionsResponse:
    """
    Given an FOI request, returns a structured set of questions with
    FOI/EIR/SAR status and a short description.

    Relevant deployment group: `group_foi_pipeline_cpu` for `backend=cpu`
    (the default), or `group_foi_pipeline` for `backend=exoscale`.
    """
    with foi_http_errors():
        result = await pipeline.process_information_request(
            body.body,
            backend=backend,
            deployments=foi_deployments(service),
            resources=resources,
        )

        return ExtractQuestionsResponse.from_result(result)


@router.post(
    "/endpoint/foi/response_status",
    response_model=ExtractionOutput,
    summary="Response status",
)
async def foi_response_status_endpoint(
    service: ServiceDependency,
    resources: FOIResourcesDependency,
    body: ResponseAnalysisInput,
) -> ExtractionOutput:
    """
    Given a structured set of questions and a structured response document,
    identifies whether questions were answered or withheld.

    Relevant deployment group: `group_foi_response_pipeline`.
    """
    with foi_http_errors():
        return await response_analysis.analyze_response(
            body, deployments=foi_deployments(service), resources=resources
        )
