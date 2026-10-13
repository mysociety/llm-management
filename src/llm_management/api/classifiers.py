"""
HTTP contracts for local correspondence classification.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import automatic, sar
from .dependencies import AutomaticClassifierDependency, SARClassifierDependency


class SARDetectionRequest(BaseModel):
    body: str = Field(min_length=1, max_length=100_000)
    fast: bool = False


class AutomaticResponseRequest(BaseModel):
    body: str = Field(max_length=100_000)
    fast: bool = False


router = APIRouter(tags=["FOI endpoints"])


@router.post("/endpoint/foi/sar_detection", response_model=sar.SARResult)
async def sar_detection_endpoint(
    sar_classifier: SARClassifierDependency, body: SARDetectionRequest
) -> sar.SARResult:
    """
    Given an FOI request text, returns a quick indication of whether it is
    likely to be a SAR, IMM, or other personal information request.

    If `fast` is true, returns only logistic inference rather than cascading
    to the larger model.

    Relevant deployment group: `group_sar_pipeline_cpu`.
    """
    return await sar_classifier.detect(body.body, fast=body.fast)


@router.post(
    "/endpoint/foi/automatic_response_detection",
    response_model=automatic.AutomaticResult,
)
async def automatic_response_detection_endpoint(
    automatic_classifier: AutomaticClassifierDependency,
    body: AutomaticResponseRequest,
) -> automatic.AutomaticResult:
    """
    Given a response to an FOI request, returns a quick indication of whether
    it is likely to be an automatic response.

    If `fast` is true, returns only logistic inference rather than cascading
    to the larger model.

    Relevant deployment group: `group_automatic_pipeline_cpu`.
    """
    return await automatic_classifier.detect(body.body, fast=body.fast)
