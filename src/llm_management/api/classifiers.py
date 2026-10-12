"""
HTTP contracts for local correspondence classification.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .. import automatic, sar
from ..automatic.pipeline import AutomaticClassifier
from ..deployments.service import DeploymentService
from ..sar.pipeline import SARClassifier


class SARDetectionRequest(BaseModel):
    request: str = Field(min_length=1, max_length=100_000)
    fast: bool = False


class AutomaticResponseRequest(BaseModel):
    body: str
    fast: bool = False


def build_router(
    service: DeploymentService,
    sar_classifier: SARClassifier,
    automatic_classifier: AutomaticClassifier,
) -> APIRouter:
    router = APIRouter()

    @router.post("/agents/sar_detection", response_model=sar.SARResult)
    async def sar_detection_endpoint(body: SARDetectionRequest) -> sar.SARResult:
        """
        Flag personal-records correspondence; inference failures retain the flag.
        """
        return await sar_classifier.detect(body.request, fast=body.fast)

    @router.post(
        "/agents/automatic_response_detection", response_model=automatic.AutomaticResult
    )
    async def automatic_response_detection_endpoint(
        body: AutomaticResponseRequest,
    ) -> automatic.AutomaticResult:
        """
        Classify one current email body, leaving notification decisions to callers.
        """
        return await automatic_classifier.detect(body.body, fast=body.fast)

    return router
