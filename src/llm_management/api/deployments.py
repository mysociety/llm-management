"""
HTTP contracts for deployment and local-resource management.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ..deployments.management import (
    AllDeploymentsResponse,
    CoolDownResponse,
    DeploymentOverview,
    DeploymentStatusResponse,
    GroupMemberResult,
    GroupWarmUpResponse,
    ScaleToZeroResponse,
    WarmUpResponse,
)
from ..deployments.types import GroupActionOutcome
from ..inference.resources import ResourceStatus
from .dependencies import DependencyService

__all__ = [
    "AllDeploymentsResponse",
    "CoolDownResponse",
    "DeploymentOverview",
    "DeploymentStatusResponse",
    "GroupMemberResult",
    "GroupWarmUpResponse",
    "ScaleToZeroResponse",
    "WarmUpResponse",
]


router = APIRouter(tags=["Deployments"])


@router.get("/deployments/{slug}/status")
def deployment_status(
    service: DependencyService, slug: str
) -> DeploymentStatusResponse:
    """
    Check whether a remote deployment exists and return its current replica count.
    """
    return service.management.status(slug)


@router.post("/deployments/{slug}/warm-up")
async def warm_up_deployment(
    service: DependencyService,
    slug: str,
    replicas: Annotated[int | None, Query(ge=1)] = None,
) -> WarmUpResponse | GroupWarmUpResponse:
    """
    Warm a local resource, remote deployment, or group_ prefixed group.
    Replica overrides apply only to individual remote deployments.
    """
    return await service.management.warm_up(slug, replicas)


@router.post("/deployments/{slug}/scale-to-zero")
async def scale_to_zero(
    service: DependencyService, slug: str
) -> ScaleToZeroResponse | GroupActionOutcome:
    """
    Stop one resource or every member of a group_ prefixed group.
    """
    return await service.management.stop(slug)


@router.post("/deployments/{slug}/cool-down")
async def cool_down_deployment(
    service: DependencyService, slug: str
) -> CoolDownResponse | GroupActionOutcome:
    """
    Shorten idle timers for one resource or a group_ prefixed group.
    New traffic resets the shortened countdown.
    """
    return await service.management.cool_down(slug)


@router.get("/deployments")
def all_deployments_overview(service: DependencyService) -> AllDeploymentsResponse:
    """
    Return configured remote deployments, status and idle countdowns.
    """
    return service.management.overview()


@router.get("/local-deployments")
async def local_deployment_status(service: DependencyService) -> list[ResourceStatus]:
    """
    CPU readiness and idle timers in this API worker.
    """
    return service.management.local_status()
