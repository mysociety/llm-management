"""
HTTP contracts for deployment and local-resource management.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from ..deployments.contracts import (
    AllDeploymentsResponse,
    CoolDownResponse,
    DeploymentStatusResponse,
    GroupActionOutcome,
    GroupWarmUpResponse,
    ScaleToZeroResponse,
    WarmUpResponse,
)
from ..inference.resources import ResourceStatus
from . import DependencyRuntime

router = APIRouter(tags=["Deployments"])


@router.get("/deployments/{slug}/status")
def deployment_status(
    runtime: DependencyRuntime, slug: str
) -> DeploymentStatusResponse:
    """
    Check whether a remote deployment exists and return its current replica count.
    """
    return runtime.management.status(slug)


@router.post("/deployments/{slug}/warm-up")
async def warm_up_deployment(
    runtime: DependencyRuntime,
    slug: str,
    replicas: Annotated[int | None, Query(ge=1)] = None,
) -> WarmUpResponse | GroupWarmUpResponse:
    """
    Warm a local resource, remote deployment, or group_ prefixed group.
    Replica overrides apply only to individual remote deployments.
    """
    return await runtime.management.warm_up(slug, replicas)


@router.post("/deployments/{slug}/scale-to-zero")
async def scale_to_zero(
    runtime: DependencyRuntime, slug: str
) -> ScaleToZeroResponse | GroupActionOutcome:
    """
    Stop one resource or every member of a group_ prefixed group.
    """
    return await runtime.management.stop(slug)


@router.post("/deployments/{slug}/cool-down")
async def cool_down_deployment(
    runtime: DependencyRuntime, slug: str
) -> CoolDownResponse | GroupActionOutcome:
    """
    Shorten idle timers for one resource or a group_ prefixed group.
    New traffic resets the shortened countdown.
    """
    return await runtime.management.cool_down(slug)


@router.get("/deployments")
def all_deployments_overview(
    runtime: DependencyRuntime,
) -> AllDeploymentsResponse:
    """
    Return configured remote deployments, status and idle countdowns.
    """
    return runtime.management.overview()


@router.get("/local-deployments")
async def local_deployment_status(
    runtime: DependencyRuntime,
) -> list[ResourceStatus]:
    """
    CPU readiness and idle timers in this API worker.
    """
    return runtime.management.local_status()
