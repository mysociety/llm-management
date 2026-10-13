"""
HTTP contracts for deployment and local-resource management.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..deployments.types import GroupActionOutcome
from ..inference.resources import ResourceStatus
from .dependencies import ServiceDependency


class DeploymentStatusResponse(BaseModel):
    slug: str
    exists: bool
    replicas: int


class WarmUpResponse(BaseModel):
    slug: str
    action: str
    replicas: int | None


class GroupMemberResult(BaseModel):
    slug: str
    success: bool
    replicas: int | None = None
    error: str | None = None


class GroupWarmUpResponse(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupMemberResult]


class ScaleToZeroResponse(BaseModel):
    slug: str
    action: str


class CoolDownResponse(BaseModel):
    slug: str
    action: str
    seconds_until_scale_to_zero: float | None


class DeploymentOverview(BaseModel):
    """
    Summary of a single deployment's current state and idle timer.
    """

    slug: str
    deployment_name: str
    exists: bool
    replicas: int
    idle_seconds: float | None
    seconds_until_scale_to_zero: float | None


class AllDeploymentsResponse(BaseModel):
    """
    Overview of all configured deployments with their current state
    and time remaining before auto-scaling to zero.
    """

    idle_timeout_minutes: int
    deployments: list[DeploymentOverview]


router = APIRouter(tags=["Deployments"])


@router.get("/deployments/{slug}/status")
def deployment_status(
    service: ServiceDependency, slug: str
) -> DeploymentStatusResponse:
    """Check whether a specific deployment exists on Exoscale and return
    its current replica count.
    """
    config = service.get_config(slug)
    state = service.cache.ensure(service.get_backend(config.slug))
    return DeploymentStatusResponse(
        slug=slug, exists=state.exists, replicas=state.replicas
    )


@router.post("/deployments/{slug}/warm-up")
async def warm_up_deployment(
    service: ServiceDependency,
    slug: str,
    replicas: Annotated[int | None, Query(ge=1)] = None,
) -> WarmUpResponse | GroupWarmUpResponse:
    """
    Warm a local resource, remote deployment, or group_ prefixed group.
    Replica overrides apply only to individual remote deployments.
    """
    if slug.startswith("group_"):
        if replicas is not None:
            raise HTTPException(
                status_code=422, detail="Groups do not support replica overrides."
            )
        group_outcome = await service.ensure_group(slug)
        return GroupWarmUpResponse.model_validate(group_outcome.model_dump())
    outcome = await service.ensure_deployment(slug, replicas)
    return WarmUpResponse(
        slug=outcome.slug, action=outcome.action, replicas=outcome.replicas
    )


@router.post("/deployments/{slug}/scale-to-zero")
async def scale_to_zero(
    service: ServiceDependency, slug: str
) -> ScaleToZeroResponse | GroupActionOutcome:
    """
    Stop one resource or every member of a group_ prefixed group.
    """
    if slug.startswith("group_"):
        return await service.change_group(slug, "scale-to-zero")
    await service.scale_to_zero(slug)
    return ScaleToZeroResponse(slug=slug, action="scaled_to_zero")


@router.post("/deployments/{slug}/cool-down")
async def cool_down_deployment(
    service: ServiceDependency, slug: str
) -> CoolDownResponse | GroupActionOutcome:
    """
    Shorten idle timers for one resource or a group_ prefixed group.
    New traffic resets the shortened countdown.
    """
    if slug.startswith("group_"):
        return await service.change_group(slug, "cool-down")
    remaining = await service.cooldown(slug)
    return CoolDownResponse(
        slug=slug,
        action="already_stopped" if remaining is None else "cooling_down",
        seconds_until_scale_to_zero=remaining,
    )


@router.get("/deployments")
def all_deployments_overview(service: ServiceDependency) -> AllDeploymentsResponse:
    """Return the current status of every configured deployment, including
    how long each has been idle and how many seconds remain before it
    will be automatically scaled to zero.
    """
    now = time.time()
    overviews = []
    for config in service.catalog.exoscale.deployment:
        state = service.cache.ensure(service.get_backend(config.slug))
        idle = (
            now - state.last_request_time
            if state.last_request_time > 0 and state.replicas > 0
            else None
        )
        remaining = max(0.0, service.idle_seconds - idle) if idle is not None else None
        overviews.append(
            DeploymentOverview(
                slug=config.slug,
                deployment_name=config.deployment_name,
                exists=state.exists,
                replicas=state.replicas,
                idle_seconds=idle,
                seconds_until_scale_to_zero=remaining,
            )
        )
    return AllDeploymentsResponse(
        idle_timeout_minutes=int(service.idle_seconds / 60), deployments=overviews
    )


@router.get("/local-deployments")
async def local_deployment_status(service: ServiceDependency) -> list[ResourceStatus]:
    """
    CPU readiness and idle timers in this API worker.
    """
    return [resource.status() for resource in service.resources.all()]
