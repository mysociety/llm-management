"""
HTTP contracts for deployment and local-resource management.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..deployments.service import DeploymentService
from ..inference.resources import ResourceStatus


class DeploymentStatusResponse(BaseModel):
    slug: str
    exists: bool
    replicas: int


class EnsureResponse(BaseModel):
    slug: str
    action: str
    replicas: int | None


class GroupMemberResult(BaseModel):
    slug: str
    success: bool
    replicas: int | None = None
    error: str | None = None


class GroupEnsureResponse(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupMemberResult]


class ScaleToZeroResponse(BaseModel):
    slug: str
    action: str


class CooldownResponse(BaseModel):
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


def build_router(service: DeploymentService) -> APIRouter:
    router = APIRouter()

    @router.get("/deployments/{slug}/status")
    def deployment_status(slug: str) -> DeploymentStatusResponse:
        """Check whether a specific deployment exists on Exoscale and return
        its current replica count.
        """
        config = service.get_config(slug)
        state = service.cache.ensure(service.get_backend(config.slug))
        return DeploymentStatusResponse(
            slug=slug, exists=state.exists, replicas=state.replicas
        )

    @router.post("/deployments/{slug}/ensure")
    async def ensure_deployment(
        slug: str, replicas: Annotated[int | None, Query(ge=1)] = None
    ) -> EnsureResponse:
        """
        Warm a local resource or ensure a deployment at an optional replica count.
        """
        outcome = await service.ensure_deployment(slug, replicas)
        return EnsureResponse(
            slug=outcome.slug, action=outcome.action, replicas=outcome.replicas
        )

    @router.post("/deployment-groups/{slug}/ensure")
    async def ensure_deployment_group(slug: str) -> GroupEnsureResponse:
        """
        Prepare group members concurrently; report failures without rolling back peers.
        """
        outcome = await service.ensure_group(slug)
        return GroupEnsureResponse(
            slug=outcome.slug,
            success=outcome.success,
            deployments=[
                GroupMemberResult(
                    slug=member.slug,
                    success=member.success,
                    replicas=member.replicas,
                    error=member.error,
                )
                for member in outcome.deployments
            ],
        )

    @router.post("/deployments/{slug}/scale-to-zero")
    async def scale_to_zero(slug: str) -> ScaleToZeroResponse:
        """
        Pause managed inference or delete a Compute VM and its access resources.
        """
        await service.scale_to_zero(slug)
        return ScaleToZeroResponse(slug=slug, action="scaled_to_zero")

    @router.post("/deployments/{slug}/cooldown")
    async def cooldown_deployment(slug: str) -> CooldownResponse:
        """
        Shorten the idle countdown to at most a minute; new traffic resets it.
        """
        remaining = await service.cooldown(slug)
        return CooldownResponse(
            slug=slug,
            action="already_stopped" if remaining is None else "cooling_down",
            seconds_until_scale_to_zero=remaining,
        )

    @router.get("/deployments")
    def all_deployments_overview() -> AllDeploymentsResponse:
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
            remaining = (
                max(0.0, service.idle_seconds - idle) if idle is not None else None
            )
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

    @router.get("/local-models")
    async def local_model_status() -> list[ResourceStatus]:
        """
        CPU readiness and idle timers in this API worker.
        """
        return [resource.status() for resource in service.resources.all()]

    @router.post("/local-models/{name}/ensure")
    async def ensure_local_model(name: str) -> ResourceStatus:
        """
        Load a CPU resource and reset its idle timer.
        """
        return await service.ensure_local(name)

    return router
