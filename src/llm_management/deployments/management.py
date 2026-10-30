"""
Management commands and status views shared by HTTP and direct callers.

This layer interprets the public group_ target convention and shapes management
results. DeploymentService owns execution, locks and leases. Reading status never
warms a local model or starts a remote deployment.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..inference.resources import ResourceStatus
from .errors import DeploymentServiceError
from .slugs import is_group_slug
from .types import GroupActionOutcome

if TYPE_CHECKING:
    from .service import DeploymentService


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


class DeploymentManagement:
    """
    Application-scoped management operations backed by one deployment service.

    Direct callers use the same commands, validation and results as HTTP routes.
    Group execution remains in the service so its existing public methods keep
    working. No models or cloud controllers are constructed by this wrapper.
    """

    def __init__(self, service: DeploymentService) -> None:
        """
        Bind the service whose resources and activity this view manages.
        """
        self.service = service

    async def warm_up(
        self, slug: str, replicas: int | None = None
    ) -> WarmUpResponse | GroupWarmUpResponse:
        """
        Warm a resource or group; only remote resources accept replica overrides.
        """
        if is_group_slug(slug):
            if replicas is not None:
                raise DeploymentServiceError(
                    status_code=422, detail="Groups do not support replica overrides."
                )
            outcome = await self.service.ensure_group(slug)
            return GroupWarmUpResponse(
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
        single = await self.service.ensure_deployment(slug, replicas)
        return WarmUpResponse(
            slug=single.slug, action=single.action, replicas=single.replicas
        )

    async def stop(self, slug: str) -> ScaleToZeroResponse | GroupActionOutcome:
        """
        Stop one resource or all group members, preserving per-member failures.
        """
        if is_group_slug(slug):
            return await self.service.change_group(slug, "scale-to-zero")
        await self.service.scale_to_zero(slug)
        return ScaleToZeroResponse(slug=slug, action="scaled_to_zero")

    async def cool_down(self, slug: str) -> CoolDownResponse | GroupActionOutcome:
        """
        Shorten idle timers without interrupting work or immediately stopping models.
        """
        if is_group_slug(slug):
            return await self.service.change_group(slug, "cool-down")
        remaining = await self.service.cooldown(slug)
        return CoolDownResponse(
            slug=slug,
            action="already_stopped" if remaining is None else "cooling_down",
            seconds_until_scale_to_zero=remaining,
        )

    def status(self, slug: str) -> DeploymentStatusResponse:
        """
        Return remote existence and replicas using the service's status cache.
        """
        config = self.service.get_config(slug)
        state = self.service.status.ensure(self.service.get_backend(config.slug))
        return DeploymentStatusResponse(
            slug=slug, exists=state.exists, replicas=state.replicas
        )

    def overview(self) -> AllDeploymentsResponse:
        """
        Describe configured remote deployments and their current idle countdowns.
        """
        now = time.time()
        overviews: list[DeploymentOverview] = []
        for config in self.service.catalog.exoscale.deployment:
            state = self.service.status.ensure(self.service.get_backend(config.slug))
            activity = self.service.activity.get(config.slug)
            idle = (
                now - activity.last_request_time
                if activity.last_request_time > 0 and state.replicas > 0
                else None
            )
            remaining = (
                max(0.0, self.service.idle_seconds - idle) if idle is not None else None
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
            idle_timeout_minutes=int(self.service.idle_seconds / 60),
            deployments=overviews,
        )

    def local_status(self) -> list[ResourceStatus]:
        """
        Snapshot only constructed local guards without creating configured owners.
        """
        return [resource.status() for resource in self.service.resources.all()]
