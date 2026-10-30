"""
Management commands and status views shared by HTTP and direct callers.

This layer interprets the public group_ target convention and shapes management
results. DeploymentService owns execution, locks and leases. Reading status never
warms a local model or starts a remote deployment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from ..inference.resources import LocalResource, ResourceStatus
from .errors import DeploymentServiceError, LLMManagementError
from .slugs import is_group_slug
from .types import (
    GroupActionMemberOutcome,
    GroupActionOutcome,
    GroupMemberResult,
    GroupWarmUpResponse,
    LifecycleAction,
    LifecycleCommand,
    WarmUpAction,
    WarmUpResponse,
)

if TYPE_CHECKING:
    from .service import DeploymentService


logger = logging.getLogger(__name__)
MemberOutcome = TypeVar("MemberOutcome")


class DeploymentStatusResponse(BaseModel):
    slug: str
    exists: bool
    replicas: int


class ScaleToZeroResponse(BaseModel):
    slug: str
    action: LifecycleAction


class CoolDownResponse(BaseModel):
    slug: str
    action: LifecycleAction
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
    Single-resource and group commands share result construction here. The service
    owns resource execution and leases. Binding this facade constructs no backends.
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
            return await self.warm_up_group(slug)
        return await self.warm_up_resource(slug, replicas)

    async def stop(self, slug: str) -> ScaleToZeroResponse | GroupActionOutcome:
        """
        Stop one resource or all group members, preserving per-member failures.
        """
        if is_group_slug(slug):
            return await self.change_group(slug, LifecycleCommand.STOP)
        return await self.stop_resource(slug)

    async def stop_resource(self, slug: str) -> ScaleToZeroResponse:
        """
        Stop one resource and describe the completed action.
        """
        await self.service.scale_to_zero(slug)
        return ScaleToZeroResponse(slug=slug, action=LifecycleAction.SCALED_TO_ZERO)

    async def cool_down(self, slug: str) -> CoolDownResponse | GroupActionOutcome:
        """
        Shorten idle timers without interrupting work or immediately stopping models.
        """
        if is_group_slug(slug):
            return await self.change_group(slug, LifecycleCommand.COOL_DOWN)
        return await self.cool_down_resource(slug)

    async def cool_down_resource(self, slug: str) -> CoolDownResponse:
        """
        Shorten one idle timer and describe whether the resource was already stopped.
        """
        remaining = await self.service.cooldown(slug)
        return CoolDownResponse(
            slug=slug,
            action=LifecycleAction.ALREADY_STOPPED
            if remaining is None
            else LifecycleAction.COOLING_DOWN,
            seconds_until_scale_to_zero=remaining,
        )

    async def warm_up_resource(
        self, slug: str, replicas: int | None = None
    ) -> WarmUpResponse:
        """
        Warm one configured resource and report the resulting management action.

        Local guards return a warmed outcome with no replica count and reject explicit
        replicas. Remote outcomes distinguish creation, resumption, resizing and an
        already-running deployment using its prior cached status.
        """
        target = self.service.resolve_resource(slug)
        if isinstance(target, LocalResource):
            if replicas is not None:
                raise DeploymentServiceError(
                    status_code=422, detail="Local resources do not support replicas."
                )
            await self.service.warmup_local(target)
            return WarmUpResponse(slug=slug, action=WarmUpAction.WARMED, replicas=None)
        backend = self.service.get_backend(slug)
        previous = await asyncio.to_thread(self.service.status.ensure, backend)
        running = await self.service.prepare(slug, allow_start=True, replicas=replicas)
        actual = running.state.replicas
        if previous.replicas > 0:
            if replicas is not None and previous.replicas != replicas:
                action = WarmUpAction.SCALED
            else:
                action = WarmUpAction.ALREADY_RUNNING
        elif previous.exists:
            action = WarmUpAction.RESUMED
        else:
            action = WarmUpAction.CREATED
        return WarmUpResponse(slug=slug, action=action, replicas=actual)

    async def run_group(
        self, slug: str, operation: Callable[[str], Awaitable[MemberOutcome]]
    ) -> list[MemberOutcome]:
        """
        Resolve a group once and execute members concurrently in configured order.

        Each operation defines its own public failure result. Keeping that policy
        in the operation preserves the distinct warm-up and lifecycle error contracts.
        """
        try:
            group = self.service.catalog.get_group(slug)
        except LLMManagementError as exc:
            raise DeploymentServiceError(
                status_code=404, detail=f"Deployment group '{slug}' not found"
            ) from exc
        return list(
            await asyncio.gather(*(operation(member) for member in group.deployments))
        )

    async def warm_up_group(self, slug: str) -> GroupWarmUpResponse:
        """
        Warm all group members concurrently and report their individual outcomes.

        Members may be local or remote. Preserve configured order in the result and
        report failures without rolling back successful members. Hide arbitrary loading
        exception messages; overall success requires every member to succeed.
        """

        async def ensure_member(member: str) -> GroupMemberResult:
            """
            Capture one member's failure without cancelling its successful peers.
            """
            try:
                replicas = await self.service.warmup_resource(member)
                return GroupMemberResult(slug=member, success=True, replicas=replicas)
            except Exception as exc:
                logger.warning(
                    "Group ensure failed for %s: %s", member, type(exc).__name__
                )
                return GroupMemberResult(
                    slug=member, success=False, error="Resource could not be warmed"
                )

        members = await self.run_group(slug, ensure_member)
        return GroupWarmUpResponse(
            slug=slug,
            success=all(member.success for member in members),
            deployments=members,
        )

    async def change_group(
        self, slug: str, action: LifecycleCommand
    ) -> GroupActionOutcome:
        """
        Apply a lifecycle action concurrently, preserving per-member failures.
        """

        async def change_member(member: str) -> GroupActionMemberOutcome:
            try:
                match action:
                    case LifecycleCommand.COOL_DOWN:
                        result = await self.cool_down_resource(member)
                        return GroupActionMemberOutcome(
                            slug=member,
                            success=True,
                            action=result.action,
                            seconds_until_scale_to_zero=result.seconds_until_scale_to_zero,
                        )
                    case LifecycleCommand.STOP:
                        stopped = await self.stop_resource(member)
                        return GroupActionMemberOutcome(
                            slug=member,
                            success=True,
                            action=stopped.action,
                        )
            except DeploymentServiceError as exc:
                return GroupActionMemberOutcome(
                    slug=member, success=False, error=exc.detail
                )
            except Exception as exc:
                logger.warning(
                    "Group %s failed for %s: %s", action, member, type(exc).__name__
                )
                return GroupActionMemberOutcome(
                    slug=member, success=False, error="Resource lifecycle action failed"
                )

        members = await self.run_group(slug, change_member)
        return GroupActionOutcome(
            slug=slug,
            success=all(member.success for member in members),
            deployments=members,
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
        for config in self.service.catalog.exoscale.deployment.values():
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
