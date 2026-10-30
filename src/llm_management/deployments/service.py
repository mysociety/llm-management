"""Deployment lifecycle coordination independent of HTTP handlers.

One service coordinates a catalog, remote state cache and local resources within
one process. Worker startup remains shielded so cancellation cannot abandon a VM
before it is tracked for later cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Literal, TypeVar

from ..inference.errors import ClassifierBusy, ClassifierUnavailable
from ..inference.resources import LocalResource, ResourceRegistry, ResourceStatus
from .activity import DeploymentActivityTracker
from .backends import DeploymentBackend, backend_for
from .cache import DeploymentCache
from .catalog import DeploymentCatalog
from .config import DeploymentConfig
from .errors import DeploymentServiceError, LLMManagementError
from .management import DeploymentManagement
from .status import DeploymentStatusCache
from .types import (
    DeploymentSnapshot,
    GroupActionMemberOutcome,
    GroupActionOutcome,
    GroupMemberResult,
    GroupWarmUpResponse,
    RunningDeployment,
    WarmUpResponse,
)

logger = logging.getLogger(__name__)

MemberOutcome = TypeVar("MemberOutcome")


class DeploymentService:
    """
    Coordinate remote deployments and local resources owned by one application.

    Remote startup and scaling share per-slug locks. Explicit inference leases
    prevent idle cleanup from stopping deployments still serving a caller.
    Local owners use separate worker-thread resource leases. This service schedules
    cleanup; neither the catalog nor the registry runs a background timer itself.
    """

    def __init__(
        self,
        catalog: DeploymentCatalog,
        cache: DeploymentCache | None = None,
        resources: ResourceRegistry | None = None,
        *,
        status: DeploymentStatusCache | None = None,
        activity: DeploymentActivityTracker | None = None,
        auto_ensure: bool = True,
        idle_seconds: float = 900,
        cpu_idle_seconds: float = 900,
        check_interval: float = 60,
    ) -> None:
        """
        Bind application-owned configuration, state and resource guards.

        All timeout values are seconds. Remote inactivity uses wall-clock request times;
        local resource expiry uses the guards' monotonic activity times. Construction
        creates no cloud backends and starts no cleanup task. Assembly supplies status
        and activity independently. The positional cache argument is retained for
        older callers; mixing it with separate state dependencies is rejected.
        """
        self.management = DeploymentManagement(self)
        self.catalog = catalog
        self.backends: dict[str, DeploymentBackend] = {}
        self._backend_lock = threading.Lock()
        if cache is not None and (status is not None or activity is not None):
            raise ValueError(
                "Supply either a legacy cache or separate status and activity"
            )
        self._cache: DeploymentCache | None = None
        if cache is not None:
            self.cache = cache
        else:
            self.status = status if status is not None else DeploymentStatusCache()
            self.activity = (
                activity if activity is not None else DeploymentActivityTracker()
            )
        self.resources = resources if resources is not None else ResourceRegistry()
        self.auto_ensure = auto_ensure
        self.idle_seconds = idle_seconds
        self.cpu_idle_seconds = cpu_idle_seconds
        self.check_interval = check_interval
        self.request_deployments: ContextVar[set[str] | None] = ContextVar(
            "request_deployments", default=None
        )

    @property
    def cache(self) -> DeploymentCache:
        """
        Legacy combined-state adapter; core execution uses status and activity directly.
        """
        if self._cache is None:
            self._cache = DeploymentCache(status=self.status, activity=self.activity)
        return self._cache

    @cache.setter
    def cache(self, value: DeploymentCache) -> None:
        """
        Preserve explicit cache replacement while binding its independent dependencies.
        """
        self._cache = value
        self.status = value.status
        self.activity = value.activity

    @asynccontextmanager
    async def use(self, slug: str) -> AsyncGenerator[RunningDeployment, None]:
        """
        Ensure a remote deployment and lease it for the complete inference operation.

        Lease acquisition happens under the same per-deployment lock as startup,
        so idle cleanup cannot stop a newly prepared deployment before inference.
        Every exit releases the lease and resets its idle countdown. Nested and
        concurrent uses hold independent leases; no HTTP middleware is required.
        Startup cancellation still waits for provisioning and records recoverable
        cloud resources before propagating cancellation.
        """
        async with self.activity.ensure_lock(slug):
            running = await self._prepare_locked(slug)
            self.activity.begin_request(slug)
        try:
            yield running
        finally:
            self.activity.end_request(slug)

    def get_config(self, slug: str) -> DeploymentConfig:
        """
        Resolve a remote deployment or raise the shared 404 service error.
        """
        try:
            return self.catalog.get(slug)
        except LLMManagementError as exc:
            raise DeploymentServiceError(
                status_code=404, detail=f"Deployment '{slug}' not found in config."
            ) from exc

    def get_backend(self, slug: str) -> DeploymentBackend:
        """
        Construct and retain one backend controller per remote deployment slug.

        Controller construction may establish Compute state storage, but does not start
        inference. The membership lock prevents duplicate controllers in this service.
        """
        with self._backend_lock:
            if slug not in self.backends:
                self.backends[slug] = backend_for(self.get_config(slug))
            return self.backends[slug]

    @contextmanager
    def track_requests(self) -> Generator[None, None, None]:
        """
        Hold remote activity leases until the complete caller operation ends.

        Compatibility context for callers using ensure_running() directly. It records
        each deployment once and releases those leases on exit. New inference code
        should use the narrower async use(slug) context instead; it also protects
        direct batch callers without requiring a surrounding request context.
        """
        leases: set[str] = set()
        token = self.request_deployments.set(leases)
        try:
            yield
        finally:
            for slug in leases:
                self.activity.end_request(slug)
            self.request_deployments.reset(token)

    async def ensure_running(
        self, slug: str, *, allow_start: bool = False, replicas: int | None = None
    ) -> RunningDeployment:
        """
        Return configuration and ready state, starting or resizing when permitted.

        Serialize status checks and startup with the per-slug lock. auto_ensure permits
        inference-triggered startup; allow_start explicitly permits management warm-up.
        An explicit replica count also resizes a running managed deployment. Compute
        supports one VM and treats an explicit count of one as normal startup.

        Shield worker startup from caller cancellation, wait for it to finish and refresh
        tracked state before propagating cancellation. A surrounding track_requests()
        context receives an activity lease; callers without one only update activity.
        Raise service errors for invalid counts, forbidden startup or failed readiness.
        """
        async with self.activity.ensure_lock(slug):
            running = await self._prepare_locked(
                slug, allow_start=allow_start, replicas=replicas
            )
            leases = self.request_deployments.get()
            if leases is not None and slug not in leases:
                self.activity.begin_request(slug)
                leases.add(slug)
        return RunningDeployment(running.config, self.cache.state(running.state))

    async def _prepare_locked(
        self, slug: str, *, allow_start: bool = False, replicas: int | None = None
    ) -> RunningDeployment:
        """
        Prepare cloud inference under the lock held by use() or ensure_running().

        This shared implementation knows nothing about request contexts or lease
        ownership. Its wrappers acquire leases before releasing the lifecycle lock.
        Only status observations flow through startup; activity stays in its tracker.
        """
        cfg = self.get_config(slug)
        if replicas is not None:
            if replicas < 1:
                raise DeploymentServiceError(
                    status_code=422, detail="Replicas must be positive."
                )
            if getattr(cfg, "backend", "exoscale_managed") == "exoscale_compute":
                if replicas != 1:
                    raise DeploymentServiceError(
                        status_code=422,
                        detail="Compute deployments support one replica.",
                    )
                replicas = None
        backend = self.get_backend(slug)
        state = await asyncio.to_thread(self.status.ensure, backend)
        if (
            not state.exists
            or state.replicas == 0
            or replicas is not None
            and state.replicas != replicas
        ):
            if not (allow_start or self.auto_ensure):
                raise DeploymentServiceError(
                    status_code=503,
                    detail=f"Deployment '{slug}' is not running. Ensure it first.",
                )
            logger.info("Auto-ensuring deployment %s.", slug)
            self.activity.touch(slug)
            # Let a startup finish under the lock if its initiating request is
            # cancelled, then track the VM so idle/shutdown cleanup can find it.
            kwargs = {"replicas": replicas} if replicas is not None else {}
            startup = asyncio.create_task(
                asyncio.to_thread(backend.create_or_resume, **kwargs)
            )
            try:
                await asyncio.shield(startup)
            except asyncio.CancelledError:
                await startup
                await asyncio.to_thread(self.status.refresh, backend)
                self.activity.touch(slug)
                raise
            except Exception:
                if getattr(cfg, "backend", "exoscale_managed") == "exoscale_compute":
                    try:
                        await asyncio.to_thread(self.status.refresh, backend)
                        self.activity.touch(slug)
                    except Exception:
                        logger.exception(
                            "Failed to refresh Compute resources after startup failure."
                        )
                raise
            state = await asyncio.to_thread(self.status.refresh, backend)
            if not state.exists or state.replicas == 0:
                raise DeploymentServiceError(
                    status_code=503,
                    detail=f"Deployment '{slug}' did not become ready.",
                )
        self.activity.touch(slug)
        return RunningDeployment(config=cfg, state=state)

    def active_deployments(self) -> list[DeploymentSnapshot]:
        """
        Select observed cloud resources still eligible for this application's cleanup.
        """
        return self.activity.active(self.status.all())

    async def idle_scaler(self) -> None:
        """
        Periodically expire unused local resources and scale down idle deployments.

        Use cpu_idle_seconds for local guards and idle_seconds for remote activity, waking
        every check_interval seconds. Busy local resources are skipped. Recheck remote
        activity under the per-slug lock and skip deployments with in-flight leases.

        Managed scale-to-zero retains the deployment; Compute scale-to-zero deletes the
        VM and its access resources. Log cleanup failures and continue later sweeps.
        """
        while True:
            await asyncio.sleep(self.check_interval)
            try:
                await asyncio.to_thread(self.resources.expire, self.cpu_idle_seconds)
                timeout_seconds = self.idle_seconds
                active = self.active_deployments()
                for ds in active:
                    elapsed = time.time() - self.activity.get(ds.slug).last_request_time
                    if elapsed >= timeout_seconds:
                        logger.info(
                            "Deployment %s idle for %.0fs — scaling to zero.",
                            ds.slug,
                            elapsed,
                        )
                        try:
                            async with self.activity.ensure_lock(ds.slug):
                                current = self.activity.get(ds.slug)
                                if (
                                    current.requests_in_flight
                                    or time.time() - current.last_request_time
                                    < timeout_seconds
                                ):
                                    continue
                                backend = self.get_backend(ds.slug)
                                await asyncio.to_thread(backend.scale_to_zero)
                                await asyncio.to_thread(self.status.refresh, backend)
                        except Exception:
                            logger.exception(
                                "Failed to auto-scale %s to zero.", ds.slug
                            )
            except Exception:
                logger.exception("Error in idle scaler loop.")

    def resolve_resource(self, slug: str) -> LocalResource[Any] | DeploymentConfig:
        """
        Resolve an individual management target to its local guard or remote config.

        Registry lookup may construct a cold local owner, but never loads it. Unknown
        names retain the shared remote-deployment 404 error. Group addresses are
        expanded before individual operations reach this method.
        """
        resource = self.resources.get(slug)
        return resource if resource is not None else self.get_config(slug)

    async def warmup_resource(self, name: str) -> int | None:
        """
        Warm a local guard or explicitly ensure a remote deployment.

        Return None for local resources and the ready replica count for remote ones.
        Local loading runs in a worker thread under the owner's guard. Translate loading
        failures to 503 service errors without exposing arbitrary exception messages.
        """
        target = self.resolve_resource(name)
        if isinstance(target, LocalResource):
            await self._warmup_local(target)
            return None
        running = await self.ensure_running(target.slug, allow_start=True)
        return running.state.replicas

    async def _warmup_local(self, resource: LocalResource[Any]) -> None:
        """
        Load a resolved guard with the shared public warm-up error contract.
        """
        try:
            await asyncio.to_thread(resource.warmup)
        except ClassifierUnavailable as exc:
            raise DeploymentServiceError(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning(
                "Local resource %s unavailable: %s", resource.name, type(exc).__name__
            )
            raise DeploymentServiceError(
                status_code=503, detail="Local resource unavailable"
            ) from exc
        return None

    async def scale_to_zero(self, slug: str) -> None:
        """
        Unload a local resource or scale/delete a remote backend under its lifecycle lock.

        Reject deployments with in-flight requests using a 409 service error. Refresh
        cached status after the backend operation; Compute scaling deletes its VM.
        """
        target = self.resolve_resource(slug)
        if isinstance(target, LocalResource):
            try:
                await asyncio.to_thread(target.stop)
            except ClassifierBusy as exc:
                raise DeploymentServiceError(status_code=409, detail=str(exc)) from exc
            return
        async with self.activity.ensure_lock(slug):
            backend = self.get_backend(slug)
            if self.activity.get(slug).requests_in_flight:
                raise DeploymentServiceError(
                    status_code=409, detail="Deployment has requests in flight."
                )
            await asyncio.to_thread(backend.scale_to_zero)
            await asyncio.to_thread(self.status.refresh, backend)

    async def cooldown(self, slug: str) -> float | None:
        """
        Shorten a local or remote resource's idle countdown without stopping it now.

        Reject in-flight work. Return remaining seconds, or None for an absent or stopped
        managed deployment. Existing Compute VMs remain eligible even before a tunnel is
        ready. New requests reset the activity timestamp and cancel the shortened window.
        """
        target = self.resolve_resource(slug)
        if isinstance(target, LocalResource):
            try:
                return await asyncio.to_thread(target.cooldown, self.cpu_idle_seconds)
            except ClassifierBusy as exc:
                raise DeploymentServiceError(status_code=409, detail=str(exc)) from exc
        async with self.activity.ensure_lock(slug):
            backend = self.get_backend(slug)
            state = await asyncio.to_thread(self.status.ensure, backend)
            if self.activity.get(slug).requests_in_flight:
                raise DeploymentServiceError(
                    status_code=409, detail="Deployment has requests in flight."
                )
            if not state.exists or (
                state.replicas == 0 and state.backend != "exoscale_compute"
            ):
                return None
            return self.activity.cooldown(slug, self.idle_seconds)

    async def close(self) -> None:
        """
        Attempt local release and scale down remote deployments tracked as active.

        Local cleanup skips guards still leased by workers, so shutdown does not unload
        models being used. Remote cleanup joins status observations with application activity rather than
        all configured deployments. Log failures and continue with the other resources;
        this method does not cancel the idle-scaler task itself.
        """
        try:
            await asyncio.to_thread(self.resources.close)
        except Exception:
            logger.exception("Failed to close local resources on shutdown.")
        for state in self.active_deployments():
            try:
                backend = self.get_backend(state.slug)
                await asyncio.to_thread(backend.scale_to_zero)
            except Exception:
                logger.exception("Failed to scale %s to zero on shutdown.", state.slug)

    async def ensure_deployment(
        self, slug: str, replicas: int | None = None
    ) -> WarmUpResponse:
        """
        Warm one configured resource and report the resulting management action.

        Local guards return a warmed outcome with no replica count and reject explicit
        replicas. Remote outcomes distinguish creation, resumption, resizing and an
        already-running deployment using its prior cached status.
        """
        target = self.resolve_resource(slug)
        if isinstance(target, LocalResource):
            if replicas is not None:
                raise DeploymentServiceError(
                    status_code=422, detail="Local resources do not support replicas."
                )
            await self._warmup_local(target)
            return WarmUpResponse(slug=slug, action="warmed", replicas=None)
        backend = self.get_backend(slug)
        previous = await asyncio.to_thread(self.status.ensure, backend)
        running = await self.ensure_running(slug, allow_start=True, replicas=replicas)
        actual = running.state.replicas
        action = (
            "scaled"
            if previous.replicas > 0
            and replicas is not None
            and previous.replicas != replicas
            else "already_running"
            if previous.replicas > 0
            else "resumed"
            if previous.exists
            else "created"
        )
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
            group = self.catalog.get_group(slug)
        except LLMManagementError as exc:
            raise DeploymentServiceError(
                status_code=404, detail=f"Deployment group '{slug}' not found"
            ) from exc
        return list(
            await asyncio.gather(*(operation(member) for member in group.deployments))
        )

    async def ensure_group(self, slug: str) -> GroupWarmUpResponse:
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
                replicas = await self.warmup_resource(member)
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
        self, slug: str, action: Literal["cool-down", "scale-to-zero"]
    ) -> GroupActionOutcome:
        """
        Apply a lifecycle action concurrently, preserving per-member failures.
        """

        async def change_member(member: str) -> GroupActionMemberOutcome:
            try:
                if action == "cool-down":
                    remaining = await self.cooldown(member)
                    return GroupActionMemberOutcome(
                        slug=member,
                        success=True,
                        action="already_stopped"
                        if remaining is None
                        else "cooling_down",
                        seconds_until_scale_to_zero=remaining,
                    )
                await self.scale_to_zero(member)
                return GroupActionMemberOutcome(
                    slug=member, success=True, action="scaled_to_zero"
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

    async def ensure_local(self, name: str) -> ResourceStatus:
        """
        Warm a registered local guard and return its readiness snapshot.

        Unknown local names raise a 404 service error; loading failures follow the shared
        warm-up error translation. This operation never falls back to a remote deployment.
        """
        resource = self.resources.get(name)
        if resource is None:
            raise DeploymentServiceError(status_code=404, detail="Unknown local model")
        await self.warmup_resource(name)
        return resource.status()
