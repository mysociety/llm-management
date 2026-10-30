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
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from ..inference.errors import ClassifierBusy, ClassifierUnavailable
from ..inference.resources import LocalResource, ResourceRegistry
from .activity import DeploymentActivityTracker
from .backends import DeploymentBackend, backend_for
from .catalog import DeploymentCatalog
from .config import DeploymentConfig
from .errors import DeploymentServiceError, LLMManagementError
from .status import DeploymentStatusCache
from .types import (
    DeploymentSnapshot,
    RunningDeployment,
)

logger = logging.getLogger(__name__)


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
        *,
        resources: ResourceRegistry | None = None,
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
        and activity independently.
        """
        self.catalog = catalog
        self.backends: dict[str, DeploymentBackend] = {}
        self._backend_lock = threading.Lock()
        self.status = status if status is not None else DeploymentStatusCache()
        self.activity = (
            activity if activity is not None else DeploymentActivityTracker()
        )
        self.resources = resources if resources is not None else ResourceRegistry()
        self.auto_ensure = auto_ensure
        self.idle_seconds = idle_seconds
        self.cpu_idle_seconds = cpu_idle_seconds
        self.check_interval = check_interval

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

    async def prepare(
        self, slug: str, *, allow_start: bool = False, replicas: int | None = None
    ) -> RunningDeployment:
        """
        Prepare a remote deployment under its lifecycle lock without leasing inference.

        Management warm-up permits startup explicitly. Inference callers must use
        use() so their lease starts before the lifecycle lock is released.
        """
        async with self.activity.ensure_lock(slug):
            return await self._prepare_locked(
                slug, allow_start=allow_start, replicas=replicas
            )

    async def _prepare_locked(
        self, slug: str, *, allow_start: bool = False, replicas: int | None = None
    ) -> RunningDeployment:
        """
        Prepare cloud inference under the lock held by use() or prepare().

        This implementation prepares status and connection details without owning
        a lease. use() acquires its inference lease before releasing the lock;
        prepare() returns directly for management. Activity stays in its tracker.
        """
        cfg = self.get_config(slug)
        if cfg.backend == "exoscale_managed":
            try:
                cfg.replica_target(replicas)
            except ValueError as exc:
                raise DeploymentServiceError(status_code=422, detail=str(exc)) from exc
        if replicas is not None:
            if replicas < 1:
                raise DeploymentServiceError(
                    status_code=422, detail="Replicas must be positive."
                )
            if cfg.backend == "exoscale_compute":
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
                if cfg.backend == "exoscale_compute":
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

    async def warmup_local(self, resource: LocalResource[Any]) -> None:
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
