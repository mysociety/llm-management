"""
Application-owned remote leases, idle timestamps and lifecycle coordination.

Cloud status refreshes never write this state. Each application has its own tracker,
including when several applications refer to the same configured deployment.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable

from .types import DeploymentActivity, DeploymentSnapshot


class DeploymentActivityTracker:
    """
    Coordinate remote use independently of cached cloud observations.

    A thread lock protects short activity updates. Per-deployment asyncio locks
    serialize startup and scaling in the application's event loop. Request leases
    protect inference after the startup lock is released; ending a lease starts a
    fresh idle countdown even when inference failed or was cancelled.
    """

    def __init__(self) -> None:
        """
        Create empty activity and coordination state without starting work.
        """
        self._lock = threading.Lock()
        self._activity: dict[str, DeploymentActivity] = {}
        self._ensure_locks: dict[str, asyncio.Lock] = {}

    def ensure_lock(self, slug: str) -> asyncio.Lock:
        """
        Return the shared startup/scaling lock for one deployment.
        """
        with self._lock:
            return self._ensure_locks.setdefault(slug, asyncio.Lock())

    def get(self, slug: str) -> DeploymentActivity:
        """
        Return a detached activity snapshot, defaulting to an unused deployment.
        """
        with self._lock:
            return self._activity.get(slug, DeploymentActivity()).model_copy()

    def set(self, slug: str, activity: DeploymentActivity) -> None:
        """
        Seed an activity record explicitly without changing observed cloud status.
        """
        with self._lock:
            self._activity[slug] = activity.model_copy()

    def touch(self, slug: str) -> None:
        """
        Reset the idle countdown without changing the number of active leases.
        """
        with self._lock:
            self._activity.setdefault(
                slug, DeploymentActivity()
            ).last_request_time = time.time()

    def begin_request(self, slug: str) -> None:
        """
        Acquire a lease while the caller holds the deployment's startup lock.
        """
        with self._lock:
            activity = self._activity.setdefault(slug, DeploymentActivity())
            activity.requests_in_flight += 1
            activity.last_request_time = time.time()

    def end_request(self, slug: str) -> None:
        """
        Release a lease and reset the idle countdown on every exit path.
        """
        with self._lock:
            activity = self._activity.setdefault(slug, DeploymentActivity())
            activity.requests_in_flight = max(0, activity.requests_in_flight - 1)
            activity.last_request_time = time.time()

    def cooldown(self, slug: str, timeout_seconds: float) -> float:
        """
        Shorten the remaining countdown to at most sixty seconds.
        """
        with self._lock:
            activity = self._activity.setdefault(slug, DeploymentActivity())
            now = time.time()
            latest = now - timeout_seconds + 60
            activity.last_request_time = (
                min(activity.last_request_time, latest)
                if activity.last_request_time > 0
                else latest
            )
            return max(0.0, timeout_seconds - (now - activity.last_request_time))

    def active(
        self, snapshots: Iterable[DeploymentSnapshot]
    ) -> list[DeploymentSnapshot]:
        """
        Select used remote resources that still require idle or shutdown cleanup.

        Compute VMs count even without a ready inference tunnel. Status is supplied
        by the caller; the tracker never queries or owns the cloud status cache.
        """
        return [
            snapshot
            for snapshot in snapshots
            if self.get(snapshot.slug).last_request_time > 0
            and (
                snapshot.replicas > 0
                or snapshot.backend == "exoscale_compute"
                and snapshot.exists
            )
        ]

    def remove(self, slug: str) -> None:
        """
        Forget inactive activity; never discard a live lease or its coordination lock.
        """
        with self._lock:
            activity = self._activity.get(slug)
            if activity is not None and not activity.requests_in_flight:
                self._activity.pop(slug, None)
