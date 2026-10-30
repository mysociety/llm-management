"""
Compatibility adapter for independently owned cloud status and activity.

Refreshing a snapshot cannot reset idle timers or replace live request leases.
Combined DeploymentState values remain available for existing callers.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .activity import DeploymentActivityTracker
from .status import DeploymentStatusCache
from .types import DeploymentActivity, DeploymentSnapshot, DeploymentState

if TYPE_CHECKING:
    from .backends import DeploymentBackend


class DeploymentCache:
    """
    Cache cloud observations while retaining the legacy combined-state interface.

    Activity lives in a separate tracker, shared with the deployment service.
    Cache reads are detached snapshots; changes go through explicit cache or
    activity operations. The cache never loads models or provisions deployments.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 60.0,
        *,
        status: DeploymentStatusCache | None = None,
        activity: DeploymentActivityTracker | None = None,
    ) -> None:
        """
        Adapt separately owned status and activity to the legacy cache interface.

        Standalone legacy construction supplies defaults. Application assembly passes
        both dependencies explicitly; normal execution uses those dependencies directly.
        """
        self.status = (
            status if status is not None else DeploymentStatusCache(cache_ttl_seconds)
        )
        self.activity = (
            activity if activity is not None else DeploymentActivityTracker()
        )

    @property
    def cache_ttl_seconds(self) -> float:
        """
        Preserve access to the status observation TTL.
        """
        return self.status.cache_ttl_seconds

    @cache_ttl_seconds.setter
    def cache_ttl_seconds(self, value: float) -> None:
        """
        Update status expiry without changing activity ownership.
        """
        self.status.cache_ttl_seconds = value

    def ensure_lock(self, slug: str) -> asyncio.Lock:
        """
        Compatibility access to the activity tracker's lifecycle lock.
        """
        return self.activity.ensure_lock(slug)

    def get(self, slug: str) -> DeploymentState | None:
        """
        Combine cached status with current activity without exposing mutable storage.
        """
        snapshot = self.status.get(slug)
        return self.state(snapshot) if snapshot is not None else None

    def state(self, snapshot: DeploymentSnapshot) -> DeploymentState:
        """
        Build the legacy flat view from independent status and activity records.
        """
        activity = self.activity.get(snapshot.slug)
        return DeploymentState(
            slug=snapshot.slug,
            backend=snapshot.backend,
            exists=snapshot.exists,
            replicas=snapshot.replicas,
            deployment_url=snapshot.deployment_url,
            api_key=snapshot.api_key,
            last_refreshed=snapshot.last_refreshed,
            last_request_time=activity.last_request_time,
            requests_in_flight=activity.requests_in_flight,
        )

    def set(self, state: DeploymentState) -> None:
        """
        Explicitly seed both status and activity from a legacy combined record.

        Normal cloud refreshes use refresh(), which never replaces activity.
        """
        self.status.set(state)
        self.activity.set(
            state.slug,
            DeploymentActivity(
                last_request_time=state.last_request_time,
                requests_in_flight=state.requests_in_flight,
            ),
        )

    def touch(self, slug: str) -> None:
        """
        Compatibility entry point for recording application activity.
        """
        if self.get(slug) is not None:
            self.activity.touch(slug)

    def cooldown(self, slug: str, timeout_seconds: float) -> float:
        """
        Compatibility entry point for shortening an application's idle countdown.
        """
        return self.activity.cooldown(slug, timeout_seconds)

    def is_stale(self, slug: str) -> bool:
        """
        Check only the cloud observation's age, independently of recent inference.
        """
        return self.status.is_stale(slug)

    def begin_request(self, slug: str) -> None:
        """
        Compatibility entry point for acquiring an application lease.
        """
        self.activity.begin_request(slug)

    def end_request(self, slug: str) -> None:
        """
        Compatibility entry point for releasing an application lease.
        """
        self.activity.end_request(slug)

    def all_active(self) -> list[DeploymentState]:
        """
        Find used running deployments, including Compute VMs without a ready tunnel.
        """
        return [
            self.state(snapshot) for snapshot in self.activity.active(self.status.all())
        ]

    def remove(self, slug: str) -> None:
        """
        Evict cloud status without discarding application activity or live leases.
        """
        self.status.remove(slug)

    def refresh(self, backend: DeploymentBackend) -> DeploymentState:
        """
        Replace the cloud observation without rewriting activity storage.
        """
        return self.state(self.status.refresh(backend))

    def ensure(self, backend: DeploymentBackend) -> DeploymentState:
        """
        Return fresh cached status, querying the provider after eviction or expiry.
        """
        return self.state(self.status.ensure(backend))
