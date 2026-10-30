"""
Cached observations from cloud backends, independent of application activity.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from .types import DeploymentSnapshot

if TYPE_CHECKING:
    from .backends import DeploymentBackend


class DeploymentStatusCache:
    """
    Retain cloud status and connection details until their observation TTL expires.

    This cache has no lease counts, idle timers or lifecycle locks. Reads and writes
    copy records so callers cannot mutate cached observations through a returned value.
    Querying a backend observes existing resources; it never starts inference.
    """

    def __init__(self, cache_ttl_seconds: float = 60.0) -> None:
        """
        Create empty status storage without constructing backends or querying cloud APIs.
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self._lock = threading.Lock()
        self._snapshots: dict[str, DeploymentSnapshot] = {}

    def get(self, slug: str) -> DeploymentSnapshot | None:
        """
        Return the last observation, or None when absent or explicitly evicted.
        """
        with self._lock:
            snapshot = self._snapshots.get(slug)
            return snapshot.model_copy() if snapshot is not None else None

    def set(self, snapshot: DeploymentSnapshot) -> None:
        """
        Store a detached copy so caller mutations cannot change cached observations.
        """
        value = snapshot.model_copy()
        with self._lock:
            self._snapshots[snapshot.slug] = value

    def all(self) -> list[DeploymentSnapshot]:
        """
        Return detached observations without constructing configured backends.
        """
        with self._lock:
            return [snapshot.model_copy() for snapshot in self._snapshots.values()]

    def remove(self, slug: str) -> None:
        """
        Forget one observation; application leases and idle timers are unaffected.
        """
        with self._lock:
            self._snapshots.pop(slug, None)

    def is_stale(self, slug: str) -> bool:
        """
        Check observation age without consulting inference activity.
        """
        snapshot = self.get(slug)
        return (
            snapshot is None
            or time.time() - snapshot.last_refreshed > self.cache_ttl_seconds
        )

    def refresh(self, backend: DeploymentBackend) -> DeploymentSnapshot:
        """
        Query a backend and replace only its observed status and connection details.
        """
        config = backend.config
        observed = backend.query_status()
        snapshot = DeploymentSnapshot(
            slug=config.slug,
            backend=getattr(config, "backend", "exoscale_managed"),
            exists=observed.exists,
            replicas=observed.replicas,
            deployment_url=observed.deployment_url,
            api_key=observed.api_key,
            last_refreshed=time.time(),
        )
        self.set(snapshot)
        return snapshot

    def ensure(self, backend: DeploymentBackend) -> DeploymentSnapshot:
        """
        Return a fresh observation, refreshing after expiry or eviction.
        """
        snapshot = self.get(backend.config.slug)
        if (
            snapshot is None
            or time.time() - snapshot.last_refreshed > self.cache_ttl_seconds
        ):
            return self.refresh(backend)
        return snapshot
