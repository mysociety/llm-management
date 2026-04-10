"""
Internal cache for deployment state and last-request timestamps.

Avoids constantly querying Exoscale management APIs by keeping a local
record of known deployments and the last time a request was forwarded.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .models import ExoscaleDeploymentConfig


class DeploymentState(BaseModel):
    """
    Cached state for a single deployment, including connection details
    and the timestamp of the last forwarded request.
    """

    slug: str
    exists: bool = False
    replicas: int = 0
    deployment_url: str = ""
    api_key: str = ""
    last_request_time: float = 0.0
    last_refreshed: float = 0.0


class DeploymentCache:
    """
    Thread-safe in-memory cache of deployment states. Entries older
    than cache_ttl_seconds are considered stale and will be refreshed
    on the next access.
    """

    def __init__(self, cache_ttl_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._deployments: dict[str, DeploymentState] = {}
        self.cache_ttl_seconds = cache_ttl_seconds

    def get(self, slug: str) -> DeploymentState | None:
        """
        Return the cached state for a deployment, or None if not cached.
        """
        with self._lock:
            return self._deployments.get(slug)

    def set(self, state: DeploymentState) -> None:
        """
        Insert or replace the cached state for a deployment.
        """
        with self._lock:
            self._deployments[state.slug] = state

    def touch(self, slug: str) -> None:
        """
        Record that a request was just forwarded to this deployment,
        resetting the idle-scaler countdown.
        """
        with self._lock:
            if slug in self._deployments:
                self._deployments[slug].last_request_time = time.time()

    def is_stale(self, slug: str) -> bool:
        """
        Return True if the entry is missing or older than the cache TTL.
        """
        with self._lock:
            entry = self._deployments.get(slug)
            if entry is None:
                return True
            return (time.time() - entry.last_refreshed) > self.cache_ttl_seconds

    def all_active(self) -> list[DeploymentState]:
        """
        Return all deployments that have received at least one request
        and currently have replicas > 0.
        """
        with self._lock:
            return [
                ds
                for ds in self._deployments.values()
                if ds.last_request_time > 0 and ds.replicas > 0
            ]

    def remove(self, slug: str) -> None:
        """
        Remove a deployment from the cache entirely.
        """
        with self._lock:
            self._deployments.pop(slug, None)

    def refresh(self, cfg: ExoscaleDeploymentConfig) -> DeploymentState:
        """
        Query the Exoscale API for the current state of a deployment and
        update the cache entry. Preserves the existing last_request_time
        so the idle-scaler timer is not reset.
        """
        status = cfg.query_status()
        with self._lock:
            existing = self._deployments.get(cfg.slug)
            last_request_time = existing.last_request_time if existing else 0.0
        state = DeploymentState(
            slug=cfg.slug,
            exists=status.exists,
            replicas=status.replicas,
            deployment_url=status.deployment_url,
            api_key=status.api_key,
            last_request_time=last_request_time,
            last_refreshed=time.time(),
        )
        self.set(state)
        return state

    def ensure(self, cfg: ExoscaleDeploymentConfig) -> DeploymentState:
        """
        Return the cached deployment state, refreshing from the Exoscale
        API if the entry is missing or has exceeded its TTL.
        """
        if self.is_stale(cfg.slug):
            return self.refresh(cfg)
        return self.get(cfg.slug)  # type: ignore[return-value]


# Singleton instance
cache = DeploymentCache()
