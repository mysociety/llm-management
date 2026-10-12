"""
Internal cache for deployment state and last-request timestamps.

Avoids constantly querying Exoscale management APIs by keeping a local
record of known deployments and the last time a request was forwarded.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

from .types import DeploymentState

if TYPE_CHECKING:
    from .backends import DeploymentBackend


class DeploymentCache:
    """
    Thread-safe in-memory cache of deployment states. Entries older
    than cache_ttl_seconds are considered stale and will be refreshed
    on the next access.
    """

    def __init__(self, cache_ttl_seconds: float = 60.0) -> None:
        self._lock = threading.Lock()
        self._deployments: dict[str, DeploymentState] = {}
        self._ensure_locks: dict[str, asyncio.Lock] = {}
        self.cache_ttl_seconds = cache_ttl_seconds

    def ensure_lock(self, slug: str) -> asyncio.Lock:
        """
        Return a per-slug asyncio.Lock, creating one if it doesn't exist.
        Ensures only one caller at a time triggers create_or_resume for
        a given deployment.
        """
        with self._lock:
            if slug not in self._ensure_locks:
                self._ensure_locks[slug] = asyncio.Lock()
            return self._ensure_locks[slug]

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

    def cooldown(self, slug: str, timeout_seconds: float) -> float:
        """
        Backdate activity to leave at most 60 seconds on the idle timer.
        """
        with self._lock:
            state = self._deployments[slug]
            now = time.time()
            latest = now - timeout_seconds + 60
            state.last_request_time = (
                min(state.last_request_time, latest)
                if state.last_request_time > 0
                else latest
            )
            return max(0.0, timeout_seconds - (now - state.last_request_time))

    def is_stale(self, slug: str) -> bool:
        """
        Return True if the entry is missing or older than the cache TTL.
        """
        with self._lock:
            entry = self._deployments.get(slug)
            if entry is None:
                return True
            return (time.time() - entry.last_refreshed) > self.cache_ttl_seconds

    def begin_request(self, slug: str) -> None:
        with self._lock:
            self._deployments[slug].requests_in_flight += 1
            self._deployments[slug].last_request_time = time.time()

    def end_request(self, slug: str) -> None:
        with self._lock:
            state = self._deployments.get(slug)
            if state:
                state.requests_in_flight = max(0, state.requests_in_flight - 1)
                state.last_request_time = time.time()

    def all_active(self) -> list[DeploymentState]:
        """
        Return all deployments that have received at least one request
        and currently have replicas > 0.
        """
        with self._lock:
            return [
                ds
                for ds in self._deployments.values()
                if ds.last_request_time > 0
                and (ds.replicas > 0 or ds.backend == "exoscale_compute" and ds.exists)
            ]

    def remove(self, slug: str) -> None:
        """
        Remove a deployment from the cache entirely.
        """
        with self._lock:
            self._deployments.pop(slug, None)

    def refresh(self, backend: DeploymentBackend) -> DeploymentState:
        """
        Query the Exoscale API for the current state of a deployment and
        update the cache entry. Preserves the existing last_request_time
        so the idle-scaler timer is not reset.
        """
        cfg = backend.config
        status = backend.query_status()
        with self._lock:
            existing = self._deployments.get(cfg.slug)
            last_request_time = existing.last_request_time if existing else 0.0
            requests_in_flight = existing.requests_in_flight if existing else 0
            state = DeploymentState(
                slug=cfg.slug,
                backend=getattr(cfg, "backend", "exoscale_managed"),
                exists=status.exists,
                replicas=status.replicas,
                deployment_url=status.deployment_url,
                api_key=status.api_key,
                last_request_time=last_request_time,
                last_refreshed=time.time(),
                requests_in_flight=requests_in_flight,
            )
            self._deployments[cfg.slug] = state
            return state

    def ensure(self, backend: DeploymentBackend) -> DeploymentState:
        """
        Return the cached deployment state, refreshing from the Exoscale
        API if the entry is missing or has exceeded its TTL.
        """
        if self.is_stale(backend.config.slug):
            return self.refresh(backend)
        return self.get(backend.config.slug)  # type: ignore[return-value]
