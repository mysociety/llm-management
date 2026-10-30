"""
Results of deployment lifecycle operations.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from .config import DeploymentConfig


class WarmUpAction(StrEnum):
    """
    Outcome of warming a local resource or remote deployment.
    """

    WARMED = "warmed"
    SCALED = "scaled"
    ALREADY_RUNNING = "already_running"
    RESUMED = "resumed"
    CREATED = "created"


class WarmUpResponse(BaseModel):
    slug: str
    action: WarmUpAction
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


class GroupActionMemberOutcome(BaseModel):
    slug: str
    success: bool
    action: str | None = None
    seconds_until_scale_to_zero: float | None = None
    error: str | None = None


class GroupActionOutcome(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupActionMemberOutcome]


class DeploymentActivity(BaseModel):
    """
    Application activity, independent of observations from the cloud provider.
    """

    last_request_time: float = 0.0
    requests_in_flight: int = 0


class DeploymentSnapshot(BaseModel):
    """
    Last observed cloud status and connection details for one deployment.

    This record contains no request counts, idle timers or coordination locks.
    """

    slug: str
    backend: str = "exoscale_managed"
    exists: bool = False
    replicas: int = 0
    deployment_url: str = ""
    api_key: str = ""
    last_refreshed: float = 0.0


class RunningDeployment(NamedTuple):
    """
    Configuration and observed connection/status for a prepared deployment.
    """

    config: DeploymentConfig
    state: DeploymentSnapshot


class DeploymentInference(Protocol):
    """
    The remote operations a feature needs: validate configuration and lease inference.

    DeploymentService implements this directly. Features can accept small test doubles
    or the legacy DeploymentAccess adapter without depending on management or caching.
    """

    def get_config(self, slug: str, /) -> DeploymentConfig:
        """
        Resolve configuration before input preparation or checkpoint validation.
        """
        ...

    def use(self, slug: str) -> AbstractAsyncContextManager[RunningDeployment]:
        """
        Prepare a deployment and protect it until the inference block exits.
        """
        ...


class DeploymentConfigLookup(Protocol):
    """
    A configuration callback stored as data by the legacy NamedTuple adapter.

    A callable protocol lets type checkers distinguish this field from a method
    bound to the adapter itself when checking DeploymentInference compatibility.
    """

    def __call__(self, slug: str, /) -> DeploymentConfig:
        """
        Resolve one configured deployment through the supplied callback.
        """
        ...


class DeploymentAccess(NamedTuple):
    """
    Legacy callback adapter implementing the feature inference protocol.

    Retained for callers constructing dependencies from individual functions.
    Normal application assembly supplies DeploymentService directly.
    """

    get_config: DeploymentConfigLookup
    ensure_running: Callable[[str], Awaitable[RunningDeployment]]
    touch: Callable[[str], None]
    lease: Callable[[str], AbstractAsyncContextManager[RunningDeployment]] | None = None

    @asynccontextmanager
    async def use(self, slug: str) -> AsyncGenerator[RunningDeployment, None]:
        """
        Hold the supplied application's lease throughout remote inference.

        Older three-callback constructions retain their ensure/touch behaviour.
        An optional lease callback forwards explicit lifetime protection. New
        integrations should pass the service directly rather than assembling callbacks.
        """
        if self.lease is not None:
            async with self.lease(slug) as running:
                yield running
            return
        try:
            yield await self.ensure_running(slug)
        finally:
            self.touch(slug)


class DeploymentQueryResult(BaseModel):
    """
    Result of querying Exoscale for a deployment's current state.
    """

    exists: bool
    replicas: int
    deployment_url: str
    api_key: str


class ConnectionInfo(BaseModel):
    """
    Connection details shared by deployment backends and CLI presentation.
    """

    url: str
    api_key: str | None
    ssh_tunnel: list[str] | None = None
