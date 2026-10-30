"""
Results of deployment lifecycle operations.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
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


class LifecycleCommand(StrEnum):
    """
    Management commands supported for a resource group.
    """

    COOL_DOWN = "cool-down"
    STOP = "scale-to-zero"


class LifecycleAction(StrEnum):
    """
    Outcome of stopping a resource or shortening its idle timer.
    """

    ALREADY_STOPPED = "already_stopped"
    COOLING_DOWN = "cooling_down"
    SCALED_TO_ZERO = "scaled_to_zero"


class GroupActionMemberOutcome(BaseModel):
    slug: str
    success: bool
    action: LifecycleAction | None = None
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
    without depending on management or caching.
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
