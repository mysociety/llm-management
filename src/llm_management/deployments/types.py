"""
Results of deployment lifecycle operations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from .config import DeploymentConfig


class EnsureOutcome(BaseModel):
    slug: str
    action: str
    replicas: int | None


class GroupMemberOutcome(BaseModel):
    slug: str
    success: bool
    replicas: int | None = None
    error: str | None = None


class GroupOutcome(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupMemberOutcome]


class DeploymentState(BaseModel):
    """
    Cached state for a single deployment, including connection details
    and the timestamp of the last forwarded request.
    """

    slug: str
    backend: str = "exoscale_managed"
    exists: bool = False
    replicas: int = 0
    deployment_url: str = ""
    api_key: str = ""
    last_request_time: float = 0.0
    last_refreshed: float = 0.0
    requests_in_flight: int = 0


class RunningDeployment(NamedTuple):
    """
    Configuration and live state of a running deployment.
    """

    config: DeploymentConfig
    state: DeploymentState


class DeploymentAccess(NamedTuple):
    """
    Deployment operations supplied to HTTP-independent feature pipelines.
    """

    get_config: Callable[[str], DeploymentConfig]
    ensure_running: Callable[[str], Awaitable[RunningDeployment]]
    touch: Callable[[str], None]


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
