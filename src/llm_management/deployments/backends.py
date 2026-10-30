"""
Shared deployment lifecycle interface and backend selection.
"""

from __future__ import annotations

from typing import Protocol

from .config import ComputeDeploymentConfig, DeploymentConfig, ExoscaleDeploymentConfig
from .types import ConnectionInfo, DeploymentQueryResult


class DeploymentBackend(Protocol):
    @property
    def config(self) -> DeploymentConfig: ...

    def create_deployment(
        self, refresh_model: bool = False, *, replicas: int | None = None
    ) -> None: ...
    def create_or_resume(self, *, replicas: int | None = None) -> None: ...
    def resume_from_zero(self, *, replicas: int | None = None) -> None: ...
    def scale_to_zero(self) -> None: ...
    def delete_deployment(self) -> None: ...
    def query_status(self) -> DeploymentQueryResult: ...
    def connection_info(self) -> ConnectionInfo: ...
    def logs(self, tail: int) -> str: ...
    def test_basic_deployment(self) -> bool | None: ...
    def test_instruct_deployment(self) -> bool | None: ...


def backend_for(config: DeploymentConfig) -> DeploymentBackend:
    """
    Construct a backend; cache ownership belongs to its service or CLI caller.
    """
    match config:
        case ComputeDeploymentConfig():
            from .compute import ComputeDeployment

            return ComputeDeployment(config)
        case ExoscaleDeploymentConfig():
            from .managed import ManagedDeployment

            return ManagedDeployment(config)
        case _:
            raise TypeError("Unsupported deployment configuration")
