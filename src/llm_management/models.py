"""Compatibility imports for deployment data and errors.

New callers import these names from their owning modules under deployments/. Cloud operations
belong to backend objects, rather than configuration models.
"""

from __future__ import annotations

from .deployments.client import get_client
from .deployments.config import (
    ComputeDeploymentConfig,
    DeploymentConfig,
    ExoscaleDeploymentConfig,
    ExoscaleDeployments,
)
from .deployments.errors import DeploymentNotFoundError, LLMManagementError
from .deployments.types import DeploymentQueryResult

__all__ = [
    "ComputeDeploymentConfig",
    "DeploymentConfig",
    "ExoscaleDeploymentConfig",
    "ExoscaleDeployments",
    "DeploymentNotFoundError",
    "LLMManagementError",
    "DeploymentQueryResult",
    "get_client",
]
