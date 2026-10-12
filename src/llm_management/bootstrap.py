"""Assemble application dependencies and register cold model factories.

Loading a catalog is a pure configuration operation. Assembly establishes runtime
ownership without loading model weights or creating cloud deployments.
"""

from .deployments.catalog import DeploymentCatalog, get_catalog
from .deployments.cache import DeploymentCache
from .deployments.service import DeploymentService
from .inference.resources import ResourceRegistry
from dataclasses import dataclass
from functools import lru_cache
from .settings import settings


@dataclass
class Application:
    """Explicit runtime dependencies shared by HTTP and direct batch callers."""

    deployments: DeploymentService


def assemble_application(
    catalog: DeploymentCatalog | None = None, registry: ResourceRegistry | None = None
) -> Application:
    catalog = catalog if catalog is not None else get_catalog()
    registry = registry if registry is not None else ResourceRegistry()
    service = DeploymentService(
        catalog,
        DeploymentCache(),
        registry,
        cpu_idle_seconds=settings.cpu_idle_timeout_minutes * 60,
    )
    return Application(deployments=service)


def assemble_service(catalog=None, registry=None) -> DeploymentService:
    """Assemble just the deployment lifecycle for management-only callers."""
    return assemble_application(catalog, registry).deployments


@lru_cache(maxsize=1)
def default_application() -> Application:
    """Process-local default for public convenience calls outside an API app.

    Batch callers wanting explicit isolation should use assemble_application().
    Controllers and loaded weights belong to this assembly's registry.
    """
    return assemble_application()
