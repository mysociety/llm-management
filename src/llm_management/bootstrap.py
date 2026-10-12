"""Assemble application dependencies and register cold model factories.

Loading a catalog is a pure configuration operation. Assembly establishes runtime
ownership without loading model weights or creating cloud deployments.
"""

from .deployments.catalog import DeploymentCatalog, LocalDeploymentConfig, get_catalog
from .deployments.cache import DeploymentCache
from .deployments.service import DeploymentService
from .inference.resources import LocalResource, ResourceRegistry
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
from collections.abc import Callable

if TYPE_CHECKING:
    from .sanitization.presidio import PresidioSanitizer
    from .foi.backends import FOIResources
from .settings import settings


def register_configured_resources(
    catalog: DeploymentCatalog, registry: ResourceRegistry
):
    """Register validated catalog entries as cold factories without owner imports."""
    from importlib import import_module

    owners = {
        "modernbert_classifier": ("foi.backends", "question_classifier"),
        "modernbert_head": ("foi.backends", "question_classification_head"),
        "topic_tokenizer": ("foi.topic_classification", "tokenizer_resource"),
        "presidio": ("sanitization", "configured_presidio"),
    }
    controllers = {}

    def create(config: "LocalDeploymentConfig"):
        module_name, factory_name = owners[config.loader]
        factory = getattr(import_module(f"llm_management.{module_name}"), factory_name)
        owner = factory(config, catalog=catalog, registry=registry)
        controllers[config.slug] = owner
        return owner if isinstance(owner, LocalResource) else owner.resource

    providers = {}

    def provider(slug):
        registry.get(slug)
        return controllers[slug]

    for config in catalog.local.deployment:
        registry.register_factory(config.slug, lambda config=config: create(config))
        providers[config.slug] = lambda slug=config.slug: provider(slug)
    return providers


@dataclass
class Application:
    """Explicit runtime dependencies shared by HTTP and direct batch callers."""

    deployments: DeploymentService
    sanitizer: Callable[[], "PresidioSanitizer"]
    foi: "FOIResources"


def assemble_application(
    catalog: DeploymentCatalog | None = None, registry: ResourceRegistry | None = None
) -> Application:
    from .foi.backends import FOIResources

    catalog = catalog if catalog is not None else get_catalog()
    registry = registry if registry is not None else ResourceRegistry()
    providers = register_configured_resources(catalog, registry)
    service = DeploymentService(
        catalog,
        DeploymentCache(),
        registry,
        cpu_idle_seconds=settings.cpu_idle_timeout_minutes * 60,
    )

    def sanitizer() -> "PresidioSanitizer":
        return providers[catalog.require_sanitizer().slug]()

    foi = FOIResources(
        catalog=catalog,
        sanitizer=sanitizer,
        classifier=lambda: providers[catalog.require_foi().classifier](),
        head=lambda: providers[catalog.require_foi().head](),
        tokenizer=lambda: providers[catalog.require_foi().tokenizer](),
    )
    return Application(deployments=service, sanitizer=sanitizer, foi=foi)


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
