"""
Assemble application dependencies and register cold model factories.

Loading a catalog is a pure configuration operation. Assembly establishes runtime
ownership without loading model weights or creating cloud deployments.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict

from .deployments import resource_types
from .deployments.activity import DeploymentActivityTracker
from .deployments.catalog import DeploymentCatalog, get_catalog
from .deployments.service import DeploymentService
from .deployments.status import DeploymentStatusCache
from .inference.providers import ResourceProviders
from .inference.providers import (
    register_configured_resources as register_configured_resources,
)
from .inference.resources import ResourceRegistry
from .settings import settings

if TYPE_CHECKING:
    from .automatic.pipeline import AutomaticClassifier
    from .foi.backends import FOIResources
    from .foi.email_finder.service import FOIFinder
    from .sanitization.presidio import PresidioSanitizer
    from .sar.pipeline import SARClassifier
    from .web.crawl import CrawlPolicy


class DeploymentAssembly(NamedTuple):
    """
    Cold deployment dependencies shared by management and feature assembly.

    The service owns the registry and remote state. Providers retain local owners
    created through that registry; both management warm-up and feature inference
    therefore reach the same application-owned objects.
    """

    service: DeploymentService
    providers: ResourceProviders


def assemble_deployments(
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> DeploymentAssembly:
    """
    Assemble deployment management without constructing any feature pipelines.

    Register deferred local factories and create an empty remote cache. No owner,
    model, backend controller or background task is created here.
    """
    catalog = catalog if catalog is not None else get_catalog()
    registry = registry if registry is not None else ResourceRegistry()
    providers = register_configured_resources(catalog, registry)
    status = DeploymentStatusCache()
    activity = DeploymentActivityTracker()
    service = DeploymentService(
        catalog,
        resources=registry,
        status=status,
        activity=activity,
        cpu_idle_seconds=settings.cpu_idle_timeout_minutes * 60,
    )
    return DeploymentAssembly(service=service, providers=providers)


class Application(BaseModel):
    """
    Explicit runtime dependencies shared by HTTP and direct batch callers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deployments: DeploymentService
    sanitizer: Callable[[], PresidioSanitizer]
    foi: FOIResources
    sar: SARClassifier
    automatic: AutomaticClassifier
    crawl_policy: CrawlPolicy
    foi_finder: FOIFinder


def assemble_application(
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> Application:
    """
    Assemble independent runtime ownership without loading models or starting VMs.

    Use the supplied catalog and registry, or create defaults for this assembly.
    The service owns a fresh cache and backend controllers; feature providers resolve
    cold local owners through the shared registry. Feature configuration is checked
    only when its provider is used, allowing management-only catalogs to assemble.

    Reuse the returned application across a batch. Call its deployment service's
    close() method when finished to release resources.
    """
    from .automatic.pipeline import AutomaticClassifier
    from .foi.backends import FOIResources
    from .foi.email_finder.service import FOIFinder
    from .sanitization.presidio import PresidioSanitizer
    from .sar.pipeline import SARClassifier
    from .web.crawl import CrawlPolicy

    deployment_assembly = assemble_deployments(catalog=catalog, registry=registry)
    service = deployment_assembly.service
    catalog = service.catalog
    providers = deployment_assembly.providers

    sanitizer = providers.provider(
        resource_types.PRESIDIO, lambda: catalog.require_sanitizer().slug
    )
    foi = FOIResources.from_config(
        catalog=catalog, providers=providers, sanitizer=sanitizer
    )
    sar = SARClassifier(
        logistic_provider=providers.provider(
            resource_types.SAR_LOGISTIC, lambda: catalog.require_sar().logistic
        ),
        encoder_provider=providers.provider(
            resource_types.SAR_CLASSIFIER, lambda: catalog.require_sar().classifier
        ),
    )

    automatic = AutomaticClassifier(
        logistic_provider=providers.provider(
            resource_types.AUTOMATIC_LOGISTIC,
            lambda: catalog.require_automatic().logistic,
        ),
        encoder_provider=providers.provider(
            resource_types.AUTOMATIC_CLASSIFIER,
            lambda: catalog.require_automatic().classifier,
        ),
        max_concurrency=settings.automatic_max_concurrency,
    )
    Application.model_rebuild(
        _types_namespace={
            "FOIResources": FOIResources,
            "FOIFinder": FOIFinder,
            "CrawlPolicy": CrawlPolicy,
            "SARClassifier": SARClassifier,
            "AutomaticClassifier": AutomaticClassifier,
            "PresidioSanitizer": PresidioSanitizer,
        }
    )
    crawl_policy = CrawlPolicy()
    return Application(
        deployments=service,
        sanitizer=sanitizer,
        foi=foi,
        sar=sar,
        automatic=automatic,
        crawl_policy=crawl_policy,
        foi_finder=FOIFinder(settings, catalog, crawl_policy=crawl_policy),
    )


def assemble_service(
    catalog: DeploymentCatalog | None = None, registry: ResourceRegistry | None = None
) -> DeploymentService:
    """
    Assemble management dependencies without constructing feature pipelines.

    The service retains registered factories and their providers through the
    registry. Direct callers manage its idle task and call close() when finished;
    callers needing feature inference should use assemble_application().
    """
    return assemble_deployments(catalog=catalog, registry=registry).service


@lru_cache(maxsize=1)
def default_application() -> Application:
    """
    Process-local default for public convenience calls outside an API app.

    Batch callers wanting explicit isolation should use assemble_application().
    Controllers and loaded weights belong to this assembly's registry.
    """
    return assemble_application()
