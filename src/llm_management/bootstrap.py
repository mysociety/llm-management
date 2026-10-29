"""Assemble application dependencies and register cold model factories.

Loading a catalog is a pure configuration operation. Assembly establishes runtime
ownership without loading model weights or creating cloud deployments.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from .deployments.cache import DeploymentCache
from .deployments.catalog import DeploymentCatalog, LocalDeploymentConfig, get_catalog
from .deployments.service import DeploymentService
from .foi.email_finder.service import FOIFinder
from .inference.resources import LocalResource, ResourceRegistry
from .settings import settings
from .web.crawl import CrawlPolicy

if TYPE_CHECKING:
    from .automatic.pipeline import AutomaticClassifier
    from .foi.backends import FOIResources
    from .sanitization.presidio import PresidioSanitizer
    from .sar.pipeline import SARClassifier


Owner = TypeVar("Owner")


class ResourceController(Protocol):
    """
    A cold owner exposing its lifecycle guard to the application registry.
    """

    resource: LocalResource[Any]


class FactoryLocation(NamedTuple):
    """
    Import path for a resource owner that is constructed only on first use.
    """

    module: str
    factory: str


def register_configured_resources(
    catalog: DeploymentCatalog, registry: ResourceRegistry
) -> dict[str, Callable[[], ResourceController | LocalResource[Any]]]:
    """
    Register cold factories and return providers for their owning controllers.

    Registration does not import feature owners or load weights. First lookup imports
    and constructs the owner, then exposes its lifecycle guard through the registry.
    Providers retain the controller so inference and warm-up use the same objects.
    """
    from importlib import import_module

    owners = {
        "foi_category_matcher": FactoryLocation(
            "foi.category_classification", "category_matcher"
        ),
        "modernbert_classifier": FactoryLocation("foi.backends", "question_classifier"),
        "modernbert_head": FactoryLocation(
            "foi.backends", "question_classification_head"
        ),
        "response_tokenizer": FactoryLocation(
            "foi.response_model", "tokenizer_resource"
        ),
        "topic_tokenizer": FactoryLocation(
            "foi.topic_classification", "tokenizer_resource"
        ),
        "sar_logistic": FactoryLocation("sar.resources", "logistic_detector"),
        "sar_deberta": FactoryLocation("sar.resources", "deberta_detector"),
        "auto_logistic": FactoryLocation("automatic.resources", "logistic_detector"),
        "auto_deberta": FactoryLocation("automatic.resources", "encoder_detector"),
        "presidio": FactoryLocation("sanitization", "configured_presidio"),
    }

    controllers: dict[str, ResourceController | LocalResource[Any]] = {}

    def create(config: LocalDeploymentConfig) -> LocalResource[Any]:
        module_name, factory_name = owners[config.loader]
        factory = getattr(import_module(f"llm_management.{module_name}"), factory_name)
        owner = factory(config, catalog=catalog, registry=registry)
        controllers[config.slug] = owner
        return owner if isinstance(owner, LocalResource) else owner.resource

    providers: dict[str, Callable[[], ResourceController | LocalResource[Any]]] = {}

    def provider(slug: str) -> ResourceController | LocalResource[Any]:
        registry.get(slug)
        return controllers[slug]

    for config in catalog.local.deployment:
        registry.register_factory(config.slug, lambda config=config: create(config))
        providers[config.slug] = lambda slug=config.slug: provider(slug)
    return providers


def typed_provider(
    providers: dict[str, Callable[[], ResourceController | LocalResource[Any]]],
    slug: Callable[[], str],
    owner_type: type[Owner],
) -> Callable[[], Owner]:
    """
    Wrap a cold provider with deferred ownership validation.

    Resolve the slug and construct the owner only when the returned callable runs.
    Check its expected type after lookup so invalid dynamic factory wiring fails
    at the ownership boundary without eagerly requiring feature configuration.
    """

    def resolve() -> Owner:
        name = slug()
        owner = providers[name]()
        if not isinstance(owner, owner_type):
            raise TypeError(f"Resource {name!r} did not produce {owner_type.__name__}")
        return owner

    return resolve


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


def response_tokenizer_slug(catalog: DeploymentCatalog) -> str:
    """
    Resolve optional response configuration only when its cold provider is used.
    """
    slug = catalog.require_foi().response_tokenizer
    if slug is None:
        raise ValueError("FOI response tokenizer is not configured")
    return slug


def category_matcher_slug(catalog: DeploymentCatalog) -> str:
    """
    Resolve the optional category matcher only when its provider is used.
    """
    slug = catalog.require_foi().category_matcher
    if slug is None:
        raise ValueError("FOI category matcher is not configured")
    return slug


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
    close() when finished to release inactive local resources and scale down remote
    deployments recorded as active by that service.
    """
    from .automatic.pipeline import AutomaticClassifier
    from .automatic.resources import LogisticDetector as AutomaticLogisticDetector
    from .foi.backends import FOIResources, ModernBertClassificationHead
    from .foi.category_classification import CategoryMatcher
    from .inference.classifiers import (
        AuthenticatedSequenceClassifier,
        LocalSequenceClassifier,
    )
    from .sanitization.presidio import PresidioSanitizer
    from .sar.pipeline import SARClassifier
    from .sar.resources import LogisticDetector as SARLogisticDetector

    catalog = catalog if catalog is not None else get_catalog()
    registry = registry if registry is not None else ResourceRegistry()
    providers = register_configured_resources(catalog, registry)
    service = DeploymentService(
        catalog,
        DeploymentCache(),
        registry,
        cpu_idle_seconds=settings.cpu_idle_timeout_minutes * 60,
    )

    sanitizer = typed_provider(
        providers, lambda: catalog.require_sanitizer().slug, PresidioSanitizer
    )

    foi = FOIResources(
        catalog=catalog,
        sanitizer=sanitizer,
        classifier=typed_provider(
            providers, lambda: catalog.require_foi().classifier, LocalSequenceClassifier
        ),
        head=typed_provider(
            providers, lambda: catalog.require_foi().head, ModernBertClassificationHead
        ),
        tokenizer=typed_provider(
            providers, lambda: catalog.require_foi().tokenizer, LocalResource
        ),
        response_tokenizer=typed_provider(
            providers, lambda: response_tokenizer_slug(catalog), LocalResource
        ),
        category_matcher=(
            typed_provider(
                providers,
                lambda: category_matcher_slug(catalog),
                CategoryMatcher,
            )
            if catalog.foi is not None and catalog.foi.category_matcher is not None
            else None
        ),
    )
    sar = SARClassifier(
        typed_provider(
            providers, lambda: catalog.require_sar().logistic, SARLogisticDetector
        ),
        typed_provider(
            providers,
            lambda: catalog.require_sar().classifier,
            AuthenticatedSequenceClassifier,
        ),
    )

    def automatic_slug(role: Literal["logistic", "classifier"]) -> str:
        automatic_config = catalog.automatic
        if automatic_config is None:
            raise ValueError("Application requires an automatic-response pipeline")
        return getattr(automatic_config, role)

    automatic = AutomaticClassifier(
        typed_provider(
            providers, lambda: automatic_slug("logistic"), AutomaticLogisticDetector
        ),
        typed_provider(
            providers,
            lambda: automatic_slug("classifier"),
            AuthenticatedSequenceClassifier,
        ),
        max_concurrency=settings.automatic_max_concurrency,
    )
    Application.model_rebuild(
        _types_namespace={
            "FOIResources": FOIResources,
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
    Return the deployment service from a cold application assembly.

    This convenience call still constructs the feature dependency containers and
    registers their cold factories. It does not load models or start deployments.
    Callers needing the feature providers should retain assemble_application()'s
    complete result instead.
    """
    return assemble_application(catalog, registry).deployments


@lru_cache(maxsize=1)
def default_application() -> Application:
    """Process-local default for public convenience calls outside an API app.

    Batch callers wanting explicit isolation should use assemble_application().
    Controllers and loaded weights belong to this assembly's registry.
    """
    return assemble_application()
