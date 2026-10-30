"""
Assemble runtime dependencies and register cold model factories.

Loading a catalog is a pure configuration operation. Assembly establishes runtime
ownership without loading model weights or creating cloud deployments.
"""

from __future__ import annotations

import asyncio
from builtins import BaseExceptionGroup
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, NamedTuple, Self

from .deployments import resource_types
from .deployments.activity import DeploymentActivityTracker
from .deployments.catalog import DeploymentCatalog, get_catalog
from .deployments.management import DeploymentManagement
from .deployments.service import DeploymentService
from .deployments.status import DeploymentStatusCache
from .inference.providers import ResourceProviders
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
    therefore reach the same application-owned objects. Management interprets commands
    using this service; the service has no dependency on that interface.
    """

    service: DeploymentService
    providers: ResourceProviders
    management: DeploymentManagement


def assemble_deployments(
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> DeploymentAssembly:
    """
    Assemble deployment management without constructing any feature pipelines.

    Register deferred local factories and create separate status and activity stores.
    No owner, model, backend controller or background task is created here.
    """
    catalog = catalog if catalog is not None else get_catalog()
    registry = registry if registry is not None else ResourceRegistry()
    providers = ResourceProviders(catalog=catalog, registry=registry)
    status = DeploymentStatusCache()
    activity = DeploymentActivityTracker()
    service = DeploymentService(
        catalog,
        resources=registry,
        status=status,
        activity=activity,
        cpu_idle_seconds=settings.cpu_idle_timeout_minutes * 60,
    )
    return DeploymentAssembly(
        service=service, providers=providers, management=DeploymentManagement(service)
    )


class ApplicationRuntime:
    """
    Own services, resources and lifecycle independently of the FastAPI HTTP app.

    Assembly is cold. Enter ``async with runtime`` to start idle cleanup;
    exiting drains classifier work before releasing local and remote resources.
    Call ``close()`` when using the runtime without a context manager.
    Shutdown is terminal: assemble a new runtime for a new runtime lifetime.
    Remote inference holds explicit leases through deployments.use(slug).
    """

    def __init__(
        self,
        *,
        deployments: DeploymentService,
        management: DeploymentManagement,
        sanitizer: Callable[[], PresidioSanitizer],
        foi: FOIResources,
        sar: SARClassifier,
        automatic: AutomaticClassifier,
        crawl_policy: CrawlPolicy,
        foi_finder: FOIFinder,
    ) -> None:
        """
        Retain injected runtime dependencies without starting background work.

        Configuration has already been validated by DeploymentCatalog. These are
        live service objects, retained by identity rather than parsed as data.
        Lifecycle tasks belong to this instance and start only on entry or close.
        """
        self.deployments = deployments
        self.management = management
        self.sanitizer = sanitizer
        self.foi = foi
        self.sar = sar
        self.automatic = automatic
        self.crawl_policy = crawl_policy
        self.foi_finder = foi_finder
        self._idle_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        """
        Start one idle cleanup loop, rejecting nested or completed lifetimes.
        """
        if self._close_task is not None or self._idle_task is not None:
            raise RuntimeError("Application lifecycle has already started or closed")
        self._idle_task = asyncio.create_task(self.deployments.idle_scaler())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Release runtime resources even when the caller fails or is cancelled.
        """
        await self.close()

    async def close(self) -> None:
        """
        Finish shutdown once, including when multiple callers request it.

        Shield cleanup from caller cancellation and wait for it before propagating
        cancellation. Stop idle cleanup, drain both classifiers, then release the
        deployment service. A failed step does not skip the remaining steps.
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                if self._close_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        """
        Perform ordered cleanup behind the public, idempotent close method.
        """
        failures: list[BaseException] = []
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                failures.append(exc)
        for close in (self.sar.close, self.automatic.close, self.deployments.close):
            try:
                await close()
            except (Exception, asyncio.CancelledError) as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("Application shutdown failed", failures)


def assemble_runtime(
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> ApplicationRuntime:
    """
    Assemble independent runtime ownership without loading models or starting VMs.

    Use the supplied catalog and registry, or create defaults for this assembly.
    The service owns separate status and activity stores and retains backend
    controllers on first use. Feature providers resolve cold local owners through
    the shared registry. Feature configuration is checked only when its provider
    is used, allowing management-only catalogs to assemble.

    Reuse the returned runtime across a batch inside an async context manager,
    or call its close() method when finished to drain work and release resources.
    """
    from .automatic.pipeline import AutomaticClassifier
    from .foi.backends import FOIResources
    from .foi.email_finder.service import FOIFinder
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
    crawl_policy = CrawlPolicy()
    return ApplicationRuntime(
        deployments=service,
        management=deployment_assembly.management,
        sanitizer=sanitizer,
        foi=foi,
        sar=sar,
        automatic=automatic,
        crawl_policy=crawl_policy,
        foi_finder=FOIFinder(settings, catalog, crawl_policy=crawl_policy),
    )
