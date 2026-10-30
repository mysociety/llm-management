"""
Application-scoped cold owners resolved through typed resource registrations.

Feature code asks for typed provider callables. The deployment service asks the
resource registry for lifecycle guards. ResourceProviders connects those two
access paths so both use the same owner and guard within an application.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, cast

from ..deployments.catalog import DeploymentCatalog
from ..deployments.errors import LLMManagementError
from ..deployments.resource_types import (
    RESOURCE_TYPES,
    Config,
    Owner,
    ResourceOwner,
    ResourceType,
)
from .resources import LocalResource, ResourceRegistry


class ResourceProviders:
    """
    Keep an application's model controllers and provide deferred access to them.

    An "owner" is the Python object responsible for a particular model resource.
    For example, a CategoryMatcher stores checkpoint configuration, holds its
    loaded encoder/tokenizer/index in matcher.loaded, and implements predict(),
    load() and unload(). The owner exists before those model objects are loaded.

    A "guard" is a LocalResource object, usually available as owner.resource.
    It holds the owner's load/unload callbacks and tracks readiness, recent use
    and active work. Its use() context manager takes a lease: while that context
    is active, idle cleanup cannot unload the resource. The guard coordinates
    access; the owner holds the loaded model objects and implements inference.

    A "provider" is a zero-argument callable returning that owner. Creating a
    provider does not construct or load the model. Calling it obtains the cold
    owner if necessary; the feature then calls the owner's inference methods.

    The catalog describes deployment instances and their configuration schemas.
    ResourceType registrations associate those schemas with factories and owner
    types. This container connects each configured instance to the application's
    ResourceRegistry without importing its factory or constructing its owner.

    There are two ways to reach a resource. Feature assembly calls provider() to
    obtain a callable with a concrete return type, such as CategoryMatcher. When
    inference invokes that callable, registry.get() constructs the owner if
    needed. Alternatively, a deployment warm-up can call registry.get() first.
    Both paths run the same registered factory and retain the same owner here.

    The registry stores lifecycle guards; this container stores the owners that
    expose inference methods. Some factories return a LocalResource directly,
    such as tokenizer factories, in which case the guard is also the owner.
    The registry serializes first construction under its lock, so concurrent
    lookups cannot create separate owners for the same deployment.

    Construction remains cold. Loading happens through the guard during warm-up
    or inference, and model owners acquire leases around their work. Idle
    unloading clears loaded state without removing the owner from this container,
    allowing the same owner to reload later. Each application gets its own
    container and registry. This container holds live Python objects; it is not
    a disk cache for model artifacts.

    """

    def __init__(self, catalog: DeploymentCatalog, registry: ResourceRegistry) -> None:
        """
        Register cold factories for every local deployment in the validated catalog.

        Bind each factory to its configuration and registration without resolving
        its lazy Python imports. The owner map stays empty until a registry lookup
        constructs an owner. Use one
        provider container per application registry so ownership is unambiguous.
        """
        self.catalog = catalog
        self.registry = registry
        self._owners: dict[str, ResourceOwner] = {}
        for config in catalog.local.deployment.values():
            resource_type = RESOURCE_TYPES[type(config)]
            registry.register_factory(
                config.slug, partial(self._create_owner, config, resource_type)
            )

    def _create_owner(
        self, config: Config, resource_type: ResourceType[Config, Owner]
    ) -> LocalResource[Any]:
        """
        Construct and retain an owner, returning its guard to the registry.

        ResourceRegistry.get() invokes this callback under its construction lock.
        It must remain an internal callback: calling it directly would bypass
        that lock and could replace an existing owner. ResourceType.create()
        checks the configuration and factory return type; this method checks that
        the owner exposes a LocalResource with the configured deployment name.

        Retain the owner only after those checks succeed. Returning the guard
        lets deployment warm-up and cleanup reach the same resource as inference.
        No model loading or warm-up occurs here.
        """
        owner = resource_type.create(
            config, catalog=self.catalog, registry=self.registry
        )
        guard = owner if isinstance(owner, LocalResource) else owner.resource
        if not isinstance(guard, LocalResource) or guard.name != config.slug:
            raise TypeError(
                f"Resource {config.slug!r} returned an invalid lifecycle guard"
            )
        self._owners[config.slug] = owner
        return guard

    def _resolve_owner(self, slug: str) -> ResourceOwner:
        """
        Resolve a configured owner through the registry's synchronized lookup.

        This is the internal target of the callables exposed by provider(). The registry constructs a missing owner through _create_owner()
        before this method reads it. A previous warm-up may already have done so.
        Construction errors propagate to the feature's existing failure handling.
        """
        self.registry.get(slug)
        return self._owners[slug]

    def provider(
        self,
        resource_type: ResourceType[Config, Owner],
        slug: Callable[[], str | None],
    ) -> Callable[[], Owner]:
        """
        Build a deferred lookup whose return type comes from its registration.

        For example, passing resource_types.CATEGORY_MATCHER returns a callable
        typed as Callable[[], CategoryMatcher]. Only canonical registrations are
        accepted, so callers cannot pair an existing schema with a different
        claimed owner type.

        The slug is itself a callable because feature configuration is optional.
        Evaluate it only when the returned provider is invoked, allowing an
        application with no FOI pipeline to assemble for deployment management.
        A missing binding raises ValueError; an unknown deployment or incompatible
        schema raises LLMManagementError before any owner is constructed.

        Check the deployment schema even when registrations share an owner class,
        as the two tokenizer types do. Owner storage is heterogeneous, so the final
        cast recovers the concrete type guaranteed by the canonical registration,
        the schema check, and ResourceType.create()'s runtime owner check.
        """
        if RESOURCE_TYPES.get(resource_type.config_type) is not resource_type:
            raise ValueError("Provider resolution requires a registered resource type")

        def resolve() -> Owner:
            """
            Validate the current feature binding and resolve its retained owner.
            """
            name = slug()
            if name is None:
                raise ValueError(f"{resource_type.loader} resource is not configured")
            self.catalog.local.get(name, resource_type.config_type)
            if name not in self.registry.names():
                raise LLMManagementError(f"No local provider found with slug {name!r}")
            return cast(Owner, self._resolve_owner(name))

        return resolve
