"""
Registration-derived validation, cold construction, typing and shared ownership.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import assert_type
from unittest.mock import Mock

import pytest
from pydantic import TypeAdapter, ValidationError

from llm_management.deployments import resource_types
from llm_management.deployments.catalog import DeploymentCatalog
from llm_management.deployments.errors import LLMManagementError
from llm_management.deployments.resource_types import LocalDeploymentConfig
from llm_management.foi.category_classification import CategoryMatcher
from llm_management.inference.providers import ResourceProviders
from llm_management.inference.resources import LocalResource, ResourceRegistry
from llm_management.sar.resources import LogisticDetector


def test_catalog_schema_serialization_and_all_registrations_agree():
    catalog = DeploymentCatalog.load()
    adapter = TypeAdapter(LocalDeploymentConfig)
    schema = adapter.json_schema()
    tags = {item.loader for item in resource_types.RESOURCE_TYPES.values()}
    assert set(schema["discriminator"]["mapping"]) == tags
    assert {type(item) for item in catalog.local.deployment.values()} <= set(
        resource_types.RESOURCE_TYPES
    )
    for config in catalog.local.deployment.values():
        assert adapter.validate_json(adapter.dump_json(config)) == config
    assert DeploymentCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        adapter.validate_python({"slug": "unknown", "loader": "import.me"})


def test_fresh_process_keeps_catalog_and_assembly_cold():
    # Other tests and application fixtures may already import owners. A fresh
    # interpreter catches accidental eager imports, including library side effects.
    script = """
import builtins
import socket
import sys

real_import = builtins.__import__
blocked = ("torch", "transformers", "huggingface_hub", "spacy", "presidio_analyzer")

def checked_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError("Cold startup imported " + name)
    return real_import(name, *args, **kwargs)

def no_network(*args, **kwargs):
    raise AssertionError("Cold startup attempted network I/O")

builtins.__import__ = checked_import
socket.socket.connect = no_network
from llm_management.deployments.catalog import DeploymentCatalog
catalog = DeploymentCatalog.load()
assert not any(name.startswith(("llm_management.foi.", "llm_management.sar.",
    "llm_management.automatic.", "llm_management.sanitization.")) for name in sys.modules)
from llm_management.bootstrap import assemble_application
from llm_management.inference.resources import ResourceRegistry
from llm_management.inference.providers import ResourceProviders
from llm_management.deployments.resource_types import RESOURCE_TYPES
registry = ResourceRegistry()
application = assemble_application(catalog, registry)
assert registry.all() == []
# All real registered factories must also construct cold owners.
other_registry = ResourceRegistry()
providers = ResourceProviders(catalog, other_registry)
for config in catalog.local.deployment.values():
    owner = providers.provider(RESOURCE_TYPES[type(config)], lambda: config.slug)()
    assert not other_registry.get(config.slug).status().ready
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_typed_providers_and_warmup_share_owners_across_resource_kinds(monkeypatch):
    catalog = DeploymentCatalog.load()
    first_registry, second_registry = ResourceRegistry(), ResourceRegistry()
    first = ResourceProviders(catalog, first_registry)
    second = ResourceProviders(catalog, second_registry)
    category_slug = catalog.require_foi().category_matcher
    logistic_slug = catalog.require_sar().logistic
    assert category_slug is not None

    category = first.provider(resource_types.CATEGORY_MATCHER, lambda: category_slug)
    logistic = first.provider(resource_types.SAR_LOGISTIC, lambda: logistic_slug)
    assert_type(category, Callable[[], CategoryMatcher])
    assert_type(logistic, Callable[[], LogisticDetector])

    for owner_type, token, slug, provider in [
        (CategoryMatcher, resource_types.CATEGORY_MATCHER, category_slug, category),
        (LogisticDetector, resource_types.SAR_LOGISTIC, logistic_slug, logistic),
    ]:
        load, unload = Mock(return_value=object()), Mock()
        monkeypatch.setattr(owner_type, "load", load)
        monkeypatch.setattr(owner_type, "unload", unload)
        # Warm-up can construct the owner before the inference provider is used.
        guard = first_registry.get(slug)
        assert guard is not None
        load.assert_not_called()
        guard.warmup()
        owner = provider()
        assert owner.resource is guard
        assert owner is provider()
        assert second.provider(token, lambda: slug)() is not owner
        load.assert_called_once()
        with guard.use():
            first_registry.expire(0)
            unload.assert_not_called()
        first_registry.expire(0)
        unload.assert_called_once()
        assert not guard.status().ready
        assert provider() is owner
        guard.warmup()
        assert load.call_count == 2


def test_typed_provider_rejects_different_schema_with_same_owner_type():
    catalog = DeploymentCatalog.load()
    registry = ResourceRegistry()
    providers = ResourceProviders(catalog, registry)
    provider = providers.provider(
        resource_types.RESPONSE_TOKENIZER, lambda: catalog.require_foi().tokenizer
    )
    with pytest.raises(LLMManagementError, match="ResponseTokenizerDeployment"):
        provider()
    assert registry.all() == []


def test_management_only_catalog_and_missing_binding_are_deferred():
    from llm_management.bootstrap import assemble_application

    catalog = DeploymentCatalog()
    application = assemble_application(catalog)
    assert application.foi.category_matcher is None
    with pytest.raises(LLMManagementError, match="FOI pipeline is not configured"):
        application.foi.classifier()
    for provider in (
        application.automatic.logistic_provider,
        application.automatic.encoder_provider,
    ):
        with pytest.raises(
            ValueError, match="Application requires an automatic-response pipeline"
        ):
            provider()
    providers = ResourceProviders(catalog, ResourceRegistry())
    with pytest.raises(ValueError, match="not configured"):
        providers.provider(resource_types.CATEGORY_MATCHER, lambda: None)()


def test_concurrent_providers_retain_one_owner(monkeypatch):
    from llm_management.foi import category_classification

    catalog = DeploymentCatalog.load()
    factory = Mock(wraps=category_classification.category_matcher)
    monkeypatch.setattr(category_classification, "category_matcher", factory)
    registry = ResourceRegistry()
    providers = ResourceProviders(catalog, registry)
    provider = providers.provider(
        resource_types.CATEGORY_MATCHER, lambda: catalog.require_foi().category_matcher
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        owners = list(executor.map(lambda _: provider(), range(24)))
    assert all(owner is owners[0] for owner in owners)
    factory.assert_called_once()
    slug = catalog.require_foi().category_matcher
    assert slug is not None
    assert owners[0].resource is registry.get(slug)


def test_invalid_factory_owner_is_rejected_and_can_retry(monkeypatch):
    from llm_management.foi import category_classification

    catalog = DeploymentCatalog.load()
    registry = ResourceRegistry()
    providers = ResourceProviders(catalog, registry)
    provider = providers.provider(
        resource_types.CATEGORY_MATCHER, lambda: catalog.require_foi().category_matcher
    )
    factory = category_classification.category_matcher
    monkeypatch.setattr(
        category_classification, "category_matcher", Mock(return_value=object())
    )
    with pytest.raises(TypeError, match="did not produce CategoryMatcher"):
        provider()
    assert registry.all() == []
    monkeypatch.setattr(category_classification, "category_matcher", factory)
    assert isinstance(provider(), CategoryMatcher)


def test_guard_owner_registration_uses_same_provider_as_warmup(monkeypatch):
    from llm_management.foi import topic_classification

    catalog = DeploymentCatalog.load()
    load = Mock(return_value=object())
    monkeypatch.setattr(topic_classification, "question_extractor_tokenizer", load)
    registry = ResourceRegistry()
    providers = ResourceProviders(catalog, registry)
    provider = providers.provider(
        resource_types.TOPIC_TOKENIZER, lambda: catalog.require_foi().tokenizer
    )
    guard = provider()
    assert isinstance(guard, LocalResource)
    assert guard is registry.get(catalog.require_foi().tokenizer)
    load.assert_not_called()
    guard.warmup()
    load.assert_called_once()


def test_provider_rejects_unregistered_type_token():
    catalog = DeploymentCatalog.load()
    registry = ResourceRegistry()
    providers = ResourceProviders(catalog, registry)
    registered = resource_types.CATEGORY_MATCHER
    unregistered = resource_types.ResourceType(
        registered.config_type, registered.binding
    )
    with pytest.raises(ValueError, match="registered resource type"):
        providers.provider(unregistered, lambda: catalog.require_foi().category_matcher)
    assert registry.all() == []


def test_one_type_supports_independent_instances_in_the_same_application():
    """
    A registration is reusable; owners and guards belong to configured slugs.
    """
    catalog = DeploymentCatalog.load()
    first_slug = catalog.require_foi().category_matcher
    assert first_slug is not None
    first_config = catalog.local.get(
        first_slug, resource_types.CATEGORY_MATCHER.config_type
    )
    data = catalog.model_dump()
    second_slug = "additional_category_matcher"
    data["local"]["deployment"].append(
        {**first_config.model_dump(), "slug": second_slug}
    )
    catalog = DeploymentCatalog.model_validate(data)
    registry = ResourceRegistry()
    providers = ResourceProviders(catalog, registry)
    first = providers.provider(resource_types.CATEGORY_MATCHER, lambda: first_slug)
    second = providers.provider(resource_types.CATEGORY_MATCHER, lambda: second_slug)
    assert_type(first, Callable[[], CategoryMatcher])
    assert_type(second, Callable[[], CategoryMatcher])
    assert registry.all() == []

    first_owner, second_owner = first(), second()
    assert first_owner is first()
    assert second_owner is second()
    assert first_owner is not second_owner
    assert first_owner.checkpoint == second_owner.checkpoint
    assert first_owner.resource is registry.get(first_slug)
    assert second_owner.resource is registry.get(second_slug)
    assert first_owner.resource is not second_owner.resource
    assert first_owner.loaded is second_owner.loaded is None
