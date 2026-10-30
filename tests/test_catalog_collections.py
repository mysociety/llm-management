"""
Slug lookup uses one collection while configuration exports remain ordered arrays.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from llm_management.deployments.catalog import DeploymentCatalog


@pytest.fixture
def catalog(api_catalog: DeploymentCatalog) -> DeploymentCatalog:
    """
    Include local, remote, group, API model and search entries in one valid catalog.
    """
    data = DeploymentCatalog.load().model_dump()
    data["api"] = api_catalog.api.model_dump()
    data["foi_email_finder"] = api_catalog.model_dump()["foi_email_finder"]
    return DeploymentCatalog.model_validate(data)


def test_collections_preserve_order_types_and_array_round_trips(
    catalog: DeploymentCatalog,
) -> None:
    exported = catalog.model_dump()
    collections = [
        catalog.exoscale.deployment,
        catalog.local.deployment,
        catalog.api.deployment,
        catalog.api.search,
        catalog.deployment_group,
    ]
    arrays = [
        exported["exoscale"]["deployment"],
        exported["local"]["deployment"],
        exported["api"]["deployment"],
        exported["api"]["search"],
        exported["deployment_group"],
    ]
    for entries, array in zip(collections, arrays, strict=True):
        assert isinstance(entries, dict)
        assert isinstance(array, list)
        assert list(entries) == [entry["slug"] for entry in array]
    assert DeploymentCatalog.model_validate(exported) == catalog
    assert DeploymentCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    assert catalog.exoscale.resolve(None, all_=True) == list(
        catalog.exoscale.deployment.values()
    )
    for slug, group in catalog.deployment_group.items():
        assert catalog.get_group(slug) is group
        assert isinstance(group.deployments, list)


@pytest.mark.parametrize(
    "section,field",
    [
        ("exoscale", "deployment"),
        ("local", "deployment"),
        ("api", "deployment"),
        ("api", "search"),
        (None, "deployment_group"),
    ],
)
def test_collections_reject_duplicates_before_indexing(
    catalog: DeploymentCatalog, section: str | None, field: str
) -> None:
    data = catalog.model_dump()
    parent = data[section] if section is not None else data
    parent[field].append(parent[field][0].copy())
    with pytest.raises(ValidationError, match="slugs must be unique"):
        DeploymentCatalog.model_validate(data)


@pytest.mark.parametrize(
    "section,field",
    [
        ("exoscale", "deployment"),
        ("local", "deployment"),
        ("api", "deployment"),
        ("api", "search"),
        (None, "deployment_group"),
    ],
)
def test_keyed_input_is_validated_and_exports_as_an_array(
    catalog: DeploymentCatalog, section: str | None, field: str
) -> None:
    data = catalog.model_dump()
    parent = data[section] if section is not None else data
    original = parent[field]
    parent[field] = {entry["slug"]: entry for entry in original}
    assert DeploymentCatalog.model_validate(data) == catalog
    entry = parent[field].pop(original[0]["slug"])
    parent[field]["wrong-key"] = entry
    with pytest.raises(ValidationError, match="does not match entry slug"):
        DeploymentCatalog.model_validate(data)


def test_keyed_remote_input_resolves_checkpoint_without_mutating_input(
    catalog: DeploymentCatalog,
) -> None:
    data = catalog.model_dump()
    remote = {entry["slug"]: entry for entry in data["exoscale"]["deployment"]}
    data["exoscale"]["deployment"] = remote
    slug = catalog.require_foi().topic_deployment
    remote[slug].pop("model")
    restored = DeploymentCatalog.model_validate(data)
    assert restored.get(slug).model == catalog.get(slug).model
    assert "model" not in remote[slug]
    remote[slug]["model"] = "wrong/checkpoint"
    with pytest.raises(ValidationError, match="disagrees with model reference"):
        DeploymentCatalog.model_validate(data)


def test_slug_identity_cannot_drift_from_its_key(catalog: DeploymentCatalog) -> None:
    for entries in (
        catalog.exoscale.deployment,
        catalog.local.deployment,
        catalog.api.deployment,
        catalog.api.search,
        catalog.deployment_group,
    ):
        for slug, entry in entries.items():
            with pytest.raises(ValidationError, match="frozen"):
                entry.slug = "renamed"
            assert entry.slug == slug


def test_serialized_collection_schemas_describe_typed_arrays() -> None:
    schema: dict[str, Any] = DeploymentCatalog.model_json_schema(mode="serialization")
    for owner, field in (
        ("ExoscaleDeployments", "deployment"),
        ("LocalDeployments", "deployment"),
        ("APIDeployments", "deployment"),
        ("APIDeployments", "search"),
    ):
        collection = schema["$defs"][owner]["properties"][field]
        assert collection["type"] == "array"
        assert collection["items"]
    assert schema["properties"]["deployment_group"]["type"] == "array"


def test_validation_collection_schemas_retain_concrete_entry_types() -> None:
    schema = DeploymentCatalog.model_json_schema()
    for owner, field in (
        ("ExoscaleDeployments", "deployment"),
        ("LocalDeployments", "deployment"),
        ("APIDeployments", "deployment"),
        ("APIDeployments", "search"),
        (None, "deployment_group"),
    ):
        model = schema["$defs"][owner] if owner else schema
        array, mapping = model["properties"][field]["anyOf"]
        assert array["type"] == "array"
        assert mapping["type"] == "object"
        assert array["items"]
        assert array["items"] == mapping["additionalProperties"]
        if owner == "LocalDeployments":
            assert "discriminator" in array["items"]


def test_slug_mapped_list_works_without_a_container_serializer() -> None:
    from pydantic import TypeAdapter

    from llm_management.deployments.api_config import APISearchDeployment
    from llm_management.deployments.collections import SlugMappedList

    adapter = TypeAdapter(SlugMappedList[APISearchDeployment])
    entries = [{"slug": "search", "service": "tavily"}]
    indexed = adapter.validate_python(entries)
    assert isinstance(indexed["search"], APISearchDeployment)
    assert adapter.validate_python({"search": entries[0]}) == indexed
    assert adapter.dump_python(indexed) == entries
    assert adapter.validate_json(adapter.dump_json(indexed)) == indexed
    with pytest.raises(ValidationError):
        adapter.validate_python([{"slug": "search"}])


def test_collection_defaults_are_independent_and_serialize_as_empty_arrays() -> None:
    from llm_management.deployments.api_config import (
        APIDeployments,
        APISearchDeployment,
    )

    first = APIDeployments()
    second = APIDeployments()
    first.search["search"] = APISearchDeployment(slug="search", service="tavily")
    assert second.search == {}
    assert second.model_dump()["search"] == []
