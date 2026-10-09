"""Catalog references, cold resource registration, and runtime configuration."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from llm_management.deployments import DeploymentCatalog, get_catalog
from llm_management.local_resources import (
    LocalResource,
    ResourceRegistry,
    register_configured_resources,
)
from llm_management.settings import Settings


@pytest.mark.parametrize(
    "case",
    [
        "unknown_loader",
        "unknown_model",
        "invalid_revision",
        "missing_repo",
        "duplicate_local",
        "duplicate_remote",
        "invalid_limit",
        "wrong_pipeline_loader",
        "wrong_checkpoint",
        "unknown_pipeline_resource",
        "unknown_remote_model",
        "unknown_field",
    ],
)
def test_invalid_catalog_rejected(case):
    data = DeploymentCatalog.load().model_dump()
    local = data["local"]["deployment"]
    remote = data["exoscale"]["deployment"]
    if case == "unknown_loader":
        local[0]["loader"] = "arbitrary_python_function"
    elif case == "unknown_model":
        local[0]["model_ref"] = "missing"
    elif case == "invalid_revision":
        data["model"]["sar_logistic_v1"]["revision"] = "main"
    elif case == "missing_repo":
        data["model"]["question_slice_v2"].pop("repo")
    elif case == "duplicate_local":
        local.append(local[0])
    elif case == "duplicate_remote":
        remote.append(remote[0])
    elif case == "invalid_limit":
        local[0]["max_tokens"] = 0
    elif case == "wrong_pipeline_loader":
        data["foi"]["classifier"] = "presidio"
    elif case == "wrong_checkpoint":
        local[1]["model_ref"] = "sar_deberta_v1"
    elif case == "unknown_pipeline_resource":
        data["sar"]["classifier"] = "missing"
    elif case == "unknown_remote_model":
        remote[3]["model_ref"] = "missing"
    elif case == "unknown_field":
        remote[0]["modle"] = "typo"
    with pytest.raises(ValidationError):
        DeploymentCatalog.model_validate(data)


def test_loading_and_registration_do_not_import_or_load_owners(monkeypatch):
    import importlib

    imported = Mock(side_effect=AssertionError("No owner imports during validation"))
    monkeypatch.setattr(importlib, "import_module", imported)
    catalog = DeploymentCatalog.load()
    registry = ResourceRegistry()
    register_configured_resources(catalog, registry)
    assert registry.names() == {d.slug for d in catalog.local.deployment}
    assert registry.all() == []
    imported.assert_not_called()

    config = catalog.local.deployment[0]
    load = Mock()
    resource = LocalResource(config.slug, load, Mock())
    factory = Mock(return_value=SimpleNamespace(resource=resource))
    imported.side_effect = None
    imported.return_value = SimpleNamespace(question_classifier=factory)
    assert registry.get(config.slug) is resource
    assert registry.get(config.slug) is resource
    factory.assert_called_once_with(config)
    load.assert_not_called()


def test_catalog_controls_sar_checkpoint_resource_name_and_limit(monkeypatch):
    from llm_management import sar

    data = DeploymentCatalog.load().model_dump()
    data["deployment_group"] = []
    data["sar"]["classifier"] = "custom_sar"
    for entry in data["local"]["deployment"]:
        if entry["loader"] == "sar_deberta_v1":
            entry["slug"] = "custom_sar"
            entry["max_tokens"] = 256
    data["model"]["sar_deberta_v1"]["repo"] = "example/custom-sar"
    data["model"]["sar_deberta_v1"]["revision"] = "a" * 40
    catalog = DeploymentCatalog.model_validate(data)
    monkeypatch.setattr(sar, "get_catalog", lambda: catalog)
    monkeypatch.setattr(sar, "local_resources", ResourceRegistry())
    try:
        classifier = sar.deberta_detector()
        assert classifier.model_name == "example/custom-sar"
        assert classifier.revision == "a" * 40
        assert classifier.max_tokens == 256
        assert classifier.resource.name == "custom_sar"
        assert not classifier.resource.status()["ready"]
    finally:
        sar._deberta_detector.cache_clear()


def test_checkpoint_changes_cannot_disagree_with_remote_model():
    data = DeploymentCatalog.load().model_dump()
    data["model"]["question_slice_v2"]["repo"] = "example/new-checkpoint"
    with pytest.raises(ValidationError, match="disagrees"):
        DeploymentCatalog.model_validate(data)


def test_environment_selects_catalog_path_without_overriding_identity(
    monkeypatch, tmp_path
):
    path = tmp_path / "deployments.toml"
    path.write_text(
        '[model.example]\nrepo = "example/model"\nrevision = "' + "b" * 40 + '"\n'
    )
    monkeypatch.setenv("DEPLOYMENT_CONFIG", str(path))
    monkeypatch.setenv("QUESTION_SLICE_MODEL", "ignored/legacy-setting")
    configured = Settings(_env_file=None)
    assert configured.deployment_config == path
    assert "question_slice_model" not in Settings.model_fields
    catalog = DeploymentCatalog.load(configured.deployment_config)
    assert catalog.model["example"].repo == "example/model"
    assert catalog.exoscale.resolve(None, True) == []


def test_shared_checkpoint_references_resolve_to_same_identity():
    catalog = get_catalog()
    foi = catalog.require_foi()
    local = {d.slug: d for d in catalog.local.deployment}
    assert local[foi.classifier].model_ref == local[foi.head].model_ref
    assert (
        catalog.get(foi.extraction_deployment).model_ref
        == local[foi.classifier].model_ref
    )
    assert catalog.get(foi.topic_deployment).model_ref == local[foi.tokenizer].model_ref
