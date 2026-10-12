"""Cold assembly and model ownership across independent applications."""

from llm_management.bootstrap import assemble_application
from llm_management.deployments.catalog import DeploymentCatalog, SARLogisticDeployment
from llm_management.inference.resources import ResourceRegistry
from llm_management.sar.resources import LogisticDetector


def test_catalog_loading_does_not_register_resources(monkeypatch):
    from unittest.mock import Mock

    from llm_management.deployments.catalog import get_catalog

    register = Mock(
        side_effect=AssertionError("Configuration must not assemble runtime resources")
    )
    monkeypatch.setattr(
        "llm_management.bootstrap.register_configured_resources", register
    )
    get_catalog.cache_clear()
    get_catalog()
    register.assert_not_called()


def test_catalog_loading_does_not_construct_compute_runtime(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from llm_management.deployments import backends
    from llm_management.settings import settings

    state_dir = tmp_path / "compute-state"
    monkeypatch.setattr(settings, "compute_state_dir", state_dir)
    factory = Mock(
        side_effect=AssertionError("Configuration must not construct a backend")
    )
    monkeypatch.setattr(backends, "backend_for", factory)

    catalog = DeploymentCatalog.load()

    assert catalog.get("clef").model
    assert not state_dir.exists()
    factory.assert_not_called()


def test_separate_catalogs_and_registries_warm_their_own_checkpoint(monkeypatch):
    first = DeploymentCatalog.load()
    data = first.model_dump()
    reference = first.local.get(
        first.require_sar().logistic,
        SARLogisticDeployment,
    ).model_ref
    data["model"][reference]["repo"] = "example/independent-sar"
    second = DeploymentCatalog.model_validate(data)
    first_registry, second_registry = ResourceRegistry(), ResourceRegistry()
    first_application = assemble_application(first, first_registry)
    second_application = assemble_application(second, second_registry)
    first_service = first_application.deployments
    second_service = second_application.deployments
    assert first_service.cache is not second_service.cache
    assert first_registry.all() == second_registry.all() == []

    # Replace actual artifact loading, retaining the real factory and owner path.
    monkeypatch.setattr(LogisticDetector, "load", lambda self: self.checkpoint.repo)
    slug = first.require_sar().logistic
    first_guard = first_registry.get(slug)
    second_guard = second_registry.get(slug)
    assert first_guard is not second_guard
    assert first_application.sar.logistic_provider().resource is first_guard
    assert second_application.sar.logistic_provider().resource is second_guard
    assert first_guard.warmup() == first.model[reference].repo
    assert second_guard.warmup() == "example/independent-sar"
    assert first_registry.get(slug) is first_guard
    assert second_registry.get(slug) is second_guard
    assert first_guard.release()
    assert not first_guard.status().ready
    assert second_guard.status().ready
