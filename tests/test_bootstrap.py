"""Cold assembly and model ownership across independent applications."""

from llm_management.bootstrap import assemble_application
from llm_management.deployments.catalog import DeploymentCatalog
from llm_management.deployments.local_config import SARLogisticDeployment
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
    assert first_guard is not None
    assert second_guard is not None
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


def test_http_dependencies_use_the_serving_application(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from starlette.testclient import TestClient

    from llm_management.api.dependencies import get_service
    from llm_management.server import create_app
    from llm_management.settings import settings

    monkeypatch.setattr(settings, "auth_tokens", {})
    first, second = assemble_application(), assemble_application()
    apps = [create_app(first), create_app(second)]
    calls = []
    for application, label in [(first, "first"), (second, "second")]:
        call = AsyncMock(
            return_value=SimpleNamespace(slug="example", action=label, replicas=1)
        )
        monkeypatch.setattr(application.deployments, "ensure_deployment", call)
        calls.append(call)

    # No lifespan needed: mocked lifecycle actions never load resources.
    clients = [TestClient(app) for app in apps]
    for client, label in zip(clients, ["first", "second"]):
        assert client.post("/deployments/example/warm-up").json()["action"] == label
    for call in calls:
        call.assert_awaited_once_with("example", None)

    apps[0].dependency_overrides[get_service] = lambda: second.deployments
    assert clients[0].post("/deployments/example/warm-up").json()["action"] == "second"
    assert clients[1].post("/deployments/example/warm-up").json()["action"] == "second"
    assert calls[0].await_count == 1
    assert calls[1].await_count == 3
    for client in clients:
        client.close()


def test_application_owns_shared_crawl_policy(api_catalog: DeploymentCatalog) -> None:
    first = assemble_application(api_catalog)
    second = assemble_application(api_catalog)

    assert first.foi_finder.crawl_policy is first.crawl_policy
    assert second.foi_finder.crawl_policy is second.crawl_policy
    assert first.crawl_policy is not second.crawl_policy


def test_management_assembly_does_not_import_features():
    import subprocess
    import sys
    import textwrap

    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import importlib.abc
            import sys

            class NoFeatures(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith((
                        "llm_management.foi", "llm_management.sar",
                        "llm_management.automatic", "llm_management.sanitization",
                    )):
                        raise AssertionError(fullname)

            sys.meta_path.insert(0, NoFeatures())
            from llm_management.bootstrap import assemble_service
            service = assemble_service()
            assert service.resources.names()
            assert service.resources.all() == []
            assert service.backends == {}
        """),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
