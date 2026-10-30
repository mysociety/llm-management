"""Cold assembly and model ownership across independent applications."""

from llm_management.bootstrap import assemble_runtime
from llm_management.deployments.catalog import DeploymentCatalog
from llm_management.deployments.local_config import SARLogisticDeployment
from llm_management.inference.resources import ResourceRegistry
from llm_management.sar.resources import LogisticDetector


def test_catalog_loading_does_not_register_resources(monkeypatch):
    from unittest.mock import Mock

    from llm_management.deployments.catalog import get_catalog

    providers = Mock(
        side_effect=AssertionError("Configuration must not assemble runtime resources")
    )
    monkeypatch.setattr("llm_management.bootstrap.ResourceProviders", providers)
    get_catalog.cache_clear()
    get_catalog()
    providers.assert_not_called()


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
    first_runtime = assemble_runtime(first, first_registry)
    second_runtime = assemble_runtime(second, second_registry)
    first_service = first_runtime.deployments
    second_service = second_runtime.deployments
    assert first_service.status is not second_service.status
    assert first_registry.all() == second_registry.all() == []

    # Replace actual artifact loading, retaining the real factory and owner path.
    monkeypatch.setattr(LogisticDetector, "load", lambda self: self.checkpoint.repo)
    slug = first.require_sar().logistic
    first_guard = first_registry.get(slug)
    second_guard = second_registry.get(slug)
    assert first_guard is not None
    assert second_guard is not None
    assert first_guard is not second_guard
    assert first_runtime.sar.logistic_provider().resource is first_guard
    assert second_runtime.sar.logistic_provider().resource is second_guard
    assert first_guard.warmup() == first.model[reference].repo
    assert second_guard.warmup() == "example/independent-sar"
    assert first_registry.get(slug) is first_guard
    assert second_registry.get(slug) is second_guard
    assert first_guard.release()
    assert not first_guard.status().ready
    assert second_guard.status().ready


def test_http_dependencies_use_the_serving_runtime(monkeypatch):
    from unittest.mock import AsyncMock

    from starlette.testclient import TestClient

    from llm_management.api import get_runtime
    from llm_management.deployments.types import WarmUpAction, WarmUpResponse
    from llm_management.server import create_app
    from llm_management.settings import settings

    monkeypatch.setattr(settings, "auth_tokens", {})
    first, second = assemble_runtime(), assemble_runtime()
    apps = [create_app(runtime=first), create_app(runtime=second)]
    calls = []
    for runtime, label in [(first, "first"), (second, "second")]:
        call = AsyncMock(
            return_value=WarmUpResponse(
                slug=label, action=WarmUpAction.WARMED, replicas=1
            )
        )
        monkeypatch.setattr(runtime.deployments.management, "warm_up_resource", call)
        calls.append(call)

    # No lifespan needed: mocked lifecycle actions never load resources.
    clients = [TestClient(app) for app in apps]
    for client, label in zip(clients, ["first", "second"]):
        assert client.post("/deployments/example/warm-up").json()["slug"] == label
    for call in calls:
        call.assert_awaited_once_with("example", None)

    apps[0].dependency_overrides[get_runtime] = lambda: second
    assert clients[0].post("/deployments/example/warm-up").json()["slug"] == "second"
    assert clients[1].post("/deployments/example/warm-up").json()["slug"] == "second"
    assert calls[0].await_count == 1
    assert calls[1].await_count == 3
    for client in clients:
        client.close()


def test_runtime_owns_shared_crawl_policy(api_catalog: DeploymentCatalog) -> None:
    first = assemble_runtime(api_catalog)
    second = assemble_runtime(api_catalog)

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


def test_application_lifecycle_orders_cleanup_and_closes_once(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    runtime = assemble_runtime()
    events = []

    async def idle():
        events.append("idle started")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("idle stopped")

    async def sar_close():
        events.append("sar drained")

    async def automatic_close():
        events.append("automatic drained")

    async def deployments_close():
        events.append("resources released")

    monkeypatch.setattr(runtime.deployments, "idle_scaler", idle)
    monkeypatch.setattr(runtime.sar, "close", AsyncMock(side_effect=sar_close))
    monkeypatch.setattr(
        runtime.automatic, "close", AsyncMock(side_effect=automatic_close)
    )
    monkeypatch.setattr(
        runtime.deployments, "close", AsyncMock(side_effect=deployments_close)
    )

    async def run():
        async with runtime as entered:
            assert entered is runtime
            await asyncio.sleep(0)
        await asyncio.gather(runtime.close(), runtime.close())

    asyncio.run(run())
    assert events == [
        "idle started",
        "idle stopped",
        "sar drained",
        "automatic drained",
        "resources released",
    ]


def test_application_shutdown_continues_after_classifier_failure(monkeypatch):
    import asyncio
    from builtins import ExceptionGroup
    from unittest.mock import AsyncMock

    import pytest

    runtime = assemble_runtime()
    monkeypatch.setattr(
        runtime.sar, "close", AsyncMock(side_effect=RuntimeError("drain failed"))
    )
    automatic_close = AsyncMock()
    deployments_close = AsyncMock()
    monkeypatch.setattr(runtime.automatic, "close", automatic_close)
    monkeypatch.setattr(runtime.deployments, "close", deployments_close)

    with pytest.raises(ExceptionGroup, match="Application shutdown failed") as caught:
        asyncio.run(runtime.close())
    assert str(caught.value.exceptions[0]) == "drain failed"
    automatic_close.assert_awaited_once()
    deployments_close.assert_awaited_once()


def test_cancelled_close_waits_for_resource_cleanup(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    import pytest

    runtime = assemble_runtime()
    monkeypatch.setattr(runtime.automatic, "close", AsyncMock())
    release_resources = AsyncMock()
    monkeypatch.setattr(runtime.deployments, "close", release_resources)

    async def run():
        draining = asyncio.Event()
        finish = asyncio.Event()

        async def drain():
            draining.set()
            await finish.wait()

        monkeypatch.setattr(runtime.sar, "close", drain)
        shutdown = asyncio.create_task(runtime.close())
        await draining.wait()
        shutdown.cancel()
        await asyncio.sleep(0)
        assert not shutdown.done()
        shutdown.cancel()
        await asyncio.sleep(0)
        assert not shutdown.done()
        release_resources.assert_not_awaited()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        await runtime.close()
        release_resources.assert_awaited_once()

    asyncio.run(run())


def test_application_context_failure_cleans_up_and_rejects_reuse(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    import pytest

    runtime = assemble_runtime()
    release_resources = AsyncMock()
    monkeypatch.setattr(runtime.deployments, "close", release_resources)

    async def run():
        with pytest.raises(ValueError, match="caller failed"):
            async with runtime:
                with pytest.raises(RuntimeError, match="already started"):
                    async with runtime:
                        pass
                raise ValueError("caller failed")
        release_resources.assert_awaited_once()
        with pytest.raises(RuntimeError, match="already started or closed"):
            async with runtime:
                pass

    asyncio.run(run())
