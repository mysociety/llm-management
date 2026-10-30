import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from starlette.testclient import TestClient

from llm_management import server
from llm_management.inference.classifiers import LocalSequenceClassifier
from llm_management.inference.resources import LocalResource, ResourceRegistry


def test_shutdown_continues_after_local_unload_failure(caplog):
    registry = ResourceRegistry()
    broken = registry.register(
        LocalResource("broken", Mock(), Mock(side_effect=RuntimeError("unload failed")))
    )
    unload = Mock()
    healthy = registry.register(LocalResource("healthy", Mock(), unload))
    broken.warmup()
    healthy.warmup()

    registry.close()

    unload.assert_called_once()
    assert not healthy.status().ready
    assert broken.status().ready
    assert "Failed to release local resource broken" in caplog.text


def test_local_cleanup_failure_does_not_skip_remote_shutdown(monkeypatch):
    registry = Mock()
    registry.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(server.app.state.service, "resources", registry)
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "token"})
    monkeypatch.setattr(
        server.app.state.service,
        "active_deployments",
        lambda: [SimpleNamespace(slug="remote")],
    )
    cfg = SimpleNamespace(scale_to_zero=Mock())
    monkeypatch.setattr(server.app.state.service, "get_config", lambda slug: cfg)
    server.app.state.service.backends["remote"] = SimpleNamespace(
        config=cfg, scale_to_zero=cfg.scale_to_zero
    )

    async def run():
        async with server.lifespan(server.app):
            pass

    asyncio.run(run())
    cfg.scale_to_zero.assert_called_once()


def test_warmup_expiry_reload_and_touch(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "llm_management.inference.resources.time.monotonic", lambda: now[0]
    )
    load, unload = Mock(), Mock()
    resource = LocalResource("test", load, unload)
    registry = ResourceRegistry()
    registry.register(resource)
    resource.warmup()
    assert resource.status().ready
    now[0] = 10
    registry.expire(20)
    unload.assert_not_called()
    resource.warmup()  # Explicit ensure restarts the idle countdown.
    now[0] = 25
    registry.expire(20)
    unload.assert_not_called()
    now[0] = 31
    registry.expire(20)
    unload.assert_called_once()
    assert not resource.status().ready
    resource.warmup()
    assert resource.status().ready
    registry.close()
    assert unload.call_count == 2


def test_thread_lease_survives_async_cancellation():
    entered, release = threading.Event(), threading.Event()
    unload = Mock()
    resource = LocalResource("test", Mock(), unload)

    def work():
        with resource.use():
            entered.set()
            assert release.wait(5)

    async def run():
        task = asyncio.create_task(asyncio.to_thread(work))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert resource.status().busy
            assert not resource.release()
            unload.assert_not_called()
        finally:
            release.set()

    asyncio.run(run())
    assert resource.release()
    unload.assert_called_once()


def test_classifier_unloads_model_and_tokenizer_under_lease():
    classifier = LocalSequenceClassifier(model="test", revision="test", labels=("A",))
    classifier._model = object()
    classifier._tokenizer = object()
    with classifier.resource.use():
        classifier.resource.mark_ready()
        assert not classifier.resource.release()
    assert classifier.resource.release()
    assert classifier._model is None and classifier._tokenizer is None


def test_cpu_ensure_status_and_failure(monkeypatch):
    sanitizer = server.app.state.application.sanitizer()
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(server.app.state.service, "active_deployments", lambda: [])
    with TestClient(server.app) as client:
        assert client.post("/deployments/presidio/warm-up").json()["action"] == "warmed"
        rows = client.get("/local-deployments").json()
        assert any(row["name"] == "presidio" and row["ready"] for row in rows)
        assert client.post("/deployments/missing/warm-up").status_code == 404
        monkeypatch.setattr(
            sanitizer.resource, "_load", Mock(side_effect=RuntimeError("load failed"))
        )
        assert client.post("/deployments/presidio/warm-up").status_code == 503
    assert not sanitizer.resource.status().ready


def test_registry_cold_factory_discovery_and_concurrent_lookup():
    registry = ResourceRegistry()
    load = Mock()
    factory = Mock(side_effect=lambda: LocalResource("cold", load, Mock()))
    registry.register_factory("cold", factory)
    assert registry.names() == {"cold"}
    assert registry.all() == []
    factory.assert_not_called()
    assert registry.get("missing") is None
    results = []
    workers = [
        threading.Thread(target=lambda: results.append(registry.get("cold")))
        for _ in range(3)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
        assert not worker.is_alive()
    factory.assert_called_once()
    load.assert_not_called()
    assert all(resource is results[0] for resource in results)
    results[0].warmup()
    load.assert_called_once()


def test_local_cooldown_expiry_and_new_work_reset(monkeypatch):
    from llm_management.inference import resources

    clock = [1000.0]
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    unload = Mock()
    resource = LocalResource("local", Mock(), unload)
    assert resource.cooldown(900) is None
    resource.stop()  # Already cold is a successful no-op.
    unload.assert_not_called()
    resource.warmup()
    assert resource.cooldown(900) == 60
    clock[0] += 30
    assert resource.cooldown(900) == 30  # Never extend an earlier deadline.
    assert not resource.release(900)
    with resource.use():
        pass
    clock[0] += 31
    assert not resource.release(900)  # New work cancels the shortened deadline.
    assert resource.cooldown(900) == 60
    clock[0] += 60
    assert resource.release(900)
    unload.assert_called_once()
    resource.warmup()
    clock[0] += 890
    assert resource.cooldown(900) == 10
    clock[0] += 10
    assert resource.release(900)


def test_local_cooldown_clamps_timestamp_rounding(monkeypatch):
    from llm_management.inference import resources

    clock = [4.01]
    assert (clock[0] + 60) - clock[0] > 60  # Reproduce the CI rounding boundary.
    monkeypatch.setattr(resources.time, "monotonic", lambda: clock[0])
    resource = LocalResource("local", Mock(), Mock())
    resource.warmup()
    assert resource.cooldown(900) == 60.0
    clock[0] += 30
    assert 0 < resource.cooldown(900) < 60
    clock[0] += 31
    assert resource.cooldown(900) == 0.0
    assert resource.release(900)
