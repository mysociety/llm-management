"""Deployment scaling contracts without contacting Exoscale."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.deployments.config import ExoscaleDeploymentConfig
from llm_management.deployments.errors import DeploymentServiceError
from llm_management.deployments.managed import ManagedDeployment
from llm_management.deployments.status import DeploymentStatusCache
from llm_management.deployments.types import DeploymentActivity, DeploymentSnapshot
from llm_management.inference.resources import LocalResource, ResourceRegistry


@pytest.fixture
def deployment(monkeypatch):
    cfg = ExoscaleDeploymentConfig(
        slug="test",
        model="test-model",
        gpu_type="test-gpu",
        gpu_count=1,
        replicas=2,
        zone="test-zone",
    )
    cloud: dict[str, dict[str, Any] | None] = {"deployment": None}
    client = Mock()
    client.list_deployments.side_effect = lambda: {
        "deployments": [cloud["deployment"]] if cloud["deployment"] else []
    }

    def create(**kwargs: Any) -> dict[str, str]:
        """
        Retain the heterogeneous SDK creation payload as simulated cloud state.
        """
        cloud["deployment"] = {"id": "test-id", **kwargs}
        return {"id": "operation"}

    def scale(**kwargs: Any) -> dict[str, str]:
        """
        Update the simulated existing deployment's replica count.
        """
        deployed = cloud["deployment"]
        assert deployed is not None
        deployed["replicas"] = kwargs["replicas"]
        return {"id": "operation"}

    client.create_deployment.side_effect = create
    client.scale_deployment.side_effect = scale
    client.wait.return_value = {}
    client.reveal_deployment_api_key.side_effect = AssertionError(
        "Deployment credentials must not be queried"
    )
    monkeypatch.setattr(server.settings, "exoscale_inference_api_key", "test-secret")
    monkeypatch.setattr(ManagedDeployment, "client", lambda self: client)
    monkeypatch.setattr(ManagedDeployment, "ensure_model", lambda *a, **kw: None)
    monkeypatch.setattr(server.settings, "server_role", "")
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(
        server.app.state.application.deployments, "get_config", lambda slug: cfg
    )
    monkeypatch.setattr(
        server.app.state.application.deployments, "resources", ResourceRegistry()
    )
    cache = DeploymentStatusCache()
    monkeypatch.setattr(server.app.state.application.deployments, "status", cache)
    return TestClient(server.app), cfg, cloud, client, cache


@pytest.mark.parametrize(
    "initial,expected_action",
    [(None, "created"), (0, "resumed"), (1, "scaled"), (3, "already_running")],
)
def test_custom_replicas_create_resume_and_resize(deployment, initial, expected_action):
    http, cfg, cloud, client, cache = deployment
    if initial is not None:
        cloud["deployment"] = {"id": "test-id", "name": "test", "replicas": initial}
    response = http.post("/deployments/test/warm-up?replicas=3")
    assert response.status_code == 200
    assert response.json() == {
        "slug": "test",
        "replicas": 3,
        "action": expected_action,
    }
    assert cfg.replicas == 2
    assert cache.get("test").replicas == 3
    assert client.create_deployment.call_count == int(initial is None)
    assert client.scale_deployment.call_count == int(initial in (0, 1))


@pytest.mark.parametrize("initial,expected", [(None, 2), (0, 2), (3, 3)])
def test_omitted_replicas_preserves_defaults_and_running_count(
    deployment, initial, expected
):
    http, _, cloud, _, _ = deployment
    if initial is not None:
        cloud["deployment"] = {"id": "test-id", "name": "test", "replicas": initial}
    response = http.post("/deployments/test/warm-up")
    assert response.status_code == 200
    assert response.json()["replicas"] == expected


@pytest.mark.parametrize("replicas", ["0", "-1", "1.5", "no"])
def test_invalid_replicas_rejected_before_startup(deployment, replicas):
    http, _, _, client, _ = deployment
    assert (
        http.post(f"/deployments/test/warm-up?replicas={replicas}").status_code == 422
    )
    client.create_deployment.assert_not_called()


def test_compute_rejects_multiple_replicas(monkeypatch):
    cfg = SimpleNamespace(backend="exoscale_compute", create_or_resume=Mock())
    monkeypatch.setattr(
        server.app.state.application.deployments, "get_config", lambda slug: cfg
    )
    with pytest.raises(DeploymentServiceError) as error:
        asyncio.run(
            server.app.state.application.deployments.prepare(
                "compute", allow_start=True, replicas=2
            )
        )
    assert error.value.status_code == 422
    cfg.create_or_resume.assert_not_called()


def test_local_resource_rejects_replicas(deployment, monkeypatch):
    http, _, _, _, _ = deployment
    registry = ResourceRegistry()
    warm = Mock()
    registry.register(LocalResource("local", warm, Mock()))
    monkeypatch.setattr(server.app.state.application.deployments, "resources", registry)
    assert http.post("/deployments/local/warm-up?replicas=2").status_code == 422
    warm.assert_not_called()


@pytest.mark.parametrize("remaining", [900, 30, -10, None])
def test_cooldown_shortens_timer_and_traffic_resets_it(
    deployment, monkeypatch, remaining
):
    http, _, _, client, cache = deployment
    monkeypatch.setattr(time, "time", lambda: 10_000.0)
    state = DeploymentSnapshot(
        slug="test",
        exists=True,
        replicas=2,
        last_refreshed=10_000,
    )
    cache.set(state)
    server.app.state.application.deployments.activity.set(
        "test",
        DeploymentActivity(
            last_request_time=0 if remaining is None else 10_000 - 900 + remaining,
        ),
    )
    response = http.post("/deployments/test/cool-down")
    assert response.status_code == 200
    expected = 60 if remaining is None else max(0, min(60, remaining))
    assert response.json()["seconds_until_scale_to_zero"] == expected
    assert (
        900
        - (
            10_000
            - server.app.state.application.deployments.activity.get(
                "test"
            ).last_request_time
        )
        <= 60
    )
    client.scale_deployment.assert_not_called()
    server.app.state.application.deployments.activity.touch("test")
    assert (
        server.app.state.application.deployments.activity.get("test").last_request_time
        == 10_000
    )


@pytest.mark.parametrize("exists", [False, True])
def test_cooldown_stopped_deployment_does_not_start_it(deployment, exists):
    http, _, _, client, cache = deployment
    cache.set(
        DeploymentSnapshot(slug="test", exists=exists, last_refreshed=time.time())
    )
    response = http.post("/deployments/test/cool-down")
    assert response.json()["action"] == "already_stopped"
    assert response.json()["seconds_until_scale_to_zero"] is None
    client.create_deployment.assert_not_called()


def test_cooldown_rejects_active_requests(deployment):
    http, _, _, _, cache = deployment
    cache.set(
        DeploymentSnapshot(
            slug="test",
            exists=True,
            replicas=1,
            last_refreshed=time.time(),
        )
    )
    server.app.state.application.deployments.activity.begin_request("test")
    before = server.app.state.application.deployments.activity.get(
        "test"
    ).last_request_time
    assert http.post("/deployments/test/cool-down").status_code == 409
    assert (
        server.app.state.application.deployments.activity.get("test").last_request_time
        == before
    )


@pytest.mark.parametrize("elapsed,stopped", [(59, False), (61, True)])
def test_idle_scaler_observes_cooldown(deployment, monkeypatch, elapsed, stopped):
    http, _, cloud, client, cache = deployment
    now = [10_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    cloud["deployment"] = {"id": "test-id", "name": "test", "replicas": 2}
    assert http.post("/deployments/test/cool-down").status_code == 200
    # Refreshes must preserve the shortened idle timer.
    cache.refresh(server.app.state.application.deployments.get_backend("test"))
    now[0] += elapsed
    sleeps = 0

    async def sleep_once(seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", sleep_once)
    monkeypatch.setattr(
        server.app.state.application.deployments,
        "resources",
        SimpleNamespace(expire=Mock()),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.app.state.application.deployments.idle_scaler())
    assert client.scale_deployment.call_count == int(stopped)
    assert cache.get("test").replicas == (0 if stopped else 2)


def test_managed_connection_uses_configured_inference_key(deployment):
    """
    Preserve the cloud endpoint and attach the configured key without revelation.
    """
    _, cfg, cloud, client, _ = deployment
    cloud["deployment"] = {
        "id": "test-id",
        "name": cfg.deployment_name,
        "replicas": 1,
        "deployment-url": "https://inference.example/v1/",
    }
    backend = ManagedDeployment(cfg)

    connection = backend.connection_info()
    status = backend.query_status()

    assert connection.url == status.deployment_url == "https://inference.example/v1/"
    assert connection.api_key == status.api_key == "test-secret"
    assert status.exists and status.replicas == 1
    client.reveal_deployment_api_key.assert_not_called()


def test_missing_inference_key_keeps_status_readable(deployment, monkeypatch):
    """
    Report cloud status without a key and reject inference connection construction.
    """
    from llm_management.deployments.errors import LLMManagementError

    _, cfg, cloud, client, _ = deployment
    cloud["deployment"] = {"id": "test-id", "name": cfg.deployment_name, "replicas": 1}
    monkeypatch.setattr(server.settings, "exoscale_inference_api_key", "")
    backend = ManagedDeployment(cfg)

    status = backend.query_status()

    assert status.exists and status.api_key == ""
    with pytest.raises(LLMManagementError, match="EXOSCALE_INFERENCE_API_KEY"):
        backend.connection_info()
    client.reveal_deployment_api_key.assert_not_called()


@pytest.mark.parametrize("version", [None, "0.29.0"])
def test_managed_creation_preserves_engine_selection(deployment, version):
    """
    Forward an explicit runtime pin while retaining provider defaults when omitted.
    """
    _, cfg, _, client, _ = deployment
    cfg.inference_engine_version = version
    ManagedDeployment(cfg).create_deployment()
    parameters = client.create_deployment.call_args.kwargs
    if version is None:
        assert "inference_engine_version" not in parameters
    else:
        assert parameters["inference_engine_version"] == version
