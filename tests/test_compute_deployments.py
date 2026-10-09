"""Compute lifecycle integration without provisioning paid cloud resources."""

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management import compute_deployments as module, server
from llm_management.cache import DeploymentCache
from llm_management.models import (
    ComputeDeploymentConfig,
    ExoscaleConfig,
    LLMManagementError,
)
from llm_management.templates import TemplateConfig


class Cloud:
    def __init__(self, recipe):
        self.recipe = recipe
        self.instances = [{"name": "unrelated", "id": "other", "state": "running"}]
        self.keys = []
        self.groups = []
        self.created = []
        self.deleted = []
        self.fail_delete = False

    def list_instances(self):
        return {"instances": self.instances}

    def list_templates(self, visibility):
        assert visibility == "private"
        return {
            "templates": [
                {"id": "prepared", "name": self.recipe.name, "default-user": "ubuntu"}
            ]
        }

    def list_instance_types(self):
        return {
            "instance-types": [
                {
                    "id": "gpu",
                    "family": "gpua5000",
                    "size": "small",
                    "zones": [self.recipe.zone],
                    "authorized": True,
                }
            ]
        }

    def list_ssh_keys(self):
        return {"ssh-keys": self.keys}

    def list_security_groups(self):
        return {"security-groups": self.groups}

    def register_ssh_key(self, **kw):
        self.keys.append(kw)
        return {"id": "key-op"}

    def create_security_group(self, **kw):
        self.groups.append(dict(kw, id="group"))
        return {"id": "group-op"}

    def add_rule_to_security_group(self, **kw):
        assert kw["network"] == "192.0.2.1/32"
        assert kw["start_port"] == kw["end_port"] == 22
        return {"id": "rule-op"}

    def create_instance(self, **kw):
        self.created.append(kw)
        self.instances.append({"id": "vm", "name": kw["name"], "state": "running"})
        return {"id": "vm-op"}

    def get_instance(self, id):
        return {"id": id, "public-ip": "192.0.2.2"}

    def wait(self, operation_id, max_wait_time):
        return {}

    def delete_instance(self, id):
        if self.fail_delete:
            raise RuntimeError("Delete failed")
        self.instances = [i for i in self.instances if i["id"] != id]
        self.deleted.append(id)
        return {"id": "delete-op"}

    def delete_security_group(self, id):
        self.groups = [g for g in self.groups if g["id"] != id]
        self.deleted.append(id)
        return {"id": "delete-group"}

    def delete_ssh_key(self, name):
        self.keys = [k for k in self.keys if k["name"] != name]
        self.deleted.append(name)
        return {"id": "delete-key"}

    def start_instance(self, id):
        next(i for i in self.instances if i["id"] == id)["state"] = "running"
        return {"id": "start-op"}


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    recipe = (
        TemplateConfig.load()
        .get("clef_flash")
        .model_copy(update={"ssh_cidr": "192.0.2.1/32"})
    )
    config = ComputeDeploymentConfig(
        slug="clef", backend="exoscale_compute", template="clef_flash"
    )
    config.__dict__["recipe"] = recipe
    monkeypatch.setattr(module.settings, "compute_state_dir", tmp_path)
    cloud = Cloud(recipe)
    monkeypatch.setattr(module, "get_client", lambda zone: cloud)
    adapter = config.adapter
    processes = []

    def run(command, **kwargs):
        if command[0] == "ssh-keygen":
            key = Path(command[-1])
            key.write_text("private-test-key")
            key.with_suffix(".pub").write_text("public-test-key")
        return SimpleNamespace(returncode=0, stdout="logs", stderr="")

    def popen(command, **kwargs):
        process = Mock()
        process.poll.return_value = None
        process.command = command
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(adapter, "health", lambda url: True)
    yield config, adapter, cloud, processes
    module.close_tunnel(adapter.runtime_key)


def test_config_resolves_recipe_and_rejects_managed_options():
    cfg = ExoscaleConfig.load().get("clef")
    assert isinstance(cfg, ComputeDeploymentConfig)
    assert cfg.template == "clef_flash"
    assert cfg.model == cfg.recipe.model
    assert cfg.zone == cfg.recipe.zone
    with pytest.raises(ValidationError):
        ComputeDeploymentConfig(
            slug="clef", backend="exoscale_compute", template="clef_flash", gpu_count=1
        )


def test_provision_ready_connection_and_delete(deployed):
    config, adapter, cloud, processes = deployed
    assert not adapter.status().exists
    config.create_or_resume()
    assert len(cloud.created) == 1
    request = cloud.created[0]
    assert request["template"] == {"id": "prepared"}
    assert request["disk_size"] == config.recipe.disk_size_gib
    assert base64.b64decode(request["user_data"]) == b"#cloud-config\n{}"
    assert request["name"] == "llm-compute-at-vie-2-clef_test"
    status = config.query_status()
    assert status.exists and status.replicas == 1
    assert status.deployment_url.startswith("http://127.0.0.1:")
    assert status.deployment_url.endswith("/v1")
    assert status.api_key == ""
    config.create_or_resume()
    assert len(cloud.created) == len(processes) == 1
    config.scale_to_zero()
    assert not cloud.keys and not cloud.groups
    assert cloud.instances == [{"name": "unrelated", "id": "other", "state": "running"}]
    processes[0].terminate.assert_called_once()
    assert not config.query_status().exists
    assert json.loads(adapter.path.read_text())["cleanup_complete"]
    assert not (adapter.directory / "id_ed25519").exists()
    config.scale_to_zero()  # Idempotent teardown.


def test_reconnect_after_manager_restart_without_creating_another_vm(
    deployed, monkeypatch
):
    config, adapter, cloud, processes = deployed
    config.create_or_resume()
    module.close_tunnel(adapter.runtime_key)
    restarted = module.ComputeDeployment(config)
    monkeypatch.setattr(restarted, "health", lambda url: True)
    assert restarted.status().exists and restarted.status().replicas == 0
    restarted.ensure()
    assert restarted.status().replicas == 1
    assert len(cloud.created) == 1 and len(processes) == 2


def test_failed_startup_deletes_new_vm_and_preserves_template(deployed, monkeypatch):
    config, adapter, cloud, processes = deployed
    monkeypatch.setattr(
        adapter, "health", Mock(side_effect=LLMManagementError("Wrong revision"))
    )
    with pytest.raises(LLMManagementError, match="Wrong revision"):
        config.create_or_resume()
    assert [i["id"] for i in cloud.instances] == ["other"]
    assert not cloud.keys and not cloud.groups
    assert cloud.list_templates("private")["templates"][0]["id"] == "prepared"
    assert adapter.runtime_key not in module._runtimes


def test_failed_vm_deletion_retains_access_and_state_for_retry(deployed):
    config, adapter, cloud, _ = deployed
    config.create_or_resume()
    cloud.fail_delete = True
    with pytest.raises(RuntimeError, match="Delete failed"):
        config.scale_to_zero()
    assert cloud.keys and cloud.groups
    assert (adapter.directory / "id_ed25519").exists()
    cloud.fail_delete = False
    config.scale_to_zero()
    assert not cloud.keys and not cloud.groups


def test_missing_private_key_for_existing_vm_does_not_create_duplicate(deployed):
    config, adapter, cloud, _ = deployed
    config.create_or_resume()
    module.close_tunnel(adapter.runtime_key)
    (adapter.directory / "id_ed25519").unlink()
    with pytest.raises(LLMManagementError, match="SSH key missing"):
        config.create_or_resume()
    assert len(cloud.created) == 1
    config.scale_to_zero()


def test_health_rejects_wrong_cached_revision(deployed, monkeypatch):
    config, adapter, _, _ = deployed
    monkeypatch.delattr(adapter, "health")
    payload = {
        "backend": "clef",
        "model": config.model,
        "max_length": config.recipe.max_length,
        "revision": "wrong",
    }
    monkeypatch.setattr(
        module.httpx, "get", lambda *a, **kw: httpx.Response(200, json=payload)
    )
    with pytest.raises(LLMManagementError, match="revision"):
        adapter.health("http://localhost:8000")


def test_server_ensure_dispatches_compute_once_and_pauses_by_deletion(
    deployed, monkeypatch
):
    config, adapter, cloud, _ = deployed
    cache = DeploymentCache()
    monkeypatch.setattr(server, "cache", cache)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: config)

    async def run():
        results = await asyncio.gather(
            server.ensure_running("clef"), server.ensure_running("clef")
        )
        assert all(state.replicas == 1 for _, state in results)
        assert len(cloud.created) == 1
        await server.scale_to_zero("clef")
        assert cache.get("clef").exists is False

    asyncio.run(run())
    assert not cloud.keys and not cloud.groups


def test_http_request_lease_prevents_teardown_and_releases_after_failure(
    deployed, monkeypatch
):
    config, _, _, _ = deployed
    cache = DeploymentCache()
    monkeypatch.setattr(server, "cache", cache)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: config)
    monkeypatch.setattr(server.settings, "auth_tokens", {})

    async def send(*args, **kwargs):
        assert cache.get("clef").requests_in_flight == 1
        with pytest.raises(server.HTTPException) as error:
            await server.scale_to_zero("clef")
        assert error.value.status_code == 409
        raise server.systemone.SystemOneUnavailable("Test failure")

    monkeypatch.setattr(server.systemone, "send_systemone", send)
    # No lifespan: avoid idle/shutdown activity in this HTTP lease test.
    response = TestClient(server.app).post("/v1/systemone", json={})
    assert response.status_code == 503
    assert cache.get("clef").requests_in_flight == 0
    assert cache.get("clef").last_request_time > 0


def test_cancelled_startup_is_tracked_for_idle_cleanup(monkeypatch):
    import threading
    from llm_management.models import DeploymentQueryResult

    cache = DeploymentCache()
    entered = threading.Event()
    finish = threading.Event()
    ready = False

    def start():
        nonlocal ready
        entered.set()
        assert finish.wait(5)
        ready = True

    cfg = SimpleNamespace(
        slug="clef",
        create_or_resume=start,
        query_status=lambda: DeploymentQueryResult(
            exists=ready,
            replicas=int(ready),
            deployment_url="http://localhost/v1",
            api_key="",
        ),
    )
    monkeypatch.setattr(server, "cache", cache)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: cfg)

    async def run():
        task = asyncio.create_task(server.ensure_running("clef"))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cache.all_active()[0].slug == "clef"

    asyncio.run(run())


@pytest.mark.parametrize("busy", [False, True])
def test_idle_loop_deletes_compute_only_after_requests_finish(
    deployed, monkeypatch, busy
):
    import time

    config, _, cloud, _ = deployed
    config.create_or_resume()
    cache = DeploymentCache()
    cache.refresh(config)
    state = cache.get("clef")
    state.last_request_time = time.time() - 3600
    state.requests_in_flight = int(busy)
    monkeypatch.setattr(server, "cache", cache)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: config)
    sleeps = 0

    async def sleep_once(seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(server.asyncio, "sleep", sleep_once)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await server.idle_scaler()

    asyncio.run(run())
    assert any(i["id"] == "vm" for i in cloud.instances) is busy


def test_unready_compute_is_still_tracked_for_teardown(deployed):
    config, adapter, cloud, _ = deployed
    config.create_or_resume()
    module.close_tunnel(adapter.runtime_key)
    cache = DeploymentCache()
    cache.refresh(config)
    cache.touch("clef")
    assert cache.get("clef").replicas == 0
    assert cache.all_active()[0].slug == "clef"
    config.scale_to_zero()
    cache.refresh(config)
    assert cache.all_active() == []


def test_changed_template_refuses_to_reuse_old_vm(deployed, monkeypatch):
    config, adapter, cloud, _ = deployed
    config.create_or_resume()
    monkeypatch.setattr(
        cloud,
        "list_templates",
        lambda **kw: {"templates": [{"name": config.recipe.name, "id": "new-release"}]},
    )
    assert config.query_status().replicas == 0
    with pytest.raises(LLMManagementError, match="different template release"):
        config.create_or_resume()
    assert len(cloud.created) == 1


def test_shutdown_tears_down_compute(deployed, monkeypatch):
    config, _, cloud, _ = deployed
    config.create_or_resume()
    cache = DeploymentCache()
    cache.refresh(config)
    cache.touch("clef")
    monkeypatch.setattr(server, "cache", cache)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: config)

    async def run():
        async with server.lifespan(server.app):
            pass

    asyncio.run(run())
    assert [i["id"] for i in cloud.instances] == ["other"]
    assert not cloud.keys and not cloud.groups


def test_pending_creation_is_settled_before_cleanup(deployed, monkeypatch):
    _, adapter, cloud, _ = deployed
    state = {
        "name": adapter.name,
        "zone": adapter.recipe.zone,
        "create_operation": "pending",
    }
    monkeypatch.setattr(
        cloud, "get_operation", lambda **kw: {"state": "pending"}, raising=False
    )
    waited = []

    def wait(operation_id, max_wait_time):
        waited.append(operation_id)
        if operation_id == "pending":
            cloud.instances.append(
                {"id": "pending-vm", "name": adapter.name, "state": "running"}
            )
        return {}

    monkeypatch.setattr(cloud, "wait", wait)
    adapter.delete_resources(cloud, state)
    assert waited[0] == "pending"
    assert "pending-vm" in cloud.deleted
    assert [i["id"] for i in cloud.instances] == ["other"]


def test_cli_uses_alternate_lifecycle(deployed, monkeypatch):
    from llm_management import __main__ as cli
    from typer.testing import CliRunner

    config, adapter, cloud, _ = deployed
    monkeypatch.setattr(
        cli.ExoscaleConfig, "load", lambda: ExoscaleConfig(deployment=[config])
    )
    runner = CliRunner()
    result = runner.invoke(cli.app, ["create-or-resume", "clef"])
    assert result.exit_code == 0, result.exception
    result = runner.invoke(cli.app, ["logs", "clef"])
    assert result.exit_code == 0 and "logs" in result.stdout
    result = runner.invoke(cli.app, ["pause", "clef"])
    assert result.exit_code == 0, result.exception
    assert [i["id"] for i in cloud.instances] == ["other"]


def test_compute_proxies_strip_client_credentials_and_use_v1_tunnel(
    deployed, monkeypatch
):
    config, _, _, _ = deployed
    monkeypatch.setattr(server, "cache", DeploymentCache())
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: config)
    monkeypatch.setattr(server.settings, "auth_tokens", {"caller": "client-token"})
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/v1/systemone"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"served": True})

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kw: client_class(transport=httpx.MockTransport(handle), **kw),
    )
    client = TestClient(server.app)
    for path in ("/v1/systemone", "/deployments/clef/v1/systemone"):
        response = client.post(
            path,
            json={"state": "test"},
            headers={"Authorization": "Bearer client-token"},
        )
        assert response.status_code == 200 and response.json() == {"served": True}
    assert len(requests) == 2
