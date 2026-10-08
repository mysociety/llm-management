"""Template configuration, name lookup and paid-resource lifecycle without cloud calls."""

import base64
import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_management.models import LLMManagementError
from llm_management.templates import TemplateConfig, TemplateRecipe
from llm_management.templates import cli, compute, lifecycle
from llm_management.templates.probes import validate_health


@pytest.fixture
def recipe():
    return TemplateConfig.load().get("clef_flash")


def test_config_rejects_duplicate_releases_and_bad_smoke_cases(recipe):
    with pytest.raises(ValidationError, match="unique"):
        TemplateConfig(template=[recipe, recipe])
    data = recipe.model_dump()
    data["backend"] = "other"
    with pytest.raises(ValidationError, match="Only the clef"):
        TemplateRecipe.model_validate(data)
    data = recipe.model_dump()
    data["smoke"][0]["expected"] = "missing"
    with pytest.raises(ValidationError, match="criteria"):
        TemplateRecipe.model_validate(data)


def test_resolve_private_template_by_exact_name(recipe):
    client = Mock()
    client.list_templates.return_value = {
        "templates": [
            {"name": "unrelated", "id": "other"},
            {"name": recipe.name, "id": "ours"},
        ]
    }
    assert recipe.resolve(client)["id"] == "ours"
    client.list_templates.assert_called_with(visibility="private")
    client.list_templates.return_value["templates"].append(
        {"name": recipe.name, "id": "duplicate"}
    )
    with pytest.raises(LLMManagementError, match="Multiple"):
        recipe.resolve(client)
    client.list_templates.return_value = {"templates": []}
    with pytest.raises(LLMManagementError, match="not found"):
        recipe.resolve(client)


def test_cloud_init_uses_configured_runtime_settings(recipe):
    recipe = recipe.model_copy(
        update={"model": "Cloudflare/another-model", "port": 8123, "max_length": 2048}
    )
    config = json.loads(
        base64.b64decode(lifecycle.cloud_init(recipe)).decode().split("\n", 1)[1]
    )
    command = config["runcmd"][0]
    assert "SYSTEMONE_MODEL=Cloudflare/another-model" in command
    assert "SYSTEMONE_PORT=8123" in command
    assert "SYSTEMONE_MAX_LENGTH=2048" in command
    assert "EXOSCALE_API" not in command
    assert "HF_HUB_OFFLINE=1" in config["write_files"][0]["content"]


def test_health_requires_the_expected_cached_release(recipe):
    health = {
        "backend": recipe.backend,
        "model": recipe.model,
        "revision": recipe.revision,
        "max_length": recipe.max_length,
    }
    validate_health(health, recipe)
    health["revision"] = "another-release"
    with pytest.raises(RuntimeError, match="revision"):
        validate_health(health, recipe)


def test_existing_release_refuses_build_before_provisioning(recipe, monkeypatch):
    client = Mock()
    client.list_templates.return_value = {
        "templates": [{"name": recipe.name, "id": "ours"}]
    }
    monkeypatch.setattr(lifecycle, "get_client", lambda zone: client)
    with pytest.raises(LLMManagementError, match="already exists"):
        lifecycle.build(recipe)
    client.create_instance.assert_not_called()


def test_execute_always_cleans_up_after_failed_probe(recipe, tmp_path, monkeypatch):
    client = Mock()
    monkeypatch.setattr(compute, "run", Mock(side_effect=RuntimeError("probe failed")))
    recover = Mock()
    monkeypatch.setattr(lifecycle, "cleanup", recover)
    path = tmp_path / "state.json"
    with pytest.raises(RuntimeError, match="probe failed"):
        lifecycle.execute(client, recipe, path)
    state = json.loads(path.read_text())
    assert state["error_type"] == "RuntimeError"
    assert state["recipe"]["model"] == recipe.model
    recover.assert_called_once_with(client, state, path)


def test_cleanup_only_deletes_owned_snapshots(tmp_path, monkeypatch):
    client = Mock()
    state = {"name": "llm-template-test", "instance_id": "ours"}
    client.list_instances.return_value = {"instances": []}
    client.delete_snapshot.return_value = {"id": "delete-op"}
    client.list_snapshots.return_value = {
        "snapshots": [
            {"id": "ours-snapshot", "instance": {"id": "ours"}},
            {"id": "other-snapshot", "instance": {"id": "other"}},
            {"id": "orphan", "instance": {}},
        ]
    }
    monkeypatch.setattr(compute, "cleanup", Mock())
    lifecycle.cleanup(client, state, tmp_path / "state.json")
    client.delete_snapshot.assert_called_once_with(id="ours-snapshot")
    with pytest.raises(ValueError, match="manifest"):
        lifecycle.cleanup(client, {"name": "production"}, tmp_path / "state.json")


def test_cli_resolves_recipe_and_dispatches_build(recipe, monkeypatch):
    client = Mock()
    client.list_templates.return_value = {
        "templates": [{"name": recipe.name, "id": "ours"}]
    }
    monkeypatch.setattr(cli, "get_client", lambda zone: client)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["resolve", "clef_flash"])
    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "ours"
    build = Mock()
    monkeypatch.setattr(cli, "build", build)
    result = runner.invoke(
        cli.app, ["create", "clef_flash", "--runs", "1", "--ssh-cidr", "192.0.2.1/32"]
    )
    assert result.exit_code == 0, result.exception
    configured, runs = build.call_args.args
    assert configured.name == recipe.name
    assert configured.ssh_cidr == "192.0.2.1/32"
    assert runs == 1
    assert (
        runner.invoke(cli.app, ["create", "clef_flash", "--runs", "0"]).exit_code != 0
    )


def test_cleanup_recovers_resources_and_preserves_access_on_failure(tmp_path):
    from tests.test_template_compute import FakeCompute

    state = {"name": "llm-template-test"}
    path = tmp_path / "state.json"
    client = FakeCompute(state["name"], fail_delete=True)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        compute.cleanup(client, state, path)
    assert client.groups and client.keys
    client.fail_delete = False
    compute.cleanup(client, state, path)
    assert client.instances == [{"name": "unrelated", "id": "other"}]
    assert state["cleanup_complete"]
    compute.cleanup(client, state, path)
    assert len(client.deleted) == 3


def test_snapshot_registration_uses_recipe_name_and_metadata(
    recipe, tmp_path, monkeypatch
):
    client = Mock()
    client.list_templates.return_value = {"templates": []}
    client.create_snapshot.return_value = {"id": "snapshot-op"}
    client.stop_instance.return_value = {"id": "stop-op"}
    client.export_snapshot.return_value = {"id": "export-op"}
    client.register_template.return_value = {"id": "register-op"}
    client.wait.side_effect = lambda operation_id, **kw: {
        "reference": {"id": operation_id + "-result"}
    }
    client.get_snapshot.return_value = {
        "export": {
            "presigned-url": "https://example.com/snapshot",
            "md5sum": "checksum",
        },
        "size": 100,
    }
    result = Mock(stdout=b"", stderr=b"")
    monkeypatch.setattr(lifecycle.subprocess, "run", Mock(return_value=result))
    state = {"zone": recipe.zone}
    lifecycle.prepare_template(
        client,
        recipe,
        state,
        tmp_path / "state.json",
        ["ssh"],
        {"id": "builder"},
        {"default-user": "ubuntu", "boot-mode": "uefi"},
    )
    assert client.register_template.call_args.kwargs["name"] == recipe.name
    assert client.register_template.call_args.kwargs["boot_mode"] == "uefi"
    assert state["template_id"] == "register-op-result"


def test_benchmark_resolves_name_and_retains_template(recipe, tmp_path, monkeypatch):
    client = Mock()
    client.list_templates.return_value = {
        "templates": [{"name": recipe.name, "id": "template"}]
    }
    monkeypatch.setattr(lifecycle, "get_client", lambda zone: client)
    monkeypatch.setattr(lifecycle.tempfile, "mkdtemp", lambda **kw: str(tmp_path))
    execute = Mock(return_value={"inference_passed": True})
    monkeypatch.setattr(lifecycle, "execute", execute)
    path = lifecycle.build(recipe, runs=2, test_only=True)
    assert execute.call_count == 2
    assert execute.call_args.kwargs["template_id"] == "template"
    assert json.loads(path.read_text())["template_id"] == "template"
    client.delete_template.assert_not_called()


def test_configured_smoke_cases_use_http_and_native_provider(
    recipe, tmp_path, monkeypatch
):
    import httpx
    import httpx2
    from pydantic_ai.providers import system_one
    from llm_management.templates import probes

    data = recipe.model_dump()
    data["model"] = "Cloudflare/future-clef"
    data["smoke"] = [
        {
            "state": "Is this a test?",
            "instructions": "Choose YES for a test.",
            "criteria": {"YES": "A test", "NO": "Something else"},
            "expected": "YES",
        }
    ]
    recipe = TemplateRecipe.model_validate(data)
    requests = []

    def response_for(request):
        body = json.loads(request.content)
        requests.append(body)
        return {
            "model": recipe.model,
            "answers": {
                "classification": {
                    "type": "choice",
                    "choice": "YES",
                    "confidence": 0.99,
                    "probabilities": {"YES": 0.99, "NO": 0.01},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }

    client = httpx.Client
    monkeypatch.setattr(
        probes.httpx,
        "Client",
        lambda **kw: client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json=response_for(req))
            ),
            **kw,
        ),
    )
    provider = system_one.SystemOneProvider
    monkeypatch.setattr(
        system_one,
        "SystemOneProvider",
        lambda **kw: provider(
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda req: httpx2.Response(200, json=response_for(req))
                )
            ),
            **kw,
        ),
    )
    probes.probe("http://localhost:8000", tmp_path, recipe)
    assert len(requests) == 2
    assert all(body["model"] == recipe.model for body in requests)
    assert json.loads((tmp_path / "pydantic-ai.json").read_text()) == {
        "classification": "YES"
    }
