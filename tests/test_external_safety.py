"""
Exercise live-test safety gates without contacting cloud or model providers.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from llm_management.deployments.compute import close_tunnel
from llm_management.deployments.status import DeploymentStatusCache
from llm_management.settings import settings
from tests import test_compute_deployments
from tests.external_safety import OwnedComputeDeployment, isolated_external_run

deployed = test_compute_deployments.deployed


@pytest.mark.parametrize(
    "arguments,role,outcome",
    [
        ([], "test", "1 skipped"),
        (["-m", "external"], "test", "1 skipped"),
        (["--run-external"], "production", "1 error"),
        (["--run-external"], "test", "1 passed"),
    ],
)
def test_live_gate_runs_before_session_fixtures(tmp_path, arguments, role, outcome):
    (tmp_path / "conftest.py").write_text(
        "from tests.conftest import (pytest_addoption, "
        "pytest_collection_modifyitems, external_test_run)\n"
    )
    (tmp_path / "test_live.py").write_text(
        """
import os
import re
from pathlib import Path
import pytest
from llm_management.settings import settings

pytestmark = pytest.mark.external

@pytest.fixture(scope="session")
def application():
    assert re.fullmatch(r"test_[a-f0-9]{32}", settings.server_role)
    assert os.environ["SERVER_ROLE"] == settings.server_role
    assert Path(os.environ["COMPUTE_STATE_DIR"]) == settings.compute_state_dir
    assert settings.compute_state_dir.name == settings.server_role
    Path("fixture-ran").touch()

def test_live(application):
    pass
"""
    )
    environment = dict(os.environ)
    environment.update(
        SERVER_ROLE=role,
        COMPUTE_STATE_DIR=str(tmp_path / "state"),
        PYTHONPATH=os.pathsep.join(
            [
                str(Path(__file__).resolve().parents[1]),
                environment.get("PYTHONPATH", ""),
            ]
        ),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert outcome in result.stdout, result.stdout + result.stderr
    assert result.returncode == (1 if outcome == "1 error" else 0)
    assert (tmp_path / "fixture-ran").exists() == (outcome == "1 passed")


def test_live_role_outlives_session_client_cleanup(pytester, monkeypatch):
    """
    Keep isolation active when a later module shadows the session client fixture.
    """
    import re
    from types import SimpleNamespace

    from tests import test_proxy

    closed_roles = []

    class FakeClient:
        def __init__(self, app):
            pass

        def __enter__(self):
            self.role = settings.server_role
            assert re.fullmatch(r"test_[a-f0-9]{32}", self.role)
            return self

        def post(self, path):
            return SimpleNamespace(raise_for_status=lambda: None)

        def __exit__(self, *args):
            assert settings.server_role == self.role, "Isolation ended before cleanup"
            closed_roles.append(self.role)

    # Exercise the real session fixture with no runtime or cloud operations.
    monkeypatch.setattr(test_proxy, "TestClient", FakeClient)
    monkeypatch.setattr(test_proxy, "assemble_runtime", lambda: None)
    monkeypatch.setattr(test_proxy, "create_app", lambda **kwargs: None)
    monkeypatch.setattr(settings, "server_role", "test")
    monkeypatch.setattr(settings, "compute_state_dir", pytester.path / "state")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    pytester.makeconftest(
        "from tests.conftest import "
        "pytest_addoption, pytest_collection_modifyitems, external_test_run"
    )
    pytester.makeini("[pytest]\nmarkers = external: isolated live-test fixture")
    pytester.makepyfile(
        test_a_proxy="""
            import pytest
            from tests.test_proxy import client

            @pytest.mark.external
            def test_proxy_client(client):
                pass
        """,
        test_b_other="""
            import pytest

            @pytest.fixture(scope="module")
            def client():
                yield None

            @pytest.mark.external
            def test_other_client(client):
                pass
        """,
    )
    result = pytester.runpytest_inprocess("-q", "--run-external")
    result.assert_outcomes(passed=2)
    assert result.ret == 0
    assert len(closed_roles) == 1


@pytest.mark.parametrize("role", ["production", "", "test/production", "testing"])
def test_operational_roles_rejected_without_creating_state(tmp_path, monkeypatch, role):
    monkeypatch.setattr(settings, "server_role", role)
    monkeypatch.setattr(settings, "compute_state_dir", tmp_path / "state")
    with pytest.raises(pytest.UsageError):
        with isolated_external_run():
            pytest.fail("Operational role was accepted")
    assert not settings.compute_state_dir.exists()


def test_isolation_is_unique_and_restored_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "server_role", "test")
    monkeypatch.setattr(settings, "compute_state_dir", tmp_path)
    monkeypatch.setenv("SERVER_ROLE", "test")
    monkeypatch.setenv("COMPUTE_STATE_DIR", str(tmp_path))
    roles = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="test failure"):
            with isolated_external_run() as role:
                roles.append(role)
                assert settings.compute_state_dir.is_dir()
                raise RuntimeError("test failure")
        assert settings.server_role == os.environ["SERVER_ROLE"] == "test"
        assert (
            settings.compute_state_dir
            == Path(os.environ["COMPUTE_STATE_DIR"])
            == tmp_path
        )
    assert roles[0] != roles[1]
    assert all((tmp_path / "pytest" / role).is_dir() for role in roles)


@pytest.fixture
def owned(deployed, http_app, monkeypatch):
    config, _, cloud, _ = deployed
    role = "test_" + "1" * 32
    monkeypatch.setattr(settings, "server_role", role)
    adapter = OwnedComputeDeployment(config, role)
    monkeypatch.setattr(adapter, "health", lambda url: True)
    http_app.state.runtime.deployments.backends[config.slug] = adapter
    try:
        yield adapter, cloud
    finally:
        close_tunnel(adapter.runtime_key)


@pytest.mark.parametrize("resource", ["vm", "key", "group", "manifest"])
def test_existing_compute_resources_are_never_adopted_or_deleted(owned, resource):
    adapter, cloud = owned
    if resource == "vm":
        cloud.instances.append({"id": "foreign", "name": adapter.name})
    elif resource == "key":
        cloud.keys.append({"name": adapter.name})
    elif resource == "group":
        cloud.groups.append({"id": "foreign", "name": adapter.name})
    else:
        adapter.state().save(adapter.path)
    with pytest.raises(RuntimeError, match="pre-existing"):
        adapter.ensure()
    adapter.delete_deployment()
    assert not cloud.created
    assert not cloud.deleted


def test_owned_compute_cleanup_is_idempotent(owned):
    adapter, cloud = owned
    adapter.ensure()
    adapter.scale_to_zero()
    deleted = cloud.deleted.copy()
    adapter.delete_deployment()
    assert cloud.deleted == deleted
    assert "vm" in deleted
    assert cloud.instances == [{"name": "unrelated", "id": "other", "state": "running"}]


def test_failed_startup_cleans_up_only_the_recorded_vm(owned, monkeypatch):
    adapter, cloud = owned

    def fail_health(url):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(adapter, "health", fail_health)
    with pytest.raises(RuntimeError, match="startup failed"):
        adapter.ensure()
    assert "vm" in cloud.deleted
    assert not cloud.keys and not cloud.groups


def test_incomplete_creation_preserves_vm_for_manual_recovery(owned, monkeypatch):
    adapter, cloud = owned
    create = cloud.create_instance

    def interrupted_create(**kwargs):
        create(**kwargs)
        raise RuntimeError("creation response lost")

    monkeypatch.setattr(cloud, "create_instance", interrupted_create)
    with pytest.raises(RuntimeError, match="creation response lost"):
        adapter.ensure()
    assert not cloud.deleted
    assert cloud.keys and cloud.groups
    assert adapter.path.exists()


def test_application_shutdown_cannot_bypass_ownership_check(
    owned, http_app, monkeypatch
):
    adapter, cloud = owned
    adapter.ensure()
    service = http_app.state.runtime.deployments
    status = DeploymentStatusCache()
    status.refresh(adapter)
    service.activity.touch("clef")
    monkeypatch.setattr(service, "status", status)
    next(instance for instance in cloud.instances if instance["id"] == "vm")["id"] = (
        "replacement"
    )
    asyncio.run(service.close())
    assert not cloud.deleted
    assert cloud.keys and cloud.groups


@pytest.mark.parametrize("recorded", [None, "different-vm"])
def test_cleanup_retains_unrecorded_or_replaced_vm_and_access(owned, recorded):
    adapter, cloud = owned
    adapter.ensure()
    state = adapter.state()
    state.instance_id = recorded
    state.save(adapter.path)
    with pytest.raises(RuntimeError, match="not recorded"):
        adapter.scale_to_zero()
    assert not cloud.deleted
    assert cloud.keys and cloud.groups
    assert adapter.path.exists()
