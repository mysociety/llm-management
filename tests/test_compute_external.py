"""Explicit live tests for the prepared Compute server; creates billed resources.

Run only with pytest -m external tests/test_compute_external.py. Uses the Clef
recipe, exercises real HTTP/System One calls, and deletes the owned VM in finally.
Use SERVER_ROLE=test and persistent COMPUTE_STATE_DIR for sequential runs.
Concurrent isolated runs can use SERVER_ROLE=test_<run> with separate state directories.
"""

import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.deployments.cache import DeploymentCache
from llm_management.deployments.client import get_client
from llm_management.deployments.compute import close_tunnel
from llm_management.deployments.config import ComputeDeploymentConfig

pytestmark = pytest.mark.external


def test_compute_lifecycle_and_agents_live():
    if (
        server.settings.server_role != "test"
        and not server.settings.server_role.startswith("test_")
    ):
        pytest.skip("Set SERVER_ROLE=test (or test_<run> for concurrent isolated runs)")
    cfg = server.app.state.service.get_config("clef")
    backend = server.app.state.service.get_backend("clef")
    assert isinstance(cfg, ComputeDeploymentConfig)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "template_name": cfg.recipe.name,
        "deployment_name": backend.name,
        "model": cfg.model,
        "revision": cfg.recipe.revision,
        "checks": [],
    }
    destination = Path(
        os.environ.get("LLM_COMPUTE_SMOKE_REPORT", "compute-smoke-results.json")
    )
    old_cache = server.app.state.service.cache
    old_auth = server.settings.auth_tokens
    server.settings.auth_tokens = {"live-smoke": secrets.token_urlsafe(32)}
    server.app.state.service.cache = DeploymentCache()
    headers = {"Authorization": "Bearer " + server.settings.auth_tokens["live-smoke"]}

    def record(name, response):
        report["checks"].append(
            {"name": name, "status": response.status_code, "response": response.json()}
        )
        destination.write_text(json.dumps(report, indent=2) + "\n")
        response.raise_for_status()

    try:
        with TestClient(server.app) as client:
            assert client.post("/deployments/clef/ensure").status_code == 401
            start = time.monotonic()
            record("ensure", client.post("/deployments/clef/ensure", headers=headers))
            report["ensure_seconds"] = round(time.monotonic() - start, 3)
            original = backend.instance(get_client(cfg.zone))
            report["instance_id"] = original["id"]
            record(
                "ready_status", client.get("/deployments/clef/status", headers=headers)
            )
            assert report["checks"][-1]["response"]["replicas"] == 1

            for index, case in enumerate(cfg.recipe.smoke):
                body = {
                    "model": cfg.model,
                    "state": case.state,
                    "questions": {
                        "classification": {
                            "type": "choice",
                            "instructions": case.instructions,
                            "criteria": case.criteria,
                        }
                    },
                }
                path = (
                    "/v1/systemone" if index == 0 else "/deployments/clef/v1/systemone"
                )
                response = client.post(path, headers=headers, json=body)
                record(f"proxy_{index + 1}", response)
                assert (
                    response.json()["answers"]["classification"]["choice"]
                    == case.expected
                )

            for text, expected in [
                ("Please update me on my visa application.", "IMM"),
                (
                    "Please provide the council's spending on road repairs last year.",
                    "FOI",
                ),
            ]:
                response = client.post(
                    "/endpoint/foi/immigration_detection/clef",
                    headers=headers,
                    json={"body": text},
                )
                record(f"native_agent_{expected}", response)
                assert response.json()["classification"] == expected

            # Drop only the local connection, then reconnect using the durable key.
            close_tunnel(backend.runtime_key)
            server.app.state.service.cache.remove("clef")
            start = time.monotonic()
            record(
                "reconnect", client.post("/deployments/clef/ensure", headers=headers)
            )
            report["reconnect_seconds"] = round(time.monotonic() - start, 3)
            assert backend.instance(get_client(cfg.zone))["id"] == original["id"]
            report["reused_instance"] = True
            logs = backend.logs(10)
            assert logs.strip() and "-- No entries --" not in logs
            report["logs_checked"] = True

            record(
                "pause", client.post("/deployments/clef/scale-to-zero", headers=headers)
            )
            record(
                "deleted_status",
                client.get("/deployments/clef/status", headers=headers),
            )
            assert not report["checks"][-1]["response"]["exists"]
            assert backend.instance(get_client(cfg.zone)) is None
    finally:
        try:
            backend.delete_deployment()
            client = get_client(cfg.zone)
            assert backend.instance(client) is None
            assert not any(
                k.get("name") == backend.name
                for k in client.list_ssh_keys().get("ssh-keys", [])
            )
            assert not any(
                g.get("name") == backend.name
                for g in client.list_security_groups().get("security-groups", [])
            )
            report["cleanup_verified"] = True
        finally:
            server.app.state.service.cache = old_cache
            server.settings.auth_tokens = old_auth
            destination.write_text(json.dumps(report, indent=2) + "\n")
