"""Explicit live tests for the prepared Compute server; creates billed resources.

Run only with pytest -m external tests/test_compute_external.py. Uses the Clef
recipe, exercises real HTTP/System One calls, and deletes the owned VM in finally.
Use a distinct SERVER_ROLE and persistent COMPUTE_STATE_DIR for isolated runs.
"""

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import time

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.cache import DeploymentCache
from llm_management.compute_deployments import close_tunnel
from llm_management.models import ComputeDeploymentConfig

pytestmark = pytest.mark.external


def test_compute_lifecycle_and_agents_live():
    if not server.settings.server_role.startswith("systemone_live_"):
        pytest.skip(
            "Set SERVER_ROLE=systemone_live_<unique-run> to isolate paid test resources"
        )
    cfg = server.get_deployment_config("clef")
    assert isinstance(cfg, ComputeDeploymentConfig)
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "template_name": cfg.recipe.name,
        "deployment_name": cfg.adapter.name,
        "model": cfg.model,
        "revision": cfg.recipe.revision,
        "checks": [],
    }
    destination = Path(
        os.environ.get("LLM_COMPUTE_SMOKE_REPORT", "compute-smoke-results.json")
    )
    old_cache = server.cache
    old_auth = server.settings.auth_tokens
    old_preload = server.settings.cpu_inference_preload
    server.settings.auth_tokens = {"live-smoke": secrets.token_urlsafe(32)}
    server.settings.cpu_inference_preload = False
    server.cache = DeploymentCache()
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
            original = cfg.adapter.instance(cfg._client())
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
                    "/agents/immigration_detection/clef",
                    headers=headers,
                    json={"request": text},
                )
                record(f"native_agent_{expected}", response)
                assert response.json()["classification"] == expected

            # Drop only the local connection, then reconnect using the durable key.
            close_tunnel(cfg.adapter.runtime_key)
            server.cache.remove("clef")
            start = time.monotonic()
            record(
                "reconnect", client.post("/deployments/clef/ensure", headers=headers)
            )
            report["reconnect_seconds"] = round(time.monotonic() - start, 3)
            assert cfg.adapter.instance(cfg._client())["id"] == original["id"]
            report["reused_instance"] = True
            logs = cfg.adapter.logs(10)
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
            assert cfg.adapter.instance(cfg._client()) is None
    finally:
        try:
            cfg.delete_deployment()
            client = cfg._client()
            assert cfg.adapter.instance(client) is None
            assert not any(
                k.get("name") == cfg.adapter.name
                for k in client.list_ssh_keys().get("ssh-keys", [])
            )
            assert not any(
                g.get("name") == cfg.adapter.name
                for g in client.list_security_groups().get("security-groups", [])
            )
            report["cleanup_verified"] = True
        finally:
            server.cache = old_cache
            server.settings.auth_tokens = old_auth
            server.settings.cpu_inference_preload = old_preload
            destination.write_text(json.dumps(report, indent=2) + "\n")
