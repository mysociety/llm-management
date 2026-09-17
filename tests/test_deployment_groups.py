import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management import server
from llm_management.cache import DeploymentCache, DeploymentState
from llm_management.models import ExoscaleConfig


def config(groups=None):
    data = ExoscaleConfig.load().model_dump()
    if groups is not None:
        data["deployment_group"] = groups
    return ExoscaleConfig.model_validate(data)


@pytest.mark.parametrize(
    "groups",
    [
        [{"slug": "bad", "deployments": ["missing"]}],
        [{"slug": "bad", "deployments": []}],
        [{"slug": "bad", "deployments": ["foi_topic_v2", "foi_topic_v2"]}],
        [{"slug": "same", "deployments": ["foi_topic_v2"]}] * 2,
    ],
)
def test_invalid_groups_rejected(groups):
    with pytest.raises(ValidationError):
        config(groups)


def test_groups_optional():
    data = config().model_dump()
    data.pop("deployment_group")
    assert ExoscaleConfig.model_validate(data).deployment_group == []


@pytest.mark.parametrize("fail", [False, True])
def test_group_runs_concurrently_and_reports_each_result(monkeypatch, fail):
    monkeypatch.setattr(server, "load_config", config)

    async def run():
        entered = set()
        both_started = asyncio.Event()

        async def ensure(slug, *, allow_start=False):
            assert allow_start
            entered.add(slug)
            if len(entered) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
            if fail and slug == "foi_topic_v2":
                raise RuntimeError("private provider detail")
            return None, DeploymentState(slug=slug, exists=True, replicas=1)

        monkeypatch.setattr(server, "ensure_running", ensure)
        return await server.ensure_deployment_group("foi_pipeline")

    result = asyncio.run(run())
    assert result.success is (not fail)
    assert [r.slug for r in result.deployments] == ["question_slice_v2", "foi_topic_v2"]
    assert result.deployments[0].success
    assert result.deployments[0].replicas == 1
    assert result.deployments[1].success is (not fail)
    assert "private provider detail" not in result.model_dump_json()


def test_unknown_group_starts_nothing(monkeypatch):
    monkeypatch.setattr(server, "load_config", config)
    ensure = AsyncMock()
    monkeypatch.setattr(server, "ensure_running", ensure)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(server.ensure_deployment_group("missing"))
    assert exc.value.status_code == 404
    ensure.assert_not_called()


def test_explicit_warmup_uses_shared_lock_when_auto_start_disabled(monkeypatch):
    cache = DeploymentCache()
    state = DeploymentState(slug="foi_topic_v2")
    calls = []
    cfg = SimpleNamespace(create_or_resume=lambda: calls.append("start"))
    monkeypatch.setattr(server, "cache", cache)
    monkeypatch.setattr(server, "AUTO_ENSURE_ON_REQUEST", False)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: cfg)
    monkeypatch.setattr(cache, "ensure", lambda cfg: state)

    def refresh(cfg):
        state.exists = True
        state.replicas = 1
        cache.set(state)
        return state

    monkeypatch.setattr(cache, "refresh", refresh)

    async def run():
        with pytest.raises(HTTPException):
            await server.ensure_running(state.slug)
        return await asyncio.gather(
            *(server.ensure_running(state.slug, allow_start=True) for _ in range(2))
        )

    results = asyncio.run(run())
    assert calls == ["start"]
    assert all(s.replicas == 1 for _, s in results)
    assert state.last_request_time > 0


def test_group_endpoint_contract_and_auth(monkeypatch):
    monkeypatch.setattr(server, "load_config", config)
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "test-token"})
    ensure = AsyncMock(return_value=(None, DeploymentState(slug="unused", replicas=1)))
    monkeypatch.setattr(server, "ensure_running", ensure)
    # No lifespan: these HTTP contract checks must not touch real deployments.
    client = TestClient(server.app)
    url = "/deployment-groups/foi_pipeline/ensure"
    assert client.post(url).status_code == 401
    ensure.assert_not_called()
    response = client.post(url, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert len(response.json()["deployments"]) == 2
