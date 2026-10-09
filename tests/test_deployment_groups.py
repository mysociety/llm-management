import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management import server
from llm_management.cache import DeploymentCache, DeploymentState, RunningDeployment
from llm_management.models import ExoscaleConfig
from llm_management.local_resources import LocalResource, ResourceRegistry


def config(groups=None):
    data = ExoscaleConfig.load().model_dump()
    data["deployment_group"] = (
        groups
        if groups is not None
        else [
            {
                "slug": "foi_pipeline",
                "deployments": ["question_slice_v2", "foi_topic_v2"],
            }
        ]
    )
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
            return RunningDeployment(
                config=None, state=DeploymentState(slug=slug, exists=True, replicas=1)
            )

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
    assert all(result.state.replicas == 1 for result in results)
    assert state.last_request_time > 0


def test_group_endpoint_contract_and_auth(monkeypatch):
    monkeypatch.setattr(server, "load_config", config)
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "test-token"})
    ensure = AsyncMock(
        return_value=RunningDeployment(
            config=None, state=DeploymentState(slug="unused", replicas=1)
        )
    )
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


@pytest.mark.parametrize("fail_local", [False, True])
def test_mixed_group_resolves_arbitrary_registered_local_resource(
    monkeypatch, fail_local
):
    data = ExoscaleConfig.load().model_dump()
    data["deployment_group"] = [
        {"slug": "mixed", "deployments": ["custom_cpu", "foi_topic_v2"]}
    ]
    registry = ResourceRegistry()
    load = Mock(
        side_effect=RuntimeError("private local detail") if fail_local else None
    )
    registry.register_factory(
        "custom_cpu", lambda: LocalResource("custom_cpu", load, Mock())
    )
    monkeypatch.setattr(server, "local_resources", registry)
    monkeypatch.setattr("llm_management.models.local_resources", registry)
    mixed = ExoscaleConfig.model_validate(data)
    monkeypatch.setattr(server, "load_config", lambda: mixed)
    remote = AsyncMock(
        return_value=RunningDeployment(
            config=None, state=DeploymentState(slug="foi_topic_v2", replicas=1)
        )
    )
    monkeypatch.setattr(server, "ensure_running", remote)
    result = asyncio.run(server.ensure_deployment_group("mixed"))
    assert [member.slug for member in result.deployments] == [
        "custom_cpu",
        "foi_topic_v2",
    ]
    assert result.success is (not fail_local)
    assert result.deployments[0].success is (not fail_local)
    assert result.deployments[0].replicas is None
    assert result.deployments[1].success
    load.assert_called_once()
    remote.assert_awaited_once_with("foi_topic_v2", allow_start=True)
    assert "private local detail" not in result.model_dump_json()


def test_single_ensure_resolves_registry_without_remote_calls(monkeypatch):
    registry = ResourceRegistry()
    load = Mock()
    resource = registry.register(LocalResource("custom_cpu", load, Mock()))
    monkeypatch.setattr(server, "local_resources", registry)
    remote = AsyncMock(side_effect=AssertionError("No remote startup"))
    monkeypatch.setattr(server, "ensure_running", remote)
    result = asyncio.run(server.ensure_deployment("custom_cpu"))
    assert result.model_dump() == {
        "slug": "custom_cpu",
        "action": "warmed",
        "replicas": None,
    }
    assert resource.status()["ready"]
    assert asyncio.run(server.ensure_local_model("custom_cpu"))["ready"]
    assert load.call_count == 2
    remote.assert_not_called()


def test_shipped_foi_group_includes_registered_local_resources():
    members = ExoscaleConfig.load().get_group("foi_pipeline").deployments
    assert members == [
        "presidio",
        "question_slice_v2_head_cpu",
        "question_extractor_tokenizer",
        "question_slice_v2",
        "foi_topic_v2",
    ]


def test_ambiguous_local_and_remote_names_rejected(monkeypatch):
    registry = ResourceRegistry()
    registry.register(LocalResource("foi_topic_v2", Mock(), Mock()))
    monkeypatch.setattr("llm_management.models.local_resources", registry)
    with pytest.raises(ValidationError, match="distinct names"):
        config()


def test_question_slice_cpu_resources_share_the_deployment_name():
    from llm_management.foi.model_spec import (
        QUESTION_SLICE_DEPLOYMENT,
        QUESTION_SLICE_CPU_RESOURCE,
        QUESTION_SLICE_HEAD_CPU_RESOURCE,
    )
    from llm_management.local_resources import local_resources

    assert QUESTION_SLICE_CPU_RESOURCE == f"{QUESTION_SLICE_DEPLOYMENT}_cpu"
    assert QUESTION_SLICE_HEAD_CPU_RESOURCE == f"{QUESTION_SLICE_DEPLOYMENT}_head_cpu"
    names = local_resources.names()
    assert {QUESTION_SLICE_CPU_RESOURCE, QUESTION_SLICE_HEAD_CPU_RESOURCE} <= names
    assert "question_classifier" not in names
    assert "question_classification_head" not in names
    cpu_members = ExoscaleConfig.load().get_group("foi_pipeline_cpu").deployments
    assert QUESTION_SLICE_CPU_RESOURCE in cpu_members
