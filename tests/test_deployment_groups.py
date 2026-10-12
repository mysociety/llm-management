import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management import server
from llm_management.deployments import DeploymentCatalog
from llm_management.deployments.cache import DeploymentCache
from llm_management.deployments.errors import DeploymentServiceError
from llm_management.deployments.types import DeploymentState, RunningDeployment
from llm_management.inference.resources import LocalResource, ResourceRegistry


def config(groups=None):
    data = DeploymentCatalog.load().model_dump()
    data["deployment_group"] = (
        groups
        if groups is not None
        else [
            {
                "slug": "foi_pipeline",
                "deployments": ["question_slice", "foi_topic"],
            }
        ]
    )
    return DeploymentCatalog.model_validate(data)


@pytest.mark.parametrize(
    "groups",
    [
        [{"slug": "bad", "deployments": ["missing"]}],
        [{"slug": "bad", "deployments": []}],
        [{"slug": "bad", "deployments": ["foi_topic", "foi_topic"]}],
        [{"slug": "same", "deployments": ["foi_topic"]}] * 2,
    ],
)
def test_invalid_groups_rejected(groups):
    with pytest.raises(ValidationError):
        config(groups)


def test_groups_optional():
    data = config().model_dump()
    data.pop("deployment_group")
    assert DeploymentCatalog.model_validate(data).deployment_group == []


@pytest.mark.parametrize("fail", [False, True])
def test_group_runs_concurrently_and_reports_each_result(monkeypatch, fail):
    monkeypatch.setattr(server.app.state.service, "catalog", config())

    async def run():
        entered = set()
        both_started = asyncio.Event()

        async def ensure(slug, *, allow_start=False):
            assert allow_start
            entered.add(slug)
            if len(entered) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
            if fail and slug == "foi_topic":
                raise RuntimeError("private provider detail")
            return RunningDeployment(
                config=None, state=DeploymentState(slug=slug, exists=True, replicas=1)
            )

        monkeypatch.setattr(server.app.state.service, "ensure_running", ensure)
        return await server.app.state.service.ensure_group("foi_pipeline")

    result = asyncio.run(run())
    assert result.success is (not fail)
    assert [r.slug for r in result.deployments] == ["question_slice", "foi_topic"]
    assert result.deployments[0].success
    assert result.deployments[0].replicas == 1
    assert result.deployments[1].success is (not fail)
    assert "private provider detail" not in str(result.model_dump())


def test_unknown_group_starts_nothing(monkeypatch):
    monkeypatch.setattr(server.app.state.service, "catalog", config())
    ensure = AsyncMock()
    monkeypatch.setattr(server.app.state.service, "ensure_running", ensure)
    with pytest.raises(DeploymentServiceError) as exc:
        asyncio.run(server.app.state.service.ensure_group("missing"))
    assert exc.value.status_code == 404
    ensure.assert_not_called()


def test_explicit_warmup_uses_shared_lock_when_auto_start_disabled(monkeypatch):
    cache = DeploymentCache()
    state = DeploymentState(slug="foi_topic")
    calls = []
    cfg = SimpleNamespace(create_or_resume=lambda: calls.append("start"))
    monkeypatch.setattr(server.app.state.service, "cache", cache)
    monkeypatch.setattr(server.app.state.service, "auto_ensure", False)
    monkeypatch.setattr(server.app.state.service, "get_config", lambda slug: cfg)
    server.app.state.service.backends[state.slug] = SimpleNamespace(
        config=cfg, create_or_resume=cfg.create_or_resume
    )
    monkeypatch.setattr(cache, "ensure", lambda cfg: state)

    def refresh(cfg):
        state.exists = True
        state.replicas = 1
        cache.set(state)
        return state

    monkeypatch.setattr(cache, "refresh", refresh)

    async def run():
        with pytest.raises(DeploymentServiceError):
            await server.app.state.service.ensure_running(state.slug)
        return await asyncio.gather(
            *(
                server.app.state.service.ensure_running(state.slug, allow_start=True)
                for _ in range(2)
            )
        )

    results = asyncio.run(run())
    assert calls == ["start"]
    assert all(result.state.replicas == 1 for result in results)
    assert state.last_request_time > 0


def test_group_endpoint_contract_and_auth(monkeypatch):
    monkeypatch.setattr(server.app.state.service, "catalog", config())
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "test-token"})
    ensure = AsyncMock(
        return_value=RunningDeployment(
            config=None, state=DeploymentState(slug="unused", replicas=1)
        )
    )
    monkeypatch.setattr(server.app.state.service, "ensure_running", ensure)
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
    data = DeploymentCatalog.load().model_dump()
    data["deployment_group"] = [
        {"slug": "mixed", "deployments": ["custom_cpu", "foi_topic"]}
    ]
    registry = ResourceRegistry()
    load = Mock(
        side_effect=RuntimeError("private local detail") if fail_local else None
    )
    registry.register_factory(
        "custom_cpu", lambda: LocalResource("custom_cpu", load, Mock())
    )
    monkeypatch.setattr(server.app.state.service, "resources", registry)
    data["local"]["deployment"].append(
        {
            "slug": "custom_cpu",
            "loader": "presidio",
            "spacy_model": "en_core_web_sm",
            "score_threshold": 0.5,
        }
    )
    mixed = DeploymentCatalog.model_validate(data)
    monkeypatch.setattr(server.app.state.service, "catalog", mixed)
    remote = AsyncMock(
        return_value=RunningDeployment(
            config=None, state=DeploymentState(slug="foi_topic", replicas=1)
        )
    )
    monkeypatch.setattr(server.app.state.service, "ensure_running", remote)
    result = asyncio.run(server.app.state.service.ensure_group("mixed"))
    assert [member.slug for member in result.deployments] == [
        "custom_cpu",
        "foi_topic",
    ]
    assert result.success is (not fail_local)
    assert result.deployments[0].success is (not fail_local)
    assert result.deployments[0].replicas is None
    assert result.deployments[1].success
    load.assert_called_once()
    remote.assert_awaited_once_with("foi_topic", allow_start=True)
    assert "private local detail" not in str(result.model_dump())


def test_single_ensure_resolves_registry_without_remote_calls(monkeypatch):
    registry = ResourceRegistry()
    load = Mock()
    resource = registry.register(LocalResource("custom_cpu", load, Mock()))
    monkeypatch.setattr(server.app.state.service, "resources", registry)
    remote = AsyncMock(side_effect=AssertionError("No remote startup"))
    monkeypatch.setattr(server.app.state.service, "ensure_running", remote)
    result = asyncio.run(server.app.state.service.ensure_deployment("custom_cpu"))
    assert result.model_dump() == {
        "slug": "custom_cpu",
        "action": "warmed",
        "replicas": None,
    }
    assert resource.status().ready
    assert asyncio.run(server.app.state.service.ensure_local("custom_cpu")).ready
    assert load.call_count == 2
    remote.assert_not_called()


def test_shipped_foi_group_includes_registered_local_resources():
    members = DeploymentCatalog.load().get_group("foi_pipeline").deployments
    assert members == [
        "presidio",
        "question_slice_head_cpu",
        "question_extractor_tokenizer",
        "question_slice",
        "foi_topic",
    ]


def test_ambiguous_local_and_remote_names_rejected():
    data = DeploymentCatalog.load().model_dump()
    data["local"]["deployment"][0]["slug"] = "foi_topic"
    with pytest.raises(ValidationError, match="distinct names"):
        DeploymentCatalog.model_validate(data)


def test_question_slice_cpu_resources_share_the_deployment_name():
    from llm_management.deployments import get_catalog

    catalog = get_catalog()
    foi = catalog.require_foi()
    assert foi.classifier == f"{foi.extraction_deployment}_cpu"
    assert foi.head == f"{foi.extraction_deployment}_head_cpu"
    assert {foi.classifier, foi.head} <= server.app.state.service.resources.names()
    cpu_members = catalog.get_group("foi_pipeline_cpu").deployments
    assert foi.classifier in cpu_members
