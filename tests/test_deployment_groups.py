import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management import server
from llm_management.deployments import DeploymentCatalog
from llm_management.deployments.errors import DeploymentServiceError
from llm_management.deployments.status import DeploymentStatusCache
from llm_management.deployments.types import DeploymentSnapshot, RunningDeployment
from llm_management.inference.resources import LocalResource, ResourceRegistry


def config(groups=None):
    data = DeploymentCatalog.load().model_dump()
    data["deployment_group"] = (
        groups
        if groups is not None
        else [
            {
                "slug": "group_foi_pipeline",
                "deployments": ["question_slice", "foi_topic"],
            }
        ]
    )
    return DeploymentCatalog.model_validate(data)


@pytest.mark.parametrize(
    "groups",
    [
        [{"slug": "group_bad", "deployments": ["group_missing"]}],
        [{"slug": "group_bad", "deployments": []}],
        [{"slug": "group_bad", "deployments": ["foi_topic", "foi_topic"]}],
        [{"slug": "group_same", "deployments": ["foi_topic"]}] * 2,
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
    monkeypatch.setattr(server.app.state.application.deployments, "catalog", config())

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
                config=None,
                state=DeploymentSnapshot(slug=slug, exists=True, replicas=1),
            )

        monkeypatch.setattr(
            server.app.state.application.deployments, "ensure_running", ensure
        )
        return await server.app.state.application.deployments.ensure_group(
            "group_foi_pipeline"
        )

    result = asyncio.run(run())
    assert result.success is (not fail)
    assert [r.slug for r in result.deployments] == ["question_slice", "foi_topic"]
    assert result.deployments[0].success
    assert result.deployments[0].replicas == 1
    assert result.deployments[1].success is (not fail)
    assert "private provider detail" not in str(result.model_dump())


def test_unknown_group_starts_nothing(monkeypatch):
    monkeypatch.setattr(server.app.state.application.deployments, "catalog", config())
    ensure = AsyncMock()
    monkeypatch.setattr(
        server.app.state.application.deployments, "ensure_running", ensure
    )
    with pytest.raises(DeploymentServiceError) as exc:
        asyncio.run(
            server.app.state.application.deployments.ensure_group("group_missing")
        )
    assert exc.value.status_code == 404
    ensure.assert_not_called()


def test_explicit_warmup_uses_shared_lock_when_auto_start_disabled(monkeypatch):
    cache = DeploymentStatusCache()
    state = DeploymentSnapshot(slug="foi_topic")
    calls = []
    cfg = SimpleNamespace(create_or_resume=lambda: calls.append("start"))
    monkeypatch.setattr(server.app.state.application.deployments, "status", cache)
    monkeypatch.setattr(server.app.state.application.deployments, "auto_ensure", False)
    monkeypatch.setattr(
        server.app.state.application.deployments, "get_config", lambda slug: cfg
    )
    server.app.state.application.deployments.backends[state.slug] = SimpleNamespace(
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
            await server.app.state.application.deployments.ensure_running(state.slug)
        return await asyncio.gather(
            *(
                server.app.state.application.deployments.ensure_running(
                    state.slug, allow_start=True
                )
                for _ in range(2)
            )
        )

    results = asyncio.run(run())
    assert calls == ["start"]
    assert all(result.state.replicas == 1 for result in results)
    assert (
        server.app.state.application.deployments.activity.get(
            state.slug
        ).last_request_time
        > 0
    )


def test_group_endpoint_contract_and_auth(monkeypatch):
    monkeypatch.setattr(server.app.state.application.deployments, "catalog", config())
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "test-token"})
    ensure = AsyncMock(
        return_value=RunningDeployment(
            config=None, state=DeploymentSnapshot(slug="unused", replicas=1)
        )
    )
    monkeypatch.setattr(
        server.app.state.application.deployments, "ensure_running", ensure
    )
    # No lifespan: these HTTP contract checks must not touch real deployments.
    client = TestClient(server.app)
    url = "/deployments/group_foi_pipeline/warm-up"
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
        {"slug": "group_mixed", "deployments": ["custom_cpu", "foi_topic"]}
    ]
    registry = ResourceRegistry()
    load = Mock(
        side_effect=RuntimeError("private local detail") if fail_local else None
    )
    registry.register_factory(
        "custom_cpu", lambda: LocalResource("custom_cpu", load, Mock())
    )
    monkeypatch.setattr(server.app.state.application.deployments, "resources", registry)
    data["local"]["deployment"].append(
        {
            "slug": "custom_cpu",
            "loader": "presidio",
            "spacy_model": "en_core_web_sm",
            "score_threshold": 0.5,
        }
    )
    mixed = DeploymentCatalog.model_validate(data)
    monkeypatch.setattr(server.app.state.application.deployments, "catalog", mixed)
    remote = AsyncMock(
        return_value=RunningDeployment(
            config=None, state=DeploymentSnapshot(slug="foi_topic", replicas=1)
        )
    )
    monkeypatch.setattr(
        server.app.state.application.deployments, "ensure_running", remote
    )
    result = asyncio.run(
        server.app.state.application.deployments.ensure_group("group_mixed")
    )
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
    monkeypatch.setattr(server.app.state.application.deployments, "resources", registry)
    remote = AsyncMock(side_effect=AssertionError("No remote startup"))
    monkeypatch.setattr(
        server.app.state.application.deployments, "ensure_running", remote
    )
    result = asyncio.run(
        server.app.state.application.deployments.ensure_deployment("custom_cpu")
    )
    assert result.model_dump() == {
        "slug": "custom_cpu",
        "action": "warmed",
        "replicas": None,
    }
    assert resource.status().ready
    assert asyncio.run(
        server.app.state.application.deployments.ensure_local("custom_cpu")
    ).ready
    assert load.call_count == 2
    remote.assert_not_called()


def test_shipped_foi_group_includes_registered_local_resources():
    members = DeploymentCatalog.load().get_group("group_foi_pipeline").deployments
    assert members == [
        "presidio",
        "question_slice_head_cpu",
        "question_extractor_tokenizer",
        "question_slice",
        "foi_topic",
        "foi_categories_cpu",
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
    assert {
        foi.classifier,
        foi.head,
    } <= server.app.state.application.deployments.resources.names()
    cpu_members = catalog.get_group("group_foi_pipeline_cpu").deployments
    assert foi.classifier in cpu_members


@pytest.mark.parametrize("action", ["cool-down", "scale-to-zero"])
@pytest.mark.parametrize("failure", [None, "busy", "provider"])
def test_group_lifecycle_concurrent_results_and_auth(monkeypatch, action, failure):
    service = server.app.state.application.deployments
    monkeypatch.setattr(service, "catalog", config())
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "test-token"})
    entered = set()

    async def run_member(slug):
        entered.add(slug)
        # Both calls must begin before either completes.
        for _ in range(100):
            if len(entered) == 2:
                break
            await asyncio.sleep(0.001)
        assert len(entered) == 2
        if slug == "foi_topic":
            if failure == "busy":
                raise DeploymentServiceError(
                    status_code=409, detail="Deployment has requests in flight."
                )
            if failure == "provider":
                raise RuntimeError("private provider details")
        return 45.0 if action == "cool-down" else None

    operation = AsyncMock(side_effect=run_member)
    monkeypatch.setattr(
        service, "cooldown" if action == "cool-down" else "scale_to_zero", operation
    )
    client = TestClient(server.app)
    url = f"/deployments/group_foi_pipeline/{action}"
    assert client.post(url).status_code == 401
    operation.assert_not_called()
    response = client.post(url, headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    result = response.json()
    assert result["success"] is (failure is None)
    assert [member["slug"] for member in result["deployments"]] == [
        "question_slice",
        "foi_topic",
    ]
    assert result["deployments"][0]["success"]
    assert result["deployments"][1]["success"] is (failure is None)
    assert result["deployments"][0]["seconds_until_scale_to_zero"] == (
        45 if action == "cool-down" else None
    )
    assert "private" not in response.text
    if failure == "busy":
        assert "requests in flight" in result["deployments"][1]["error"]
    operation.reset_mock()
    assert (
        client.post(
            f"/deployments/group_missing/{action}",
            headers={"Authorization": "Bearer test-token"},
        ).status_code
        == 404
    )
    operation.assert_not_called()


@pytest.mark.parametrize("action", ["cool-down", "scale-to-zero"])
def test_group_lifecycle_handles_local_and_remote_members(monkeypatch, action):
    service = server.app.state.application.deployments
    monkeypatch.setattr(
        service,
        "catalog",
        config([{"slug": "group_mixed", "deployments": ["presidio", "foi_topic"]}]),
    )
    registry = ResourceRegistry()
    unload = Mock()
    resource = registry.register(LocalResource("presidio", Mock(), unload))
    resource.warmup()
    monkeypatch.setattr(service, "resources", registry)
    backend = SimpleNamespace(scale_to_zero=Mock())
    state = DeploymentSnapshot(slug="foi_topic", exists=True, replicas=1)
    monkeypatch.setattr(service, "get_backend", lambda slug: backend)
    monkeypatch.setattr(service.status, "ensure", lambda backend: state)
    monkeypatch.setattr(service.status, "refresh", Mock())
    service.status.set(state)
    result = asyncio.run(service.change_group("group_mixed", action))
    assert result.success
    assert [member.slug for member in result.deployments] == ["presidio", "foi_topic"]
    if action == "scale-to-zero":
        unload.assert_called_once()
        backend.scale_to_zero.assert_called_once()
        assert not resource.status().ready
    else:
        unload.assert_not_called()
        backend.scale_to_zero.assert_not_called()
        assert 0 <= result.deployments[0].seconds_until_scale_to_zero <= 60


@pytest.mark.parametrize("action", ["cool-down", "scale-to-zero"])
def test_group_busy_local_member_does_not_block_remote(monkeypatch, action):
    service = server.app.state.application.deployments
    monkeypatch.setattr(
        service,
        "catalog",
        config([{"slug": "group_mixed", "deployments": ["presidio", "foi_topic"]}]),
    )
    registry = ResourceRegistry()
    unload = Mock()
    resource = registry.register(LocalResource("presidio", Mock(), unload))
    resource.warmup()
    monkeypatch.setattr(service, "resources", registry)
    backend = SimpleNamespace(scale_to_zero=Mock())
    monkeypatch.setattr(service, "get_backend", lambda slug: backend)
    state = DeploymentSnapshot(slug="foi_topic", exists=False)
    monkeypatch.setattr(service.status, "ensure", lambda backend: state)
    monkeypatch.setattr(service.status, "refresh", Mock())
    with resource.use():
        result = asyncio.run(service.change_group("group_mixed", action))
    assert not result.success
    assert not result.deployments[0].success
    assert "in use" in result.deployments[0].error
    assert result.deployments[1].success
    unload.assert_not_called()


@pytest.mark.parametrize("slug", ["foi_pipeline", "group-foi_pipeline", "group_", ""])
def test_group_slug_requires_reserved_prefix_and_name(slug):
    with pytest.raises(ValidationError, match="group_"):
        config([{"slug": slug, "deployments": ["foi_topic"]}])


@pytest.mark.parametrize("kind", ["local", "exoscale"])
def test_individual_deployments_cannot_use_group_prefix(kind):
    data = DeploymentCatalog.load().model_dump()
    data[kind]["deployment"][0]["slug"] = "group_reserved"
    with pytest.raises(ValidationError, match="prefix is reserved"):
        DeploymentCatalog.model_validate(data)


def test_group_warmup_replica_override_rejected_before_work(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    ensure = AsyncMock()
    monkeypatch.setattr(
        server.app.state.application.deployments, "ensure_group", ensure
    )
    response = TestClient(server.app).post(
        "/deployments/group_foi_pipeline/warm-up?replicas=2"
    )
    assert response.status_code == 422
    ensure.assert_not_awaited()


def test_merged_warmup_unknown_group_returns_404_without_startup(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    ensure = AsyncMock()
    monkeypatch.setattr(
        server.app.state.application.deployments, "ensure_running", ensure
    )
    response = TestClient(server.app).post("/deployments/group_missing/warm-up")
    assert response.status_code == 404
    assert "group" in response.json()["detail"].lower()
    ensure.assert_not_awaited()
