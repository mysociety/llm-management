"""
External API configuration, credentials and model transports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx2
import pytest
from pydantic import HttpUrl, ValidationError

from llm_management.deployments.api_config import (
    APIDeployments,
    APIModelDeployment,
    APIModelService,
    EmailFinderConfig,
)
from llm_management.deployments.catalog import DeploymentCatalog
from llm_management.inference.external import (
    ExternalUnavailable,
    build_model,
    credential,
)
from llm_management.settings import Settings


def test_credentials_profiles_and_redaction() -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        openai_api_key="environment-value",
        tavily_api_key="",
    )
    assert credential("openai", settings) == "environment-value"
    with pytest.raises(ExternalUnavailable):
        credential("../secret", settings)
    with pytest.raises(ExternalUnavailable):
        credential("missing", settings)
    with pytest.raises(ExternalUnavailable):
        credential("tavily", settings)
    assert "environment-value" not in repr(settings)
    assert "environment-value" not in settings.model_dump_json()


def test_model_service_can_be_switched() -> None:
    async def run() -> None:
        async with httpx2.AsyncClient() as client:
            model = build_model(
                APIModelDeployment(
                    slug="alternative", service="alternative", model="alternative"
                ),
                APIModelService(
                    provider="openai_compatible",
                    protocol="chat",
                    base_url=HttpUrl("https://provider.example/v1"),
                    credential="alternative",
                ),
                api_key="test",
                client=client,
            )
            assert model.model_name == "alternative"
            assert str(model.client.base_url) == "https://provider.example/v1/"

    asyncio.run(run())


def test_responses_request_preserves_deployment_settings(
    api_catalog: DeploymentCatalog,
) -> None:
    import json

    from pydantic_ai import Agent

    assert api_catalog.foi_email_finder is not None
    deployment = api_catalog.api.get(api_catalog.foi_email_finder.deployment)
    service = api_catalog.api.model_service(deployment.service)
    seen: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        payload = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert payload["model"] == deployment.model
        assert payload["reasoning"]["effort"] == deployment.reasoning_effort
        assert payload["max_output_tokens"] == deployment.max_tokens
        assert payload["store"] is False
        return httpx2.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": deployment.model,
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Done", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            model = build_model(deployment, service, api_key="test", client=client)
            assert (await Agent(model).run("Hello")).output == "Done"

    asyncio.run(run())
    assert len(seen) == 1


def test_catalog_validation_rejects_invalid_external_bindings(
    api_catalog: DeploymentCatalog,
) -> None:
    config = api_catalog.model_dump(mode="json")
    changes = [
        lambda data: data["foi_email_finder"].update(deployment="unknown"),
        lambda data: data["foi_email_finder"].update(search="unknown"),
        lambda data: data["api"]["deployment"][0].update(service="unknown"),
        lambda data: data["api"]["deployment"][0].update(service="tavily"),
        lambda data: data["api"]["deployment"][0].update(slug="web_test"),
        lambda data: data["api"]["search"][0].update(service="openai"),
    ]
    from copy import deepcopy

    for change in changes:
        data = deepcopy(config)
        change(data)
        with pytest.raises(ValidationError):
            DeploymentCatalog.model_validate(data)


def test_api_models_can_share_service_with_distinct_reasoning_settings() -> None:
    config = DeploymentCatalog(
        api=APIDeployments(
            service={
                "compatible": APIModelService(
                    provider="openai_compatible",
                    protocol="chat",
                    base_url=HttpUrl("https://provider.example/v1"),
                    credential="compatible",
                )
            },
            deployment=[
                APIModelDeployment(
                    slug="quick",
                    service="compatible",
                    model="example-model",
                    reasoning_effort="none",
                ),
                APIModelDeployment(
                    slug="thorough",
                    service="compatible",
                    model="example-model",
                    reasoning_effort="high",
                ),
            ],
        ),
        foi_email_finder=EmailFinderConfig(deployment="quick"),
    )
    assert config.api.get("quick").reasoning_effort == "none"
    assert config.api.get("thorough").reasoning_effort == "high"


def test_individual_provider_keys_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "model-key")
    monkeypatch.setenv("TAVILY_API_KEY", "search-key")
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert credential("openai", settings) == "model-key"
    assert credential("tavily", settings) == "search-key"


def test_individual_provider_keys_load_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=model-key\nTAVILY_API_KEY=search-key\n")
    settings = Settings(_env_file=dotenv)  # pyright: ignore[reportCallIssue]
    assert credential("openai", settings) == "model-key"
    assert credential("tavily", settings) == "search-key"


@pytest.mark.parametrize(
    "model_field,value",
    [("protocol", "responses"), ("base_url", "https://example.com")],
)
def test_search_service_rejects_model_settings(model_field: str, value: str) -> None:
    from llm_management.deployments.api_config import APISearchService

    with pytest.raises(ValidationError):
        APISearchService.model_validate(
            {"provider": "tavily", "credential": "tavily", model_field: value}
        )


def test_api_slugs_cannot_overlap_local_resources(
    api_catalog: DeploymentCatalog,
) -> None:
    data = api_catalog.model_dump(mode="json")
    data["local"] = {
        "deployment": [
            {
                "slug": "research_test",
                "loader": "presidio",
                "spacy_model": "unused",
                "score_threshold": 0.5,
            }
        ]
    }
    with pytest.raises(ValidationError, match="distinct names"):
        DeploymentCatalog.model_validate(data)

