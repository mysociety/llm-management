"""
FOI email service contracts and behaviour.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx2
import pytest
from pydantic import HttpUrl
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from llm_management.deployments.api_config import (
    APIDeployments,
    APIModelDeployment,
    APIModelService,
    EmailFinderConfig,
)
from llm_management.deployments.catalog import DeploymentCatalog
from llm_management.foi.email_finder.schemas import FOIFinding, PublicAuthority
from llm_management.foi.email_finder.service import (
    FOIFinder,
    find_foi_email,
)
from llm_management.inference.errors import ClassifierBusy
from llm_management.inference.external import (
    ExternalFailure,
    ExternalUnavailable,
)
from llm_management.settings import Settings
from llm_management.web.crawl import CrawlPolicy
from llm_management.web.pages import PageReader
from tests.foi_email_helpers import QUOTE, URL, finding


def test_complete_tool_loop() -> None:
    calls: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("read_page", {"url": URL})],
                usage=RequestUsage(
                    input_tokens=100,
                    output_tokens=10,
                    cache_read_tokens=20,
                    cache_write_tokens=5,
                    details={"reasoning_tokens": 3},
                ),
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name, finding().model_dump(mode="json")
                )
            ],
            usage=RequestUsage(
                input_tokens=150,
                output_tokens=20,
                cache_read_tokens=50,
                cache_write_tokens=10,
                details={"reasoning_tokens": 5},
            ),
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: (
                    httpx2.Response(404)
                    if request.url.path == "/robots.txt"
                    else httpx2.Response(
                        200,
                        text=f"<p>{QUOTE}</p>",
                        headers={"Content-Type": "text/html"},
                    )
                )
            )
        ) as client:
            result = await find_foi_email(
                PublicAuthority(name="Council", website=HttpUrl(URL)),
                model=FunctionModel(respond),
                reader=PageReader(client, policy=CrawlPolicy(interval=0)),
            )
            assert result.finding.email == "foi@council.example"
            assert result.pages[0].url == HttpUrl(URL)
            assert result.checked_at.tzinfo is not None
            assert not client.is_closed
            assert result.usage.model.requests == 2
            assert result.usage.model.input_tokens == 250
            assert result.usage.model.cached_input_tokens == 70
            assert result.usage.model.cache_write_tokens == 15
            assert result.usage.model.output_tokens == 30
            assert result.usage.model.reasoning_tokens == 8
            assert result.usage.search.provider is None
            assert result.usage.search.requests == 0
            assert result.usage.search.credits == 0
            payload = result.model_dump(mode="json")
            assert payload["usage"]["model"]["output_tokens"] == 30

    asyncio.run(run())
    assert len(calls) == 2


def test_budget_exhaustion_is_operational_failure() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("read_page", {"url": URL})])

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: (
                    httpx2.Response(404)
                    if request.url.path == "/robots.txt"
                    else httpx2.Response(200, text=QUOTE)
                )
            )
        ) as client:
            with pytest.raises(ExternalFailure):
                await find_foi_email(
                    PublicAuthority(name="Council", website=HttpUrl(URL)),
                    model=FunctionModel(respond),
                    reader=PageReader(client, policy=CrawlPolicy(interval=0)),
                    request_limit=1,
                )

    asyncio.run(run())


def test_concurrency_limit_and_counter_reset(api_catalog: DeploymentCatalog) -> None:
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        openai_api_key="",
        tavily_api_key="",
    )
    finder = FOIFinder(settings, api_catalog, active=4)

    async def run() -> None:
        with pytest.raises(ClassifierBusy):
            await finder.find(PublicAuthority(name="Council"))
        finder.active = 0
        with pytest.raises(ExternalUnavailable):
            await finder.find(PublicAuthority(name="Council"))
        assert finder.active == 0

    asyncio.run(run())


def test_invalid_evidence_retries_model() -> None:
    calls: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(
                parts=[ToolCallPart("read_page", {"url": URL})],
                usage=RequestUsage(input_tokens=30, output_tokens=5),
            )
        output = finding().model_dump(mode="json")
        if len(calls) == 2:
            output["evidence"]["quote"] = "Invented instructions"
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, output)],
            usage=RequestUsage(input_tokens=30, output_tokens=5),
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: (
                    httpx2.Response(404)
                    if request.url.path == "/robots.txt"
                    else httpx2.Response(
                        200, text=QUOTE, headers={"Content-Type": "text/plain"}
                    )
                )
            )
        ) as client:
            result = await find_foi_email(
                PublicAuthority(name="Council", website=HttpUrl(URL)),
                model=FunctionModel(respond),
                reader=PageReader(client, policy=CrawlPolicy(interval=0)),
            )
            assert result.finding.evidence is not None
            assert result.finding.evidence.quote == QUOTE
            assert result.usage.model.requests == 3
            assert result.usage.model.input_tokens == 90
            assert result.usage.model.output_tokens == 15
            assert result.usage.model.reasoning_tokens is None

    asyncio.run(run())
    assert len(calls) == 3


def test_research_timeout_releases_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    config = DeploymentCatalog(
        api=APIDeployments(
            service={"test": APIModelService(provider="openai", credential="test")},
            deployment=[APIModelDeployment(slug="test", service="test", model="test")],
        ),
        foi_email_finder=EmailFinderConfig(deployment="test", timeout_seconds=0.001),
    )

    async def stalled(*args: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(FOIFinder, "_configured_find", stalled)
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    finder = FOIFinder(settings, config)

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await finder.find(PublicAuthority(name="Council", website=HttpUrl(URL)))
        assert finder.active == 0

    asyncio.run(run())


def test_finder_uses_owning_catalog_and_optional_configuration(
    api_catalog: DeploymentCatalog,
) -> None:
    from llm_management.bootstrap import assemble_runtime

    first = api_catalog
    second = DeploymentCatalog()
    runtimes = [assemble_runtime(first), assemble_runtime(second)]
    assert runtimes[0].foi_finder.catalog is first
    assert runtimes[1].foi_finder.catalog is second
    assert runtimes[0].foi_finder.catalog.api.get("research_test").service == "openai"

    async def run() -> None:
        with pytest.raises(ExternalUnavailable, match="not configured"):
            await runtimes[1].foi_finder.find(PublicAuthority(name="Council"))

    asyncio.run(run())


@pytest.mark.parametrize(
    "credits,fail_last,expected_credits",
    [
        ([1, 2], False, 3),
        ([1, None], False, None),
        ([1, None], True, None),
    ],
)
def test_search_usage_counts_requests_and_preserves_unknown_credits(
    credits: list[int | None],
    fail_last: bool,
    expected_credits: int | None,
) -> None:
    import json

    from llm_management.web.search import TavilySearch

    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("search_web", {"query": "first"}),
                    ToolCallPart("search_web", {"query": "second"}),
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    FOIFinding(
                        status="not_found", explanation="No official evidence"
                    ).model_dump(mode="json"),
                )
            ]
        )

    def search_response(request: httpx2.Request) -> httpx2.Response:
        query = json.loads(request.content)["query"]
        index = 0 if query == "first" else 1
        if index == 1 and fail_last:
            return httpx2.Response(503)
        return httpx2.Response(
            200, json={"results": [], "usage": {"credits": credits[index]}}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(search_response)
        ) as client:
            model = FunctionModel(respond)
            result = await find_foi_email(
                PublicAuthority(name="Council"),
                model=model,
                reader=AsyncMock(spec=PageReader),
                search_provider=TavilySearch(client, "test"),
            )
        assert result.usage.model.provider == model.system
        assert result.usage.model.model == model.model_name
        assert result.usage.search.provider == "tavily"
        assert result.usage.search.requests == 2
        assert result.usage.search.credits == expected_credits

    asyncio.run(run())


def test_finder_identifies_itself_on_page_and_robots_requests(
    api_catalog: DeploymentCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    from llm_management.foi.email_finder import service

    assert api_catalog.foi_email_finder is not None
    api_catalog.foi_email_finder.search = None
    seen: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers["User-Agent"])
        return httpx2.Response(
            404 if request.url.path == "/robots.txt" else 200,
            text=QUOTE,
            headers={"Content-Type": "text/plain"},
        )

    def model_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("read_page", {"url": URL})])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name, finding().model_dump(mode="json")
                )
            ]
        )

    monkeypatch.setattr(
        service, "build_model", Mock(return_value=FunctionModel(model_response))
    )
    monkeypatch.setattr(
        service, "PublicWebTransport", Mock(return_value=httpx2.MockTransport(respond))
    )
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        openai_api_key="test-key",
    )
    finder = FOIFinder(settings, api_catalog, crawl_policy=CrawlPolicy(interval=0))
    result = asyncio.run(
        finder.find(PublicAuthority(name="Council", website=HttpUrl(URL)))
    )
    assert result.finding.email == "foi@council.example"
    assert seen == [service.USER_AGENT] * 2
