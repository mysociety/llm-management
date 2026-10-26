"""
FOI email agent contracts and behaviour.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx2
import pytest
from pydantic import HttpUrl, ValidationError
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from llm_management.foi.email_finder.agent import ResearchContext, validate_evidence
from llm_management.foi.email_finder.schemas import FOIFinding, PublicAuthority
from llm_management.foi.email_finder.service import (
    find_foi_email,
)
from llm_management.web.crawl import CrawlPolicy
from llm_management.web.pages import PageReader
from llm_management.web.schemas import WebPage
from tests.foi_email_helpers import QUOTE, URL, finding


@pytest.mark.parametrize(
    "change",
    [
        {"evidence": {"source_url": "https://other.example/", "quote": QUOTE}},
        {"evidence": {"source_url": URL, "quote": "Invented foi@council.example"}},
        {"email": "other@council.example"},
    ],
)
@pytest.mark.parametrize("email_type", ["foi_eir", "eir_only", "generic"])
def test_unsupported_evidence_rejected(
    change: dict[str, object], email_type: str
) -> None:
    context = ResearchContext(
        reader=AsyncMock(spec=PageReader), search_provider=None, tool_limit=3
    )
    context.pages.append(
        WebPage(requested_url=HttpUrl(URL), url=HttpUrl(URL), content=QUOTE)
    )
    with pytest.raises(ModelRetry):
        validate_evidence(
            RunContext(deps=context, model=TestModel(), usage=RunUsage()),
            finding(email_type=email_type, **change),
        )


def test_email_cannot_be_substring_of_another_address() -> None:
    context = ResearchContext(
        reader=AsyncMock(spec=PageReader), search_provider=None, tool_limit=3
    )
    quote = "Send to prefixfoi@council.example.uk"
    context.pages.append(
        WebPage(requested_url=HttpUrl(URL), url=HttpUrl(URL), content=quote)
    )
    with pytest.raises(ModelRetry):
        validate_evidence(
            RunContext(deps=context, model=TestModel(), usage=RunUsage()),
            finding(evidence={"source_url": URL, "quote": quote}),
        )


def test_forms_and_not_found() -> None:
    with pytest.raises(ValidationError):
        finding(status="form_only", email=None, email_type=None)
    context = ResearchContext(
        reader=AsyncMock(spec=PageReader), search_provider=None, tool_limit=3
    )
    form = "https://council.example/form"
    context.pages.append(
        WebPage(
            requested_url=HttpUrl(URL),
            url=HttpUrl(URL),
            content="Use the form",
            links=[HttpUrl(form)],
        )
    )
    result = finding(
        status="form_only",
        email=None,
        email_type=None,
        request_form_url=form,
        evidence={"source_url": URL, "quote": "Use the form"},
    )
    assert (
        validate_evidence(
            RunContext(deps=context, model=TestModel(), usage=RunUsage()), result
        )
        == result
    )
    result.request_form_url = HttpUrl("https://invented.example/")
    with pytest.raises(ModelRetry):
        validate_evidence(
            RunContext(deps=context, model=TestModel(), usage=RunUsage()), result
        )
    empty = FOIFinding(status="not_found", explanation="Ambiguous")
    with pytest.raises(ModelRetry):
        validate_evidence(
            RunContext(
                deps=ResearchContext(AsyncMock(spec=PageReader), None, 3),
                model=TestModel(),
                usage=RunUsage(),
            ),
            empty,
        )
    assert (
        validate_evidence(
            RunContext(deps=context, model=TestModel(), usage=RunUsage()), empty
        )
        == empty
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"email_type": None},
        {"email_type": "unknown"},
        {
            "status": "not_found",
            "email": None,
            "email_type": "generic",
            "evidence": None,
        },
        {
            "status": "form_only",
            "email": None,
            "email_type": "generic",
            "request_form_url": URL,
        },
    ],
)
def test_email_type_is_required_only_with_a_found_email(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        finding(**updates)


def test_general_contact_fallback_survives_unreadable_policy() -> None:
    policy_url = "https://council.example/foi-policy.pdf"
    contact_url = "https://council.example/contact"
    quote = "Contact the parish council: clerk@council.example."
    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls < 3:
            url = policy_url if model_calls == 1 else contact_url
            return ModelResponse(parts=[ToolCallPart("read_page", {"url": url})])
        output = finding(
            email="clerk@council.example",
            email_type="generic",
            evidence={"source_url": contact_url, "quote": quote},
            explanation="The FOI policy could not be read; the official contact page publishes this general mailbox.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(info.output_tools[0].name, output.model_dump(mode="json"))
            ]
        )

    def page_response(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        if str(request.url) == policy_url:
            return httpx2.Response(
                200,
                content=b"%PDF-placeholder",
                headers={"Content-Type": "application/pdf"},
            )
        return httpx2.Response(200, text=quote, headers={"Content-Type": "text/plain"})

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(page_response)
        ) as client:
            result = await find_foi_email(
                PublicAuthority(
                    name="Example Parish Council",
                    website=HttpUrl("https://council.example"),
                ),
                model=FunctionModel(respond),
                reader=PageReader(client, policy=CrawlPolicy(interval=0)),
            )
        assert result.finding.status == "found"
        assert result.finding.email == "clerk@council.example"
        assert result.finding.email_type == "generic"
        assert result.finding.evidence is not None
        assert result.finding.evidence.quote == quote
        assert result.pages[0].error is not None
        assert result.pages[1].error is None
        assert result.usage.model.requests == 3

    asyncio.run(run())


def test_eir_only_route_preserves_scope_and_evidence() -> None:
    quote = (
        "This agency is not subject to FOI. For requests under the Environmental "
        "Information Regulations, email EIR@agency.example."
    )
    model_calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart("read_page", {"url": URL})])
        output = finding(
            email="EIR@agency.example",
            email_type="eir_only",
            evidence={"source_url": URL, "quote": quote},
            explanation="The official page publishes an EIR route and explicitly excludes FOI coverage.",
        )
        return ModelResponse(
            parts=[
                ToolCallPart(info.output_tools[0].name, output.model_dump(mode="json"))
            ]
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: (
                    httpx2.Response(404)
                    if request.url.path == "/robots.txt"
                    else httpx2.Response(
                        200, text=quote, headers={"Content-Type": "text/plain"}
                    )
                )
            )
        ) as client:
            result = await find_foi_email(
                PublicAuthority(name="Example Agency", website=HttpUrl(URL)),
                model=FunctionModel(respond),
                reader=PageReader(client, policy=CrawlPolicy(interval=0)),
            )
        assert result.finding.status == "found"
        assert result.finding.email == "EIR@agency.example"
        assert result.finding.email_type == "eir_only"
        assert result.finding.evidence is not None
        assert result.finding.evidence.quote == quote
        assert result.model_dump(mode="json")["finding"]["email_type"] == "eir_only"

    asyncio.run(run())


@pytest.mark.parametrize(
    "href,email",
    [
        ("mailto:foi%40council.example", "foi@council.example"),
        (
            "mailto:foi%2Bteam%40council.example?subject=FOI%20request",
            "foi+team@council.example",
        ),
    ],
)
def test_encoded_mailto_addresses_support_evidence(href: str, email: str) -> None:
    from llm_management.foi.email_finder.agent import validate_evidence
    from llm_management.foi.email_finder.schemas import FOIFinding

    async def run() -> None:
        html = f'<p>For FOI requests, <a href="{href}">email us</a>.</p>'
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: (
                    httpx2.Response(404)
                    if request.url.path == "/robots.txt"
                    else httpx2.Response(
                        200, text=html, headers={"Content-Type": "text/html"}
                    )
                )
            )
        ) as client:
            page = await PageReader(client, policy=CrawlPolicy(interval=0)).fetch(
                HttpUrl(URL)
            )
        assert email in page.content
        assert "subject=" not in page.content
        context = ResearchContext(
            reader=AsyncMock(spec=PageReader),
            search_provider=None,
            tool_limit=1,
            pages=[page],
        )
        finding = FOIFinding.model_validate(
            {
                "status": "found",
                "email": email,
                "email_type": "foi_eir",
                "evidence": {"source_url": URL, "quote": page.content.strip()},
                "explanation": "Published FOI instructions",
            }
        )
        assert (
            validate_evidence(
                RunContext(deps=context, model=TestModel(), usage=RunUsage()), finding
            ).email
            == email
        )

    asyncio.run(run())


def test_search_failure_redacts_provider_errors() -> None:
    from llm_management.foi.email_finder.agent import search_web
    from llm_management.web.search import TavilySearch

    async def run() -> None:
        client = AsyncMock(spec=httpx2.AsyncClient)
        client.post.side_effect = RuntimeError("secret-value")
        context = ResearchContext(
            reader=AsyncMock(spec=PageReader),
            search_provider=TavilySearch(client, "search-secret"),
            tool_limit=5,
        )
        failure = await search_web(
            RunContext(deps=context, model=TestModel(), usage=RunUsage()),
            "Council FOI",
        )
        assert failure.error == "Search failed (RuntimeError)"
        assert failure.credits is None
        assert "secret-value" not in failure.model_dump_json()

    asyncio.run(run())
