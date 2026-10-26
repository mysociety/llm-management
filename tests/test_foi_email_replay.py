"""
Verify replay tools and network restrictions without paid calls or live scraping.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import httpx2
import pytest
from pydantic import HttpUrl
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from typer.testing import CliRunner

from llm_management.foi.email_finder import capture
from llm_management.foi.email_finder.replay import ReplayResearch
from llm_management.foi.email_finder.service import find_foi_email
from llm_management.web.pages import PageReader
from llm_management.web.search import TavilySearch
from tests.foi_email_evaluation import (
    EvaluationCase,
    ModelOnlyTransport,
    assert_finding_matches,
)

FIXTURES = Path(__file__).parent / "fixtures" / "foi_email"
CASE_IDS = ["croydon", "abbey_lane_primary", "aria", "east_dunbartonshire"]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_recorded_tools_use_the_normal_agent_without_live_fetches(
    case_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = EvaluationCase.load(FIXTURES / case_id)
    replay = ReplayResearch(case.capture)
    network = AsyncMock(side_effect=AssertionError("Replay accessed the network"))
    monkeypatch.setattr(httpx2.AsyncClient, "send", network)
    monkeypatch.setattr(PageReader, "fetch", network)
    monkeypatch.setattr(TavilySearch, "search", network)
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_web",
                        {"query": "A differently phrased authority search"},
                    ),
                    ToolCallPart(
                        "read_page", {"url": str(case.expected.evidence_urls[0])}
                    ),
                ]
            )
        assert case.capture.result is not None
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    case.capture.result.finding.model_dump(mode="json"),
                )
            ]
        )

    result = asyncio.run(
        find_foi_email(
            case.capture.authority,
            model=FunctionModel(respond),
            reader=replay,
            search_provider=replay,
        )
    )
    assert_finding_matches(result, case.expected)
    assert result.usage.search.provider == "replay"
    assert result.usage.search.requests == 1
    assert result.usage.search.credits == 0
    assert replay.trace.queries == ["A differently phrased authority search"]
    assert replay.trace.page_urls == [case.expected.evidence_urls[0]]
    network.assert_not_called()


def test_replay_is_query_independent_and_missing_pages_never_fall_back() -> None:
    case = EvaluationCase.load(FIXTURES / "aria")
    replay = ReplayResearch(case.capture)
    original = case.capture.model_dump_json()

    async def run() -> None:
        first = await replay.search("ARIA EIR contact")
        second = await replay.search("Advanced Research and Invention Agency requests")
        assert first.results == second.results
        assert first.query != second.query
        first.results.clear()
        assert (await replay.search("fresh result")).results
        missing = await replay.fetch(HttpUrl("https://aria.org.uk/not-in-the-capture"))
        assert missing.error and "fixture" in missing.error
        assert replay.trace.missing_urls == [missing.requested_url]
        assert case.capture.model_dump_json() == original

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "https://www.croydon.gov.uk/contact",
        "https://api.example.evil/v1/responses",
        "http://api.example/v1/responses",
        "https://api.example:8443/v1/responses",
        "https://api.example/v10/responses",
    ],
)
def test_model_transport_blocks_every_other_origin_and_path(url: str) -> None:
    sent: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        sent.append(str(request.url))
        return httpx2.Response(200)

    async def run() -> None:
        transport = ModelOnlyTransport(
            HttpUrl("https://api.example/v1/"), httpx2.MockTransport(respond)
        )
        async with httpx2.AsyncClient(transport=transport) as client:
            with pytest.raises(AssertionError, match="only the selected model"):
                await client.post(url)
            assert not sent
            assert (
                await client.post("https://api.example/v1/responses")
            ).status_code == 200
        assert sent == ["https://api.example/v1/responses"]

    asyncio.run(run())


def test_capture_cli_protects_existing_fixtures_before_making_paid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "capture.json"
    existing.write_text("preserved capture")
    live = AsyncMock(side_effect=AssertionError("Unexpected paid capture"))
    monkeypatch.setattr(capture, "capture_authority", live)
    result = CliRunner().invoke(
        capture.app, ["--name", "Example", "--output", str(existing)]
    )
    assert result.exit_code != 0
    assert existing.read_text() == "preserved capture"
    live.assert_not_called()
