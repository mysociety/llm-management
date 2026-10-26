"""
Reviewed expectations, isolated model transport and replay evaluation reports.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl

from llm_management.deployments.api_config import APIModelDeployment
from llm_management.foi.email_finder.capture import ResearchCapture
from llm_management.foi.email_finder.replay import ReplayTrace
from llm_management.foi.email_finder.schemas import FOIResult


class ExpectedFinding(BaseModel):
    """
    Pin reviewed semantic expectations to the exact captured fixture bytes.
    """

    model_config = ConfigDict(extra="forbid")

    capture_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["found", "form_only", "not_found"]
    email_type: Literal["foi_eir", "eir_only", "generic"] | None = None
    emails: list[EmailStr] = Field(default_factory=list)
    evidence_urls: list[HttpUrl] = Field(default_factory=list)
    request_form_urls: list[HttpUrl] = Field(default_factory=list)
    review_notes: str = Field(min_length=1)


@dataclass
class EvaluationCase:
    """
    Load reviewed captures without contacting the recorded sources.
    """

    id: str
    capture: ResearchCapture
    expected: ExpectedFinding

    @classmethod
    def load(cls, directory: Path) -> EvaluationCase:
        content = (directory / "capture.json").read_bytes()
        capture = ResearchCapture.model_validate_json(content)
        expected = ExpectedFinding.model_validate_json(
            (directory / "expected.json").read_text()
        )
        assert hashlib.sha256(content).hexdigest() == expected.capture_sha256, (
            "Capture changed; review expectations before spending model tokens"
        )
        readable = [page for page in capture.pages if page.error is None]
        assert readable, "No successfully captured pages are available"
        for email in expected.emails:
            assert any(
                str(email).casefold() in page.content.casefold()
                and page.url in expected.evidence_urls
                for page in readable
            ), f"Expected email lacks captured evidence: {email}"
        if expected.status == "found":
            assert expected.emails and expected.email_type and expected.evidence_urls
        if expected.status == "form_only":
            assert expected.request_form_urls and expected.evidence_urls
        return cls(directory.name, capture, expected)


class EvaluationReport(BaseModel):
    """
    Retain model identity, latency, usage, complete result and replay diagnostics.
    """

    case_id: str
    model_slug: str
    deployment: APIModelDeployment
    started_at: datetime
    capture_sha256: str
    elapsed_seconds: float = 0
    result: FOIResult | None = None
    trace: ReplayTrace
    error: str | None = None


class ModelOnlyTransport(httpx2.AsyncBaseTransport):
    """
    Restrict evaluation HTTP traffic to the selected model service's base URL.
    """

    def __init__(
        self, base_url: HttpUrl, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self.base_url = httpx2.URL(str(base_url))
        self.path_prefix = self.base_url.path.rstrip("/") + "/"
        self.transport = (
            transport
            if transport is not None
            else httpx2.AsyncHTTPTransport(trust_env=False)
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        url = request.url
        if (
            url.scheme != self.base_url.scheme
            or url.host != self.base_url.host
            or url.port != self.base_url.port
            or not url.path.startswith(self.path_prefix)
        ):
            raise AssertionError(
                "Replay evaluation may contact only the selected model service"
            )
        return await self.transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self.transport.aclose()


def assert_finding_matches(result: FOIResult, expected: ExpectedFinding) -> None:
    """
    Compare route semantics while permitting different supported evidence quotes.
    """
    finding = result.finding
    assert finding.status == expected.status
    assert finding.email_type == expected.email_type
    if expected.status == "found":
        assert finding.email is not None
        assert str(finding.email).casefold() in {
            str(email).casefold() for email in expected.emails
        }
    else:
        assert finding.email is None
    if expected.status != "not_found":
        assert finding.evidence is not None
        assert finding.evidence.source_url in expected.evidence_urls
    if expected.status == "form_only":
        assert finding.request_form_url in expected.request_form_urls
