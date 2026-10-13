"""Security boundaries tested without model loading or outbound connections."""

import asyncio

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management.api.auth import require_bearer_auth_when_enabled
from llm_management.api.classifiers import AutomaticResponseRequest
from llm_management.api.limits import RequestLimits
from llm_management.settings import Settings, settings


@pytest.mark.parametrize("token", ["", "  ", "é"])
def test_invalid_configured_tokens_fail_validation(token):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, auth_tokens={"client": token})


def test_empty_and_nonascii_bearer_rejected_even_if_settings_mutated(monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"client": ""})
    app = FastAPI()
    app.middleware("http")(require_bearer_auth_when_enabled)
    with TestClient(app) as client:
        assert (
            client.get("/private", headers={"Authorization": "Bearer"}).status_code
            == 401
        )
        assert (
            client.get(
                "/private", headers={b"authorization": b"Bearer \xff"}
            ).status_code
            == 401
        )


def test_automatic_text_limit():
    with pytest.raises(ValidationError):
        AutomaticResponseRequest(body="x" * 100_001)


def test_chunked_body_bound_and_admission(monkeypatch):
    monkeypatch.setattr(settings, "http_max_body_bytes", 8)
    monkeypatch.setattr(settings, "http_max_concurrent_requests", 1)

    async def run():
        called = []

        async def downstream(scope, receive, send):
            called.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestLimits(downstream)

        async def request(chunks, headers=()):
            messages = iter(chunks)
            output = []

            async def receive():
                return next(messages)

            async def send(message):
                output.append(message)

            await middleware({"type": "http", "headers": headers}, receive, send)
            return output[0]["status"]

        assert (
            await request(
                [
                    {"type": "http.request", "body": b"12345", "more_body": True},
                    {"type": "http.request", "body": b"6789", "more_body": False},
                ]
            )
            == 413
        )
        assert not called
        assert middleware.active == 0
        assert await request([], [(b"content-length", b"99")]) == 413
        middleware.active = 1
        assert await request([]) == 429
        middleware.active = 0
        assert await request([{"type": "http.request", "body": b"12345678"}]) == 200
        assert called[0]["body"] == b"12345678"
        assert middleware.active == 0

    asyncio.run(run())


def test_slow_upload_times_out_and_releases_admission(monkeypatch):
    monkeypatch.setattr(settings, "http_body_timeout_seconds", 0.01)

    async def run():
        async def downstream(*args):
            raise AssertionError("Timed out request reached downstream")

        async def receive():
            await asyncio.Event().wait()

        messages = []

        async def send(message):
            messages.append(message)

        middleware = RequestLimits(downstream)
        await middleware({"type": "http", "headers": []}, receive, send)
        assert messages[0]["status"] == 408
        assert middleware.active == 0

    asyncio.run(run())


@pytest.mark.parametrize("case", ["sources", "source_text", "questions", "total_text"])
def test_response_status_input_bounds(case):
    from llm_management.foi.examples import validation_examples
    from llm_management.foi.response_schemas import ResponseAnalysisInput

    body = validation_examples().responses[0].input.model_dump(mode="json")
    source = body["sources"][0]
    if case == "sources":
        body["sources"] = [{**source, "id": f"s{i}"} for i in range(65)]
    elif case == "source_text":
        source["text"] = "x" * 100_001
        source["availability"] = "visible"
    elif case == "questions":
        body["request"] = [
            {"question_id": f"q{i}", "text": "A question"} for i in range(1, 258)
        ]
    else:
        body["sources"] = [
            {**source, "id": f"s{i}", "availability": "visible", "text": "x" * 100_000}
            for i in range(3)
        ]
    with pytest.raises(ValidationError):
        ResponseAnalysisInput.model_validate(body)
