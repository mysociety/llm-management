"""Security boundaries tested without model loading or outbound connections."""

import asyncio

import httpx2
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management.api.auth import PUBLIC_PATHS, require_bearer_auth_when_enabled
from llm_management.api.classifiers import CorrespondenceClassificationRequest
from llm_management.api.limits import HTTPRequestPolicy, RequestLimits
from llm_management.http_app import create_app
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


@pytest.mark.parametrize("body", ["", "x" * 100_001])
def test_classifier_text_limits(body):
    with pytest.raises(ValidationError):
        CorrespondenceClassificationRequest(body=body)


@pytest.mark.parametrize("endpoint", ["sar_detection", "automatic_response_detection"])
def test_classifier_endpoints_reject_empty_text(http_app, endpoint):
    with TestClient(http_app) as client:
        response = client.post(f"/endpoint/foi/{endpoint}", json={"body": ""})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "body"]
    assert response.json()["detail"][0]["type"] == "string_too_short"


def test_chunked_body_bound_and_admission(monkeypatch):
    monkeypatch.setattr(settings, "http_max_body_bytes", 8)
    monkeypatch.setattr(settings, "http_max_concurrent_requests", 1)

    async def run():
        called = []

        async def downstream(scope, receive, send):
            called.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = RequestLimits(
            downstream, policy=HTTPRequestPolicy.from_settings(settings)
        )
        monkeypatch.setattr(settings, "http_max_body_bytes", 1_000_000)
        monkeypatch.setattr(settings, "http_max_concurrent_requests", 64)

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

        middleware = RequestLimits(
            downstream, policy=HTTPRequestPolicy.from_settings(settings)
        )
        await middleware({"type": "http", "headers": []}, receive, send)
        assert messages[0]["status"] == 408
        assert middleware.active == 0

    asyncio.run(run())


@pytest.mark.parametrize("path", sorted(PUBLIC_PATHS))
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_public_routes_bypass_buffering_and_admission(path, method):
    async def run():
        async def receive():
            raise AssertionError("Public request attempted to read an upload")

        async def downstream(scope, receive, send):
            assert middleware.active == occupied
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = RequestLimits(
            downstream,
            policy=HTTPRequestPolicy(
                max_body_bytes=8, max_concurrent_requests=1, body_timeout_seconds=30
            ),
        )
        for occupied in (0, 1):
            middleware.active = occupied
            messages = []

            async def send(message):
                messages.append(message)

            await middleware(
                {
                    "type": "http",
                    "http_version": "1.1",
                    "path": path,
                    "method": method,
                    "headers": [(b"content-length", b"1")],
                },
                receive,
                send,
            )
            assert messages[0]["status"] == 200
            assert middleware.active == occupied

    asyncio.run(run())


def test_public_routes_remain_responsive_when_authenticated_api_is_full(
    runtime, monkeypatch
):
    monkeypatch.setattr(settings, "auth_tokens", {"client": "test-token"})
    app = create_app(
        runtime,
        request_policy=HTTPRequestPolicy(
            max_body_bytes=8, max_concurrent_requests=1, body_timeout_seconds=30
        ),
    )

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        @app.get("/protected-capacity-check")
        async def protected():
            entered.set()
            await release.wait()
            return {"ok": True}

        async def public_request(path, method, headers):
            messages = []

            async def receive():
                # Simulate a client that never finishes sending its body.
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "root_path": "",
                "headers": headers,
                "server": ("test", 80),
                "client": ("192.0.2.1", 12345),
            }
            await asyncio.wait_for(app(scope, receive, send), timeout=2)
            return messages[0]["status"]

        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app),
            base_url="http://test",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            pending = asyncio.create_task(client.get("/protected-capacity-check"))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                assert (
                    await client.get("/protected-capacity-check")
                ).status_code == 429
                for path in sorted(PUBLIC_PATHS):
                    assert await public_request(path, "GET", []) == 200
                    expected_head = 405 if path == "/health" else 200
                    assert await public_request(path, "HEAD", []) == expected_head
                    assert (
                        await public_request(path, "POST", [(b"content-length", b"1")])
                        == 405
                    )
                    assert (
                        await public_request(path, "GET", [(b"content-length", b"1")])
                        == 200
                    )
                    assert (
                        await public_request(
                            path, "GET", [(b"transfer-encoding", b"chunked")]
                        )
                        == 200
                    )
                assert (
                    await client.get("/protected-capacity-check")
                ).status_code == 429
            finally:
                release.set()
                response = await asyncio.wait_for(pending, timeout=2)
            assert response.status_code == 200
            assert (await client.get("/protected-capacity-check")).status_code == 200

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
