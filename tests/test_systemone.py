import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from starlette.testclient import TestClient

from llm_management import server, systemone


@pytest.fixture
def clef_client(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {"client": "client-secret"})
    monkeypatch.setattr(server.cache, "all_active", lambda: [])
    ensure = AsyncMock(
        return_value=(
            SimpleNamespace(model="Cloudflare/clef-flash"),
            SimpleNamespace(
                deployment_url="https://clef.test/v1/", api_key="upstream-secret"
            ),
        )
    )
    monkeypatch.setattr(server, "ensure_running", ensure)
    requests = []
    replies = []

    def handle(request):
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        systemone.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    def handle_decision(request):
        reply = handle(request)
        return httpx2.Response(
            reply.status_code, content=reply.content, headers=dict(reply.headers)
        )

    real_decision_client = httpx2.AsyncClient
    monkeypatch.setattr(
        server.httpx2,
        "AsyncClient",
        lambda **kwargs: real_decision_client(
            transport=httpx2.MockTransport(handle_decision), **kwargs
        ),
    )
    with TestClient(server.app) as client:
        yield client, requests, replies


def post(client, path, **kwargs):
    return client.post(
        path, headers={"Authorization": "Bearer client-secret"}, **kwargs
    )


def test_proxy_preserves_payload_query_status_and_uses_upstream_auth(clef_client):
    client, requests, replies = clef_client
    replies.append(
        httpx.Response(429, json={"error": "busy"}, headers={"Retry-After": "2"})
    )
    body = b'{"model":"clef","state":{"text":"hello"},"questions":{},"images":[]}'
    response = post(client, "/v1/systemone?tag=a&tag=b", content=body)
    assert response.status_code == 429
    assert response.json() == {"error": "busy"}
    assert response.headers["retry-after"] == "2"
    assert requests[0].content == body
    assert str(requests[0].url) == "https://clef.test/v1/systemone?tag=a&tag=b"
    assert requests[0].headers["authorization"] == "Bearer upstream-secret"


@pytest.mark.parametrize(
    "path", ["/v1/systemone", "/agents/immigration_detection/clef"]
)
def test_auth_rejects_before_contacting_clef(clef_client, path):
    client, requests, _ = clef_client
    assert client.post(path, json={"request": "visa"}).status_code == 401
    assert requests == []
    server.ensure_running.assert_not_awaited()


def test_selected_deployment_uses_shared_lifecycle(clef_client):
    client, requests, replies = clef_client
    replies.append(httpx.Response(200, json={"answers": {}}))
    assert (
        post(client, "/v1/systemone?deployment=clef_flash&tag=a", json={}).status_code
        == 200
    )
    server.ensure_running.assert_awaited_once_with("clef_flash")
    assert str(requests[0].url) == "https://clef.test/v1/systemone?tag=a"


@pytest.mark.parametrize("label", ["IMM", "FOI"])
def test_immigration_choice_contract(clef_client, label):
    client, requests, replies = clef_client
    replies.append(
        httpx.Response(
            200,
            json={
                "model": "Cloudflare/clef-flash",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "answers": {
                    "classification": {
                        "type": "choice",
                        "choice": label,
                        "confidence": 0.9,
                        "probabilities": {
                            label: 0.9,
                            "FOI" if label == "IMM" else "IMM": 0.1,
                        },
                    }
                },
            },
        )
    )
    response = post(
        client,
        "/agents/immigration_detection/clef?deployment=clef_flash",
        json={"request": "My request text"},
    )
    assert response.status_code == 200
    assert response.json() == {"classification": label}
    payload = json.loads(requests[0].content)
    assert payload["model"] == "Cloudflare/clef-flash"
    server.ensure_running.assert_awaited_once_with("clef_flash")
    assert "My request text" in payload["state"]
    assert str(requests[0].url) == "https://clef.test/v1/systemone"
    assert requests[0].headers["authorization"] == "Bearer upstream-secret"
    assert list(payload["questions"]) == ["classification"]
    question = payload["questions"]["classification"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"IMM", "FOI"}


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"answers": {}}),
        httpx.Response(
            200,
            json={"answers": {"classification": {"type": "choice", "choice": "OTHER"}}},
        ),
        httpx.Response(
            200, json={"answers": {"classification": {"type": "noul", "choice": "IMM"}}}
        ),
        httpx.Response(500, text="private upstream details"),
    ],
)
def test_bad_classification_returns_502(clef_client, reply):
    client, _, replies = clef_client
    replies.append(reply)
    response = post(
        client, "/agents/immigration_detection/clef", json={"request": "visa"}
    )
    assert response.status_code == 502
    assert "private" not in response.text


@pytest.mark.parametrize(
    "path", ["/v1/systemone", "/agents/immigration_detection/clef"]
)
@pytest.mark.parametrize(
    "error, status",
    [(httpx.ConnectError("secret"), 503), (httpx.ReadTimeout("secret"), 504)],
)
def test_transport_errors(clef_client, path, error, status):
    client, _, replies = clef_client
    if path.endswith("/clef"):
        error = (
            httpx2.ReadTimeout("secret")
            if status == 504
            else httpx2.ConnectError("secret")
        )
    replies.append(error)
    response = post(client, path, json={"request": "visa"})
    assert response.status_code == status
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "path", ["/v1/systemone", "/agents/immigration_detection/clef"]
)
def test_unknown_deployment_returns_404(clef_client, path):
    client, requests, _ = clef_client
    server.ensure_running.side_effect = server.HTTPException(
        status_code=404, detail="Unknown deployment"
    )
    response = post(client, path, json={"request": "visa"})
    assert response.status_code == 404
    assert requests == []


@pytest.mark.parametrize("text", ["", "x" * 100_001])
def test_invalid_immigration_input_does_not_contact_clef(clef_client, text):
    client, requests, _ = clef_client
    response = post(
        client, "/agents/immigration_detection/clef", json={"request": text}
    )
    assert response.status_code == 422
    assert requests == []
    server.ensure_running.assert_not_awaited()


@pytest.mark.parametrize(
    "choice, probabilities",
    [
        ("OTHER", {"IMM": 0.9, "FOI": 0.1}),
        ("IMM", {"IMM": 0.9}),
        ("IMM", {"IMM": 0.9, "FOI": 0.9}),
    ],
)
def test_native_model_validates_choices_and_probabilities(
    clef_client, choice, probabilities
):
    client, _, replies = clef_client
    replies.append(
        httpx.Response(
            200,
            json={
                "model": "Cloudflare/clef-flash",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "answers": {
                    "classification": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.9,
                        "probabilities": probabilities,
                    }
                },
            },
        )
    )
    response = post(
        client, "/agents/immigration_detection/clef", json={"request": "visa"}
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "Clef returned an invalid classification"}
