import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from llm_management import server
from llm_management.deployments.types import RunningDeployment


@pytest.fixture
def clef_client(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {"client": "client-secret"})
    monkeypatch.setattr(server.app.state.service.cache, "all_active", lambda: [])
    ensure = AsyncMock(
        return_value=RunningDeployment(
            config=SimpleNamespace(model="Cloudflare/clef-flash"),
            state=SimpleNamespace(
                deployment_url="https://clef.test/v1/", api_key="upstream-secret"
            ),
        )
    )
    monkeypatch.setattr(server.app.state.service, "ensure_running", ensure)
    requests = []
    replies = []

    def handle(request):
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    real_client = httpx2.AsyncClient
    monkeypatch.setattr(
        httpx2,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx2.MockTransport(handle), **kwargs),
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
        httpx2.Response(429, json={"error": "busy"}, headers={"Retry-After": "2"})
    )
    body = b'{"model":"clef","state":{"text":"hello"},"questions":{},"images":[]}'
    response = post(client, "/deployments/clef/v1/systemone?tag=a&tag=b", content=body)
    assert response.status_code == 429
    assert response.json() == {"error": "busy"}
    assert response.headers["retry-after"] == "2"
    assert requests[0].content == body
    assert str(requests[0].url) == "https://clef.test/v1/systemone?tag=a&tag=b"
    assert requests[0].headers["authorization"] == "Bearer upstream-secret"


@pytest.mark.parametrize(
    "path", ["/deployments/clef/v1/systemone", "/endpoint/foi/immigration_detection/clef"]
)
def test_auth_rejects_before_contacting_clef(clef_client, path):
    client, requests, _ = clef_client
    assert client.post(path, json={"body": "visa"}).status_code == 401
    assert requests == []
    server.app.state.service.ensure_running.assert_not_awaited()


def test_selected_deployment_uses_shared_lifecycle(clef_client):
    client, requests, replies = clef_client
    replies.append(httpx2.Response(200, json={"answers": {}}))
    assert (
        post(client, "/deployments/clef_flash/v1/systemone?tag=a", json={}).status_code
        == 200
    )
    server.app.state.service.ensure_running.assert_awaited_once_with("clef_flash")
    assert str(requests[0].url) == "https://clef.test/v1/systemone?tag=a"


@pytest.mark.parametrize("label", ["IMM", "FOI"])
def test_immigration_choice_contract(clef_client, label):
    client, requests, replies = clef_client
    replies.append(
        httpx2.Response(
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
        "/endpoint/foi/immigration_detection/clef?deployment=clef_flash",
        json={"body": "My request text"},
    )
    assert response.status_code == 200
    assert response.json() == {"classification": label}
    payload = json.loads(requests[0].content)
    assert payload["model"] == "Cloudflare/clef-flash"
    server.app.state.service.ensure_running.assert_awaited_once_with("clef_flash")
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
        httpx2.Response(200, content=b"not json"),
        httpx2.Response(200, json={"answers": {}}),
        httpx2.Response(
            200,
            json={"answers": {"classification": {"type": "choice", "choice": "OTHER"}}},
        ),
        httpx2.Response(
            200, json={"answers": {"classification": {"type": "noul", "choice": "IMM"}}}
        ),
        httpx2.Response(500, text="private upstream details"),
    ],
)
def test_bad_classification_returns_502(clef_client, reply):
    client, _, replies = clef_client
    replies.append(reply)
    response = post(
        client, "/endpoint/foi/immigration_detection/clef", json={"body": "visa"}
    )
    assert response.status_code == 502
    assert "private" not in response.text


@pytest.mark.parametrize(
    "path", ["/deployments/clef/v1/systemone", "/endpoint/foi/immigration_detection/clef"]
)
@pytest.mark.parametrize(
    "error, status",
    [(httpx2.ConnectError("secret"), 503), (httpx2.ReadTimeout("secret"), 504)],
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
    response = post(client, path, json={"body": "visa"})
    assert response.status_code == status
    assert "secret" not in response.text


@pytest.mark.parametrize(
    "path", ["/deployments/clef/v1/systemone", "/endpoint/foi/immigration_detection/clef"]
)
def test_unknown_deployment_returns_404(clef_client, path):
    client, requests, _ = clef_client
    server.app.state.service.ensure_running.side_effect = HTTPException(
        status_code=404, detail="Unknown deployment"
    )
    response = post(client, path, json={"body": "visa"})
    assert response.status_code == 404
    assert requests == []


@pytest.mark.parametrize("text", ["", "x" * 100_001])
def test_invalid_immigration_input_does_not_contact_clef(clef_client, text):
    client, requests, _ = clef_client
    response = post(
        client, "/endpoint/foi/immigration_detection/clef", json={"body": text}
    )
    assert response.status_code == 422
    assert requests == []
    server.app.state.service.ensure_running.assert_not_awaited()


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
        httpx2.Response(
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
        client, "/endpoint/foi/immigration_detection/clef", json={"body": "visa"}
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "Clef returned an invalid classification"}


@pytest.mark.parametrize(
    "error, status",
    [(httpx2.ReadTimeout("secret"), 504), (httpx2.ConnectError("secret"), 503)],
)
def test_generic_proxy_transport_errors(clef_client, error, status):
    client, _, replies = clef_client
    replies.append(error)
    response = post(client, "/deployments/clef/v1/systemone", json={})
    assert response.status_code == status
    assert "secret" not in response.text


def test_generic_proxy_decoded_response_headers(clef_client):
    import gzip

    client, _, replies = clef_client
    replies.append(
        httpx2.Response(
            200,
            content=gzip.compress(b'{"answers": {}}'),
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
        )
    )
    response = post(client, "/deployments/clef/v1/systemone", json={})
    assert response.json() == {"answers": {}}
    assert "content-encoding" not in response.headers
    assert int(response.headers["content-length"]) == len(response.content)


def test_systemone_alias_is_removed(clef_client):
    client, requests, _ = clef_client
    assert post(client, "/v1/systemone", json={}).status_code == 404
    assert requests == []
    server.app.state.service.ensure_running.assert_not_awaited()
