from llm_management.server import app
from llm_management.settings import ExoscaleCredentials, settings
from starlette.testclient import TestClient


def test_auth_tokens_are_loaded_from_json_mapping(monkeypatch):
    monkeypatch.setenv(
        "AUTH_TOKENS", '{"service-a":"first-token","service-b":"second-token"}'
    )

    loaded_settings = ExoscaleCredentials(_env_file=None)

    assert loaded_settings.auth_tokens == {
        "service-a": "first-token",
        "service-b": "second-token",
    }


def test_requests_are_unprotected_when_auth_tokens_are_empty(monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {})

    with TestClient(app) as client:
        response = client.get("/not-a-route")

    assert response.status_code == 404


def test_any_auth_token_enables_authentication(monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"service-a": "secret-token"})

    with TestClient(app) as client:
        missing_auth = client.get("/not-a-route")
        wrong_auth = client.get(
            "/not-a-route", headers={"Authorization": "Bearer wrong-token"}
        )
        ok_auth = client.get(
            "/not-a-route", headers={"Authorization": "Bearer secret-token"}
        )

    assert missing_auth.status_code == 401
    assert missing_auth.headers["WWW-Authenticate"] == "Bearer"

    assert wrong_auth.status_code == 401
    assert wrong_auth.headers["WWW-Authenticate"] == "Bearer"

    assert ok_auth.status_code == 404


def test_each_configured_auth_token_is_accepted(monkeypatch):
    monkeypatch.setattr(
        settings,
        "auth_tokens",
        {"service-a": "first-token", "service-b": "second-token"},
    )

    with TestClient(app) as client:
        first = client.get(
            "/not-a-route", headers={"Authorization": "Bearer first-token"}
        )
        second = client.get(
            "/not-a-route", headers={"Authorization": "Bearer second-token"}
        )

    assert first.status_code == 404
    assert second.status_code == 404


def test_health_and_api_docs_are_public(monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"service-a": "secret-token"})

    with TestClient(app) as client:
        health = client.get("/health")
        docs = client.get("/")
        openapi = client.get("/openapi.json")
        redoc = client.get("/redoc")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert docs.status_code == 200
    assert openapi.status_code == 200
    assert redoc.status_code == 200
