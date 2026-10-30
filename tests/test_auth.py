import pytest
from starlette.testclient import TestClient

from llm_management.settings import Settings, settings


def test_auth_tokens_are_loaded_from_json_mapping(monkeypatch):
    monkeypatch.setenv(
        "AUTH_TOKENS", '{"service-a":"first-token","service-b":"second-token"}'
    )

    loaded_settings = Settings(_env_file=None)

    assert loaded_settings.auth_tokens == {
        "service-a": "first-token",
        "service-b": "second-token",
    }


def test_local_testing_explicitly_allows_empty_auth(http_app, monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {})
    monkeypatch.setattr(settings, "allow_unauthenticated", True)

    with TestClient(http_app) as client:
        response = client.get("/not-a-route")

    assert response.status_code == 404


def test_unauthenticated_mode_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    assert Settings(_env_file=None).allow_unauthenticated is False


def test_empty_auth_without_opt_in_prevents_startup(http_app, monkeypatch):
    monkeypatch.setattr(settings, "allow_unauthenticated", False)
    with pytest.raises(RuntimeError, match="AUTH_TOKENS must be configured"):
        with TestClient(http_app):
            pass


def test_empty_auth_fails_closed_without_lifespan(http_app, monkeypatch):
    monkeypatch.setattr(settings, "allow_unauthenticated", False)
    client = TestClient(http_app)
    assert client.get("/not-a-route").status_code == 401


def test_configured_auth_starts_without_testing_opt_in(http_app, monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"client": "secret"})
    monkeypatch.setattr(settings, "allow_unauthenticated", False)
    with TestClient(http_app) as client:
        assert client.get("/not-a-route").status_code == 401
        assert (
            client.get(
                "/not-a-route", headers={"Authorization": "Bearer secret"}
            ).status_code
            == 404
        )


@pytest.mark.parametrize("enabled", ["true", "false"])
def test_unauthenticated_mode_can_be_set_in_dotenv(tmp_path, monkeypatch, enabled):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"ALLOW_UNAUTHENTICATED={enabled}\n")
    assert Settings(_env_file=env_file).allow_unauthenticated is (enabled == "true")


def test_any_auth_token_enables_authentication(http_app, monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"service-a": "secret-token"})

    with TestClient(http_app) as client:
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


def test_each_configured_auth_token_is_accepted(http_app, monkeypatch):
    monkeypatch.setattr(
        settings,
        "auth_tokens",
        {"service-a": "first-token", "service-b": "second-token"},
    )

    with TestClient(http_app) as client:
        first = client.get(
            "/not-a-route", headers={"Authorization": "Bearer first-token"}
        )
        second = client.get(
            "/not-a-route", headers={"Authorization": "Bearer second-token"}
        )

    assert first.status_code == 404
    assert second.status_code == 404


def test_health_and_api_docs_are_public(http_app, monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"service-a": "secret-token"})

    with TestClient(http_app) as client:
        health = client.get("/health")
        docs = client.get("/")
        openapi = client.get("/openapi.json")
        redoc = client.get("/redoc")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert docs.status_code == 200
    assert openapi.status_code == 200
    assert redoc.status_code == 200


def test_swagger_style_authorization_reaches_endpoint(http_app, monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {"docs-user": "docs-test-token"})
    with TestClient(http_app) as client:
        path = "/mock/endpoint/misc/capital_city"
        body = {"country": "France"}
        assert client.post(path, json=body).status_code == 401
        assert (
            client.post(
                path, json=body, headers={"Authorization": "Bearer incorrect"}
            ).status_code
            == 401
        )
        response = client.post(
            path, json=body, headers={"Authorization": "Bearer docs-test-token"}
        )
        assert response.status_code == 200
        assert response.json()["city"] == "Fake Paris"


def test_documented_security_does_not_require_token_in_open_mode(http_app, monkeypatch):
    monkeypatch.setattr(settings, "auth_tokens", {})
    with TestClient(http_app) as client:
        response = client.post(
            "/mock/endpoint/misc/capital_city", json={"country": "France"}
        )
        assert response.status_code == 200
