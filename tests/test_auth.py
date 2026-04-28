from llm_management.server import app
from llm_management.settings import settings
from starlette.testclient import TestClient


def test_requests_are_unprotected_when_auth_token_is_empty(monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "")

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200


def test_requests_require_bearer_token_when_auth_token_is_set(monkeypatch):
    monkeypatch.setattr(settings, "auth_token", "secret-token")

    with TestClient(app) as client:
        missing_auth = client.get("/")
        wrong_auth = client.get("/", headers={"Authorization": "Bearer wrong-token"})
        ok_auth = client.get("/", headers={"Authorization": "Bearer secret-token"})

    assert missing_auth.status_code == 401
    assert missing_auth.headers["WWW-Authenticate"] == "Bearer"

    assert wrong_auth.status_code == 401
    assert wrong_auth.headers["WWW-Authenticate"] == "Bearer"

    assert ok_auth.status_code == 200
