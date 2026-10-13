"""Security boundaries tested without model loading or outbound connections."""

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.testclient import TestClient

from llm_management.api.auth import require_bearer_auth_when_enabled
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
