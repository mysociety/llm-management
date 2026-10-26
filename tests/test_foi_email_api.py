"""
FOI email api contracts and behaviour.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.foi.email_finder.service import (
    FinderInputError,
    FOIFinder,
)
from llm_management.inference.errors import ClassifierBusy
from llm_management.inference.external import (
    ExternalFailure,
    ExternalUnavailable,
)


@pytest.mark.parametrize(
    "error,status",
    [
        (ExternalUnavailable("Missing credential"), 503),
        (ExternalFailure("Research failed"), 502),
        (TimeoutError(), 504),
        (FinderInputError("Website needed"), 422),
        (ClassifierBusy("Busy"), 429),
    ],
)
def test_route_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(FOIFinder, "find", AsyncMock(side_effect=error))
    client = TestClient(server.app)
    response = client.post("/endpoint/foi/find_email", json={"name": "Council"})
    assert response.status_code == status


@pytest.mark.parametrize("path", ["/endpoint/foi/find_email"])
def test_route_input_and_auth(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    method = AsyncMock()
    monkeypatch.setattr(FOIFinder, "find", method)
    client = TestClient(server.app)
    for body in [
        {"name": " "},
        {"name": "x" * 301},
        {"name": "Council", "body": "private correspondence"},
    ]:
        assert client.post(path, json=body).status_code == 422
    method.assert_not_called()
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "secret"})
    assert client.post(path, json={"name": "Council"}).status_code == 401
