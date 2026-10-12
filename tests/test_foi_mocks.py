"""
Verify mock contracts, gold payload preservation and absence of inference work.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.api import mocks
from llm_management.foi.examples import ExtractionExpectation, validation_examples
from llm_management.foi.response_schemas import ExtractionOutput
from llm_management.foi.schemas import InformationRequestResult, QuestionSliceResult


@pytest.fixture
def client(monkeypatch):
    """
    Fail if any mock route starts deployments or loads a local model.
    """
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    service = server.app.state.service
    monkeypatch.setattr(service.cache, "all_active", lambda: [])
    monkeypatch.setattr(
        service,
        "ensure_running",
        AsyncMock(side_effect=AssertionError("Mock started a deployment")),
    )
    resources = server.app.state.application.foi
    for name in ("classifier", "head", "sanitizer", "tokenizer", "response_tokenizer"):
        if hasattr(resources, name):
            monkeypatch.setattr(
                resources,
                name,
                Mock(side_effect=AssertionError("Mock loaded a local resource")),
            )
    with TestClient(server.app) as connection:
        yield connection


@pytest.mark.parametrize(
    "example", validation_examples().requests, ids=lambda case: case.id
)
@pytest.mark.parametrize("backend", ["cpu", "exoscale"])
def test_mock_extraction_preserves_gold(client, monkeypatch, example, backend):
    """
    Every possible request example produces a typed result without inference.
    """
    monkeypatch.setattr(mocks.random, "choice", lambda pool: example)
    response = client.post(
        "/mock/agents/foi_structure/extract",
        params={"backend": backend},
        json={"request": "Unrelated integration input."},
    )
    assert response.status_code == 200, response.text
    result = QuestionSliceResult.model_validate(response.json())
    assert ExtractionExpectation.from_result(result) == example.expected
    assert result.extraction_backend == backend
    assert result.extraction_model == "mock:validation-examples"
    assert result.unit_predictions == []


@pytest.mark.parametrize(
    "example", validation_examples().requests, ids=lambda case: case.id
)
def test_mock_full_result_preserves_gold(client, monkeypatch, example):
    """
    Extraction, topics and original text come from the same selected example.
    """
    monkeypatch.setattr(mocks.random, "choice", lambda pool: example)
    response = client.post(
        "/mock/agents/foi_structure", json={"request": "Different submitted request."}
    )
    assert response.status_code == 200, response.text
    result = InformationRequestResult.model_validate(response.json())
    assert ExtractionExpectation.from_result(result) == example.expected
    assert result.classification == example.classification
    assert result.request_text == example.input.request
    assert result.classification_model == "mock:validation-examples"


@pytest.mark.parametrize(
    "example", validation_examples().responses, ids=lambda case: case.id
)
def test_mock_response_returns_unchanged_example(client, monkeypatch, example):
    """
    Accept unrelated valid input while retaining the chosen example's source IDs.
    """
    body = validation_examples().responses[0].input.model_copy(deep=True)
    body.request.questions[0].question_id = "q9"
    body.sources[0].id = "unrelated_source"
    monkeypatch.setattr(mocks.random, "choice", lambda pool: example)
    response = client.post(
        "/mock/agents/foi_response_analysis", json=body.model_dump(mode="json")
    )
    assert response.status_code == 200, response.text
    output = ExtractionOutput.model_validate(response.json())
    assert output == example.expected
    assert response.json() == example.expected.model_dump(mode="json")


@pytest.mark.parametrize(
    "path", ["foi_structure/extract", "foi_structure", "foi_response_analysis"]
)
def test_mock_rejects_invalid_input(client, path):
    """
    Mirror the real endpoints' request validation rather than accept arbitrary JSON.
    """
    assert client.post("/mock/agents/" + path, json={}).status_code == 422


def test_mock_rejects_invalid_backend(client):
    """
    Enforce the same extraction backend enumeration even without inference.
    """
    assert (
        client.post(
            "/mock/agents/foi_structure/extract?backend=other",
            json={"request": "Hello"},
        ).status_code
        == 422
    )


def test_mock_uses_normal_authentication(client, monkeypatch):
    """
    Require configured bearer authentication for all three mock routes.
    """
    monkeypatch.setattr(
        server.settings, "auth_tokens", {"integration": "mock-test-token"}
    )
    bodies = {
        "foi_structure/extract": {"request": "Hello"},
        "foi_structure": {"request": "Hello"},
        "foi_response_analysis": validation_examples()
        .responses[0]
        .input.model_dump(mode="json"),
    }
    for path, body in bodies.items():
        url = "/mock/agents/" + path
        assert client.post(url, json=body).status_code == 401
        response = client.post(
            url, json=body, headers={"Authorization": "Bearer mock-test-token"}
        )
        assert response.status_code == 200
