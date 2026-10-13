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
from llm_management.foi.schemas import ExtractQuestionsResponse, QuestionSliceResult


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
        "/mock/endpoint/foi/extract_questions/extract",
        params={"backend": backend},
        json={"body": "Unrelated integration input."},
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
        "/mock/endpoint/foi/extract_questions",
        json={"body": "Different submitted request."},
    )
    assert response.status_code == 200, response.text
    result = ExtractQuestionsResponse.model_validate(response.json())
    assert ExtractionExpectation.from_result(result) == example.expected
    assert [q.model_dump(exclude={"text"}) for q in result.questions] == [
        q.model_dump() for q in example.classification.questions
    ]
    assert result.request_topics == example.classification.request_topics
    assert result.request_text == example.input.body
    assert result.meta.classification_model == "mock:validation-examples"


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
        "/mock/endpoint/foi/response_status", json=body.model_dump(mode="json")
    )
    assert response.status_code == 200, response.text
    output = ExtractionOutput.model_validate(response.json())
    assert output == example.expected
    assert response.json() == example.expected.model_dump(mode="json")


@pytest.mark.parametrize(
    "path", ["extract_questions/extract", "extract_questions", "response_status"]
)
def test_mock_rejects_invalid_input(client, path):
    """
    Mirror the real endpoints' request validation rather than accept arbitrary JSON.
    """
    assert client.post("/mock/endpoint/foi/" + path, json={}).status_code == 422


def test_mock_rejects_invalid_backend(client):
    """
    Enforce the same extraction backend enumeration even without inference.
    """
    assert (
        client.post(
            "/mock/endpoint/foi/extract_questions/extract?backend=other",
            json={"body": "Hello"},
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
        "extract_questions/extract": {"body": "Hello"},
        "extract_questions": {"body": "Hello"},
        "response_status": validation_examples()
        .responses[0]
        .input.model_dump(mode="json"),
    }
    for path, body in bodies.items():
        url = "/mock/endpoint/foi/" + path
        assert client.post(url, json=body).status_code == 401
        response = client.post(
            url, json=body, headers={"Authorization": "Bearer mock-test-token"}
        )
        assert response.status_code == 200


@pytest.mark.parametrize("country", ["France", "Germany", "Atlantis"])
def test_capital_city_mock_is_fixed_without_inference(client, country):
    response = client.post(
        "/mock/endpoint/misc/capital_city", json={"country": country}
    )
    assert response.status_code == 200
    assert response.json() == {"country": country, "city": "Fake Paris"}


@pytest.mark.parametrize("form", ["object", "list"])
def test_mock_response_status_accepts_minimal_questions(client, monkeypatch, form):
    example = validation_examples().responses[0]
    body = example.input.model_dump(mode="json")
    questions = body["request"]["questions"]
    body["request"] = {"questions": questions} if form == "object" else questions
    monkeypatch.setattr(mocks.random, "choice", lambda pool: example)
    response = client.post("/mock/endpoint/foi/response_status", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == example.expected.model_dump(mode="json")
