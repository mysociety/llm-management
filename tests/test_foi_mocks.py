"""
Verify mock contracts, gold payload preservation and absence of inference work.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import httpx2
import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.api import mocks
from llm_management.foi.email_finder.schemas import FOIResult, PublicAuthority
from llm_management.foi.email_finder.service import FOIFinder
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
    for name in (
        "classifier",
        "head",
        "sanitizer",
        "tokenizer",
        "response_tokenizer",
        "category_matcher",
    ):
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
    assert [q.model_dump(exclude={"text", "category"}) for q in result.questions] == [
        q.model_dump() for q in example.classification.questions
    ]
    assert all(q.category is None for q in result.questions)
    assert result.request_text == example.input.body
    assert result.meta.classification_model == "mock:validation-examples"
    assert result.meta.category_classification.status == "not_configured"


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
    "path",
    ["extract_questions/extract", "extract_questions", "response_status", "find_email"],
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
    Require configured bearer authentication for the FOI mock routes.
    """
    monkeypatch.setattr(
        server.settings, "auth_tokens", {"integration": "mock-test-token"}
    )
    bodies = {
        "extract_questions/extract": {"body": "Hello"},
        "extract_questions": {"body": "Hello"},
        "find_email": {"name": "Example Council"},
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


@pytest.mark.parametrize(
    "previous_email,expected_match",
    [(None, None), ("FOI@COUNCIL.EXAMPLE", True), ("old@example.org", False)],
)
def test_mock_email_finder_returns_fictional_result_without_external_calls(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    previous_email: str | None,
    expected_match: bool | None,
) -> None:
    finder_call = AsyncMock(side_effect=AssertionError("Mock called the finder"))
    network_call = AsyncMock(
        side_effect=AssertionError("Mock made an external request")
    )
    monkeypatch.setattr(FOIFinder, "find", finder_call)
    monkeypatch.setattr(httpx2.AsyncClient, "send", network_call)
    monkeypatch.setattr(
        server.app.state.application.foi_finder.catalog, "foi_email_finder", None
    )
    authority = PublicAuthority(
        name="Example Council",
        context="Public authority",
        previous_email=previous_email,
    )

    response = client.post(
        "/mock/endpoint/foi/find_email", json=authority.model_dump(mode="json")
    )

    assert response.status_code == 200, response.text
    result = FOIResult.model_validate(response.json())
    assert result.authority == authority
    assert response.json()["matches_previous_email"] is expected_match
    assert result.finding.status == "found"
    assert result.finding.email == "foi@council.example"
    assert result.finding.email_type == "foi_eir"
    assert result.finding.evidence is not None
    assert str(result.finding.evidence.source_url) == "https://council.example/foi"
    assert "fictional" in result.finding.explanation
    assert result.checked_at.utcoffset() is not None
    assert result.searches == result.pages == []
    assert result.usage.model.provider == "mock"
    assert result.usage.model.model == "mock:foi-email-finder"
    assert result.usage.model.requests == 0
    assert result.usage.model.input_tokens == result.usage.model.output_tokens == 0
    assert (
        result.usage.model.cached_input_tokens
        == result.usage.model.cache_write_tokens
        == 0
    )
    assert result.usage.search.provider is None
    assert result.usage.search.requests == result.usage.search.credits == 0
    finder_call.assert_not_called()
    network_call.assert_not_called()
