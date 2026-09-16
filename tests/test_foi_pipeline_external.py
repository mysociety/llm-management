"""Paid end-to-end smoke tests for the fine-tuned extraction/classification flow."""

import pytest
from starlette.testclient import TestClient
from llm_management.server import app
from llm_management.foi.model_spec import GRANITE_MERGED
from llm_management.foi.schemas import InformationRequestResult

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("backend", ["cpu", "exoscale"])
@pytest.mark.parametrize(
    "text,regime",
    [
        (
            "Dear Council,\n\nPlease provide the total number of FOI requests received last year.\n\nYours faithfully,\nA resident",
            "FOI",
        ),
        ("Please provide the latest air pollution monitoring report.", "EIR"),
        (
            "Please provide a copy of all personal data you hold about me under my right of subject access.",
            "SAR",
        ),
        (
            "Dear Council,\n\nThank you for your response.\n\nKind regards,\nA resident",
            None,
        ),
    ],
    ids=["promoted_foi", "eir", "sar", "no_questions"],
)
def test_full_pipeline(client, backend, text, regime):
    response = client.post(
        "/agents/foi_structure", params={"backend": backend}, json={"request": text}
    )
    assert response.status_code == 200, response.text
    result = InformationRequestResult.model_validate(response.json())
    if regime is None:
        assert result.extraction_status == "no_questions_found"
        assert result.classification is None
        return
    assert result.extraction_status == "questions_found"
    assert result.classification_model == GRANITE_MERGED
    assert result.classification is not None
    assert [q.question_id for q in result.classification.questions] == [
        q.question_id for q in result.questions
    ]
    assert all(q.regime == regime for q in result.classification.questions)
    if regime == "FOI":
        assert result.promoted_continuation_index == 1
        assert (
            result.questions[0].text
            == "Please provide the total number of FOI requests received last year."
        )


def test_multiple_questions(client):
    text = "Dear Council,\n\nPlease provide the following information:\n\n1. The latest report on air pollution monitoring.\n2. Minutes of meetings discussing that report.\n3. Internal guidance on publication of council minutes.\n\nThank you."
    response = client.post("/agents/foi_structure", json={"request": text})
    assert response.status_code == 200, response.text
    result = InformationRequestResult.model_validate(response.json())
    assert len(result.questions) == 3
    assert [q.question_id for q in result.classification.questions] == [
        "q1",
        "q2",
        "q3",
    ]
    assert result.classification.questions[0].regime == "EIR"
