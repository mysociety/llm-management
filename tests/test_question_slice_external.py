"""Paid smoke tests for the real ModernBERT CPU and Exoscale backends."""

import pytest
from starlette.testclient import TestClient

from llm_management.server import app

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def client():
    # App shutdown scales deployments touched by this client back to zero.
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    ("request_text", "expected_status"),
    [
        (
            "Dear Council,\n\nThank you for your response.\n\nKind regards,\nA resident",
            "no_questions_found",
        ),
        (
            "Dear Council,\n\nPlease provide the total number of FOI requests "
            "received last year.\n\nYours faithfully,\nA resident",
            # The raw orphan continuation is recovered by the local heuristic.
            "questions_found",
        ),
        (
            "Dear Council,\n\nPlease provide the following information:\n\n"
            "1. The latest report on air pollution monitoring.\n"
            "2. Minutes of meetings discussing that report.\n"
            "3. Internal guidance on publication of council minutes.\n\n"
            "Please supply electronic copies.\n\nThank you.",
            "questions_found",
        ),
    ],
    ids=["no_questions", "single_question", "numbered_questions"],
)
def test_cpu_exoscale_parity(client, request_text, expected_status):
    results = []
    for backend in ("cpu", "exoscale"):
        response = client.post(
            "/agents/foi_structure/extract",
            params={"backend": backend},
            json={"request": request_text},
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["extraction_backend"] == backend
        assert result["extraction_status"] == expected_status
        results.append(result)

    cpu, gpu = results
    for field in (
        "extraction_model",
        "questions",
        "orphan_continuation_indices",
        "promoted_continuation_index",
        "additional_text",
        "ignored_text",
    ):
        assert cpu[field] == gpu[field], field
    assert [p["label"] for p in cpu["unit_predictions"]] == [
        p["label"] for p in gpu["unit_predictions"]
    ]
    if expected_status == "no_questions_found":
        assert cpu["questions"] == []
    else:
        assert cpu["questions"]


def test_single_question_recovers_without_escalation(client):
    response = client.post(
        "/agents/foi_structure/extract",
        json={
            "request": "Dear Council,\n\nPlease provide the total number of FOI requests "
            "received last year.\n\nYours faithfully,\nA resident"
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["extraction_status"] == "questions_found"
    assert result["promoted_continuation_index"] == 1
    assert result["unit_predictions"][1]["label"] == "QUESTION_CONTINUATION"
    assert result["orphan_continuation_indices"] == [1]
    assert [q["text"] for q in result["questions"]] == [
        "Please provide the total number of FOI requests received last year."
    ]
    assert result["questions"][0]["unit_indices"] == [1]
