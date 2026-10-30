"""Paid smoke tests for the real ModernBERT CPU and Exoscale backends."""

import pytest
from starlette.testclient import TestClient

from llm_management.bootstrap import assemble_runtime
from llm_management.foi.examples import ExtractionExpectation, validation_examples
from llm_management.foi.schemas import QuestionSliceResult
from llm_management.server import create_app
from tests.foi_validation import assert_extraction_matches, post_example

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def client():
    # App shutdown scales deployments touched by this client back to zero.
    with TestClient(create_app(runtime=assemble_runtime())) as test_client:
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
            "/endpoint/foi/extract_questions/extract",
            params={"backend": backend},
            json={"body": request_text},
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
        "/endpoint/foi/extract_questions/extract",
        json={
            "body": "Dear Council,\n\nPlease provide the total number of FOI requests "
            "received last year.\n\nYours faithfully,\nA resident"
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["extraction_status"] == "questions_found"
    assert result["promoted_continuation_index"] == 1
    assert result["unit_predictions"][1]["label"] == "QUESTION_CONTINUATION"
    assert "orphan_continuation_indices" not in result
    assert [q["text"] for q in result["questions"]] == [
        "Please provide the total number of FOI requests received last year."
    ]
    assert "unit_indices" not in result["questions"][0]


@pytest.mark.parametrize("backend", ["cpu", "exoscale"])
@pytest.mark.parametrize(
    "example", validation_examples().requests, ids=lambda case: case.id
)
def test_dataset_extraction(client, backend, example, record_property):
    """
    Compare R1–R5 gold or reviewed reconstructions on both extraction backends.
    """
    response = post_example(
        client,
        "/endpoint/foi/extract_questions/extract",
        example.id,
        example.input,
        record_property,
        backend=backend,
    )
    result = QuestionSliceResult.model_validate(response.json())
    assert result.extraction_backend == backend
    assert_extraction_matches(
        ExtractionExpectation.from_result(result),
        example.expected,
        accepted=example.accepted_extractions,
        record_property=record_property,
    )
