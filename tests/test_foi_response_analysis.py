from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.foi import response_analysis
from llm_management.foi.response_schemas import ExtractionOutput


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(server.cache, "all_active", lambda: [])
    with TestClient(server.app) as client:
        yield client


@pytest.fixture
def body():
    return {
        "request": {
            "questions": [{"question_id": "q1", "text": "Please provide the report."}],
            "extraction_status": "questions_found",
            "additional_text": [],
        },
        "sources": [
            {
                "id": "email1",
                "kind": "email",
                "role": "current",
                "sender": "authority",
                "filename": None,
                "availability": "visible",
                "text": "Here is the report.",
            }
        ],
    }


@pytest.fixture
def output():
    return {
        "outcomes": [
            {
                "scope": {"kind": "question", "question_id": "q1"},
                "act": "supplied",
                "answer_content_visibility": "visible",
                "content_source_ids": ["email1"],
                "references": [],
            }
        ],
        "events": [],
        "process_states": None,
        "process_references": [],
    }


def test_placeholder_does_not_provision(client, body, monkeypatch):
    ensure = AsyncMock(side_effect=AssertionError("must not provision"))
    monkeypatch.setattr(server, "ensure_running", ensure)
    response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 501
    assert response_analysis.RESPONSE_ANALYSIS_MODEL in response.json()["detail"]
    ensure.assert_not_called()


@pytest.mark.parametrize("upstream", ["compact", "extraction", "full"])
def test_request_output_is_projected_for_model(
    client, body, output, monkeypatch, upstream
):
    expected_request = deepcopy(body["request"])
    if upstream != "compact":
        body["request"].update(
            {
                "ignored_text": [],
                "unit_predictions": [],
                "extraction_backend": "cpu",
                "extraction_model": "modernbert",
                "extraction_revision": None,
                "promoted_continuation_index": None,
            }
        )
    if upstream == "full":
        body["request"].update(
            {
                "request_text": "Original request",
                "classification": None,
                "classification_model": None,
            }
        )
    predict = AsyncMock(return_value=ExtractionOutput.model_validate(output))
    monkeypatch.setattr(response_analysis, "predict", predict)
    response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == output
    observed = predict.call_args.args[0].value
    assert observed.request.model_dump() == expected_request
    assert observed.request_text == ("Original request" if upstream == "full" else None)
    assert (
        predict.call_args.kwargs["model"] == response_analysis.RESPONSE_ANALYSIS_MODEL
    )


@pytest.mark.parametrize(
    "invalid",
    [
        "duplicate_question",
        "status",
        "duplicate_source",
        "invisible_text",
        "missing_visible_text",
        "empty_sources",
        "extra_field",
        "strict_type",
    ],
)
def test_invalid_input_precedes_inference(client, body, monkeypatch, invalid):
    if invalid == "duplicate_question":
        body["request"]["questions"] *= 2
    elif invalid == "status":
        body["request"]["extraction_status"] = "no_questions_found"
    elif invalid == "duplicate_source":
        body["sources"] *= 2
    elif invalid == "invisible_text":
        body["sources"][0]["availability"] = "unavailable"
    elif invalid == "missing_visible_text":
        body["sources"][0]["text"] = None
    elif invalid == "empty_sources":
        body["sources"] = []
    elif invalid == "extra_field":
        body["request"]["unexpected"] = True
    elif invalid == "strict_type":
        body["sources"][0]["id"] = 1
    predict = AsyncMock()
    monkeypatch.setattr(response_analysis, "predict", predict)
    assert client.post("/agents/foi_response_analysis", json=body).status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize(
    "invalid",
    [
        "question",
        "source",
        "visibility",
        "non_supply",
        "duplicate_content_source",
        "event_question",
        "state_question",
        "reference_question",
    ],
)
def test_model_references_and_invariants_are_validated(
    client, body, output, monkeypatch, invalid
):
    outcome = output["outcomes"][0]
    if invalid == "question":
        outcome["scope"]["question_id"] = "q2"
    elif invalid == "source":
        outcome["content_source_ids"] = ["missing"]
    elif invalid == "visibility":
        outcome["answer_content_visibility"] = "unobservable"
    elif invalid == "non_supply":
        outcome["act"] = "withheld"
    elif invalid == "duplicate_content_source":
        outcome["content_source_ids"] *= 2
    elif invalid == "event_question":
        output["events"] = [
            {
                "scope": {
                    "kind": "narrower",
                    "question_id": "q2",
                    "description": "Part",
                },
                "kind": "acknowledgement",
                "references": [],
            }
        ]
    elif invalid == "state_question":
        output["process_states"] = [
            {
                "scope": {"kind": "question", "question_id": "q2"},
                "completion": "open",
                "next_actor": "authority",
            }
        ]
    elif invalid == "reference_question":
        output["process_references"] = [
            {"scope": {"kind": "question", "question_id": "q2"}, "references": []}
        ]
    monkeypatch.setattr(response_analysis, "predict", AsyncMock(return_value=output))
    assert client.post("/agents/foi_response_analysis", json=body).status_code == 502


@pytest.mark.parametrize(
    "status,questions", [("no_questions_found", []), ("uncertain", [])]
)
def test_no_questions_still_allows_whole_request_analysis(
    client, body, output, monkeypatch, status, questions
):
    body["request"].update(extraction_status=status, questions=questions)
    output["outcomes"][0]["scope"] = {"kind": "whole_request"}
    monkeypatch.setattr(response_analysis, "predict", AsyncMock(return_value=output))
    assert client.post("/agents/foi_response_analysis", json=body).status_code == 200


def test_endpoint_exposes_output_contract():
    operation = server.app.openapi()["paths"]["/agents/foi_response_analysis"]["post"]
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ExtractionOutput"
    }


def test_conflicting_original_request_text_is_rejected(client, body, monkeypatch):
    body["request"].update(
        {
            "ignored_text": [],
            "unit_predictions": [],
            "extraction_backend": "cpu",
            "extraction_model": "modernbert",
            "extraction_revision": None,
            "request_text": "Original",
            "classification": None,
            "classification_model": None,
        }
    )
    body["request_text"] = "Different"
    predict = AsyncMock()
    monkeypatch.setattr(response_analysis, "predict", predict)
    assert client.post("/agents/foi_response_analysis", json=body).status_code == 422
    predict.assert_not_called()


def test_unavailable_attachment_and_event_variants(client, body, output, monkeypatch):
    body["sources"].append(
        {
            "id": "attachment1",
            "kind": "attachment",
            "role": "current",
            "sender": None,
            "filename": "report.pdf",
            "availability": "unavailable",
            "text": None,
        }
    )
    output["outcomes"][0].update(
        answer_content_visibility="unobservable", content_source_ids=["attachment1"]
    )
    scope = {"kind": "whole_request"}
    output["events"] = [
        {
            "scope": scope,
            "kind": "acknowledgement",
            "automatic": None,
            "references": [],
        },
        {
            "scope": scope,
            "kind": "internal_review_decision",
            "review_result": "revised",
            "references": [],
        },
        {"scope": scope, "kind": "review_rights", "references": []},
    ]
    output["process_states"] = []
    monkeypatch.setattr(response_analysis, "predict", AsyncMock(return_value=output))
    response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == output
