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
    monkeypatch.setattr(server.app.state.service, "active_deployments", lambda: [])
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


def test_unconfigured_response_does_not_provision(client, body, monkeypatch):
    ensure = AsyncMock(side_effect=AssertionError("must not provision"))
    monkeypatch.setattr(server.app.state.service, "_prepare_locked", ensure)
    foi = server.app.state.application.foi.catalog.require_foi()
    monkeypatch.setattr(foi, "response_deployment", None)
    response = client.post("/endpoint/foi/response_status", json=body)
    assert response.status_code == 503
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
    response = client.post("/endpoint/foi/response_status", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == output
    observed = predict.call_args.args[0].value
    assert observed.request.model_dump() == expected_request
    assert observed.request_text == ("Original request" if upstream == "full" else None)
    assert (
        predict.call_args.kwargs["model"]
        == server.app.state.service.catalog.get("foi_status").model
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
    assert client.post("/endpoint/foi/response_status", json=body).status_code == 422
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
    assert client.post("/endpoint/foi/response_status", json=body).status_code == 502


@pytest.mark.parametrize(
    "status,questions", [("no_questions_found", []), ("uncertain", [])]
)
def test_no_questions_still_allows_whole_request_analysis(
    client, body, output, monkeypatch, status, questions
):
    body["request"].update(extraction_status=status, questions=questions)
    output["outcomes"][0]["scope"] = {"kind": "whole_request"}
    monkeypatch.setattr(response_analysis, "predict", AsyncMock(return_value=output))
    assert client.post("/endpoint/foi/response_status", json=body).status_code == 200


def test_endpoint_exposes_output_contract():
    operation = server.app.openapi()["paths"]["/endpoint/foi/response_status"]["post"]
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
    assert client.post("/endpoint/foi/response_status", json=body).status_code == 422
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
    response = client.post("/endpoint/foi/response_status", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == output


@pytest.mark.parametrize("form", ["object", "list"])
@pytest.mark.parametrize("empty", [False, True])
def test_minimal_questions_without_extraction_diagnostics(
    client, body, output, monkeypatch, form, empty
):
    questions = [] if empty else body["request"]["questions"]
    body["request"] = {"questions": questions} if form == "object" else questions
    if empty:
        output["outcomes"][0]["scope"] = {"kind": "whole_request"}
    predict = AsyncMock(return_value=output)
    monkeypatch.setattr(response_analysis, "predict", predict)
    response = client.post("/endpoint/foi/response_status", json=body)
    assert response.status_code == 200, response.text
    assert predict.call_args.args[0].value.request.model_dump() == {
        "questions": questions,
        "extraction_status": "no_questions_found" if empty else "questions_found",
        "additional_text": [],
    }


@pytest.mark.parametrize("form", ["object", "list"])
@pytest.mark.parametrize("invalid", ["duplicate", "bad_id", "missing_text"])
def test_minimal_questions_reject_invalid_references(
    client, body, monkeypatch, form, invalid
):
    questions = body["request"]["questions"]
    if invalid == "duplicate":
        questions *= 2
    elif invalid == "bad_id":
        questions[0]["question_id"] = "bad"
    else:
        questions[0].pop("text")
    body["request"] = {"questions": questions} if form == "object" else questions
    predict = AsyncMock()
    monkeypatch.setattr(response_analysis, "predict", predict)
    assert client.post("/endpoint/foi/response_status", json=body).status_code == 422
    predict.assert_not_called()


@pytest.mark.parametrize("form", ["full", "object", "list"])
def test_enriched_questions_chain_to_response_status(
    client, body, output, monkeypatch, form
):
    from llm_management.foi.examples import validation_examples
    from llm_management.foi.schemas import ExtractQuestionsResponse

    result = ExtractQuestionsResponse.from_result(
        validation_examples().requests[0].information_result("cpu")
    ).model_dump(mode="json")
    original_questions = [
        {"question_id": q["question_id"], "text": q["text"]}
        for q in result["questions"]
    ]
    body["request"] = (
        result
        if form == "full"
        else {"questions": result["questions"]}
        if form == "object"
        else result["questions"]
    )
    predict = AsyncMock(return_value=output)
    monkeypatch.setattr(response_analysis, "predict", predict)
    response = client.post("/endpoint/foi/response_status", json=body)
    assert response.status_code == 200, response.text
    observed = predict.call_args.args[0].value
    assert observed.request.model_dump()["questions"] == original_questions
    assert observed.request_text == (result["request_text"] if form == "full" else None)

    if form == "full":
        body["request_text"] = "Conflicting request"
        assert (
            client.post("/endpoint/foi/response_status", json=body).status_code == 422
        )


@pytest.mark.parametrize("explicit", [False, True])
def test_response_dependencies_resolve_once(body, output, monkeypatch, explicit):
    import asyncio
    from unittest.mock import Mock

    from llm_management import bootstrap
    from llm_management.foi.response_schemas import ResponseAnalysisInput

    application = bootstrap.default_application()
    default = Mock(
        return_value=application,
        side_effect=AssertionError("Explicit dependencies must stay isolated")
        if explicit
        else None,
    )
    monkeypatch.setattr(bootstrap, "default_application", default)
    predict = AsyncMock(return_value=ExtractionOutput.model_validate(output))
    monkeypatch.setattr(response_analysis, "predict", predict)
    kwargs = (
        {"resources": application.foi, "deployments": application.deployments}
        if explicit
        else {}
    )
    asyncio.run(
        response_analysis.analyze_response(
            ResponseAnalysisInput.model_validate(body), **kwargs
        )
    )
    assert default.call_count == (0 if explicit else 1)
    assert predict.call_args.kwargs["resources"] is application.foi
    assert predict.call_args.kwargs["deployments"] is application.deployments


@pytest.mark.parametrize("dependency", ["resources", "deployments"])
def test_response_rejects_partial_application_dependencies(
    body, monkeypatch, dependency
):
    import asyncio
    from unittest.mock import Mock

    from llm_management import bootstrap
    from llm_management.foi.response_schemas import ResponseAnalysisInput

    default = Mock(side_effect=AssertionError("Must not resolve a second application"))
    monkeypatch.setattr(bootstrap, "default_application", default)
    predict = AsyncMock()
    monkeypatch.setattr(response_analysis, "predict", predict)
    with pytest.raises(ValueError, match="Supply resources and deployments together"):
        asyncio.run(
            response_analysis.analyze_response(
                ResponseAnalysisInput.model_validate(body), **{dependency: Mock()}
            )
        )
    default.assert_not_called()
    predict.assert_not_called()
