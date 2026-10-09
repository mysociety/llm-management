import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_management import inference
from llm_management.cache import RunningDeployment
from llm_management.errors import ClassifierUnavailable
from llm_management.local_resources import ResourceRegistry
from llm_management.foi import backends, question_extractor, pipeline, response_analysis
from llm_management.foi.backends import ClassificationRows, DeploymentAccess
from llm_management.settings import settings
from llm_management.foi.response_schemas import (
    ExtractionInput,
    ExtractionOutput,
    ResponseAnalysisInput,
)
from llm_management.sanitization import (
    PresidioSanitizer,
    Sanitized,
    presidio,
    require_sanitized,
)


def observed():
    return ExtractionInput(
        request={
            "questions": [{"question_id": "q1", "text": "Records about Alice Smith"}],
            "extraction_status": "questions_found",
            "additional_text": ["Email alice@example.org"],
        },
        request_text="Alice Smith: alice@example.org",
        sources=[
            {
                "id": "alice@example.org",
                "kind": "email",
                "role": "current",
                "sender": "authority",
                "filename": "Alice Smith.pdf",
                "availability": "visible",
                "text": "Dear Alice Smith, alice@example.org",
            }
        ],
    )


def test_response_fields_share_mapping_preserve_metadata_and_snapshot():
    original = observed()
    sanitized = presidio.sanitize_extraction(original)
    clean = require_sanitized(sanitized)
    assert clean.request_text == "<PERSON_1>: <EMAIL_ADDRESS_1>"
    assert clean.request.questions[0].text == "Records about <PERSON_1>"
    assert clean.request.additional_text == ["Email <EMAIL_ADDRESS_1>"]
    assert clean.sources[0].filename == "<PERSON_1>.pdf"
    assert clean.sources[0].id == "alice@example.org"
    assert clean.sources[0].text == "Dear <PERSON_1>, <EMAIL_ADDRESS_1>"
    clean.sources[0].text = "New raw text"
    assert sanitized.value.sources[0].text != "New raw text"
    assert "Alice Smith" in original.sources[0].text
    with pytest.raises(FrozenInstanceError):
        sanitized.policy_version = "old"
    with pytest.raises(TypeError):
        Sanitized(original)


def test_existing_placeholders_and_request_scoped_mapping():
    clean = presidio.sanitize_texts(
        [
            "<PERSON_1> Alice Smith",
            "Alice Smith",
            "alice@example.org",
            "alice@example.org",
        ]
    )
    assert clean.value == [
        "<PERSON_1> <PERSON_2>",
        "<PERSON_2>",
        "<EMAIL_ADDRESS_1>",
        "<EMAIL_ADDRESS_1>",
    ]
    assert presidio.sanitize_texts(clean.value).value == clean.value
    assert presidio.sanitize_text("Alice Smith").value == "<PERSON_1>"


def test_request_model_inputs_clean_results_original(monkeypatch):
    classifier = AsyncMock(
        return_value=ClassificationRows([[0, 0, 1, 0]], "test", "pin")
    )
    monkeypatch.setattr(backends, "classify_question_units", classifier)
    deployments = DeploymentAccess(Mock(), AsyncMock(), Mock())
    original = "Please provide records about Alice Smith at alice@example.org."
    result = asyncio.run(pipeline.extract_questions(original, deployments=deployments))
    submitted = require_sanitized(classifier.call_args.args[0])
    assert all(
        "Alice Smith" not in text and "alice@example.org" not in text
        for text in submitted
    )
    assert result.questions[0].text == original


def test_topic_payload_is_sanitized_before_provisioning(monkeypatch):
    monkeypatch.setattr(
        backends,
        "classify_question_units",
        AsyncMock(return_value=ClassificationRows([[0, 0, 1, 0]], "test", "pin")),
    )
    seen = []

    def prepare(text, questions):
        seen.extend([text, questions[0].text])
        return presidio.sanitize_payload(
            {"messages": [{"role": "user", "content": text}]}
        )

    monkeypatch.setattr(question_extractor, "prepare_topic_request", prepare)
    classify = AsyncMock(return_value=None)
    monkeypatch.setattr(question_extractor, "classify_topics", classify)
    cfg = SimpleNamespace(model=settings.foi_topic_model)
    deployments = DeploymentAccess(
        lambda _: cfg,
        AsyncMock(
            return_value=RunningDeployment(
                config=cfg,
                state=SimpleNamespace(deployment_url="unused", api_key="unused"),
            )
        ),
        Mock(),
    )
    result = asyncio.run(
        pipeline.process_information_request(
            "Please provide Alice Smith's records.", deployments=deployments
        )
    )
    assert all("Alice Smith" not in value for value in seen)
    assert "Alice Smith" not in str(
        require_sanitized(classify.call_args.kwargs["payload"])
    )
    assert "Alice Smith" in result.request_text


def test_sanitizer_failure_prevents_request_and_response_inference(monkeypatch):
    monkeypatch.setattr(
        presidio,
        "sanitize_strings",
        Mock(side_effect=ClassifierUnavailable("unavailable")),
    )
    classify = AsyncMock()
    monkeypatch.setattr(backends, "classify_question_units", classify)
    deployments = DeploymentAccess(Mock(), AsyncMock(), Mock())
    with pytest.raises(pipeline.PipelineUnavailable):
        asyncio.run(
            pipeline.extract_questions(
                "Please provide records.", deployments=deployments
            )
        )
    classify.assert_not_called()
    deployments.ensure_running.assert_not_called()
    predict = AsyncMock()
    monkeypatch.setattr(response_analysis, "predict", predict)
    with pytest.raises(pipeline.PipelineUnavailable):
        asyncio.run(
            response_analysis.analyze_response(
                ResponseAnalysisInput.model_validate(observed().model_dump())
            )
        )
    predict.assert_not_called()


def test_external_adapters_reject_raw_input_before_http(monkeypatch):
    monkeypatch.setattr(
        inference.httpx,
        "AsyncClient",
        Mock(side_effect=AssertionError("HTTP forbidden")),
    )
    with pytest.raises(TypeError, match="sanitization policy"):
        asyncio.run(
            inference.classify_remote(
                texts=["raw"],
                model="test",
                deployment_url="unused",
                api_key="unused",
                embedding_head=lambda x: x,
            )
        )
    with pytest.raises(TypeError, match="sanitization policy"):
        asyncio.run(
            question_extractor.classify_topics(
                payload={},
                model="test",
                deployment_url="unused",
                api_key="unused",
                question_ids=[],
            )
        )
    with pytest.raises(TypeError, match="sanitization policy"):
        asyncio.run(response_analysis.predict(observed(), model="test"))


def test_response_predict_receives_sanitized_input(monkeypatch):
    output = ExtractionOutput(outcomes=[], events=[], process_references=[])
    predict = AsyncMock(return_value=output)
    monkeypatch.setattr(response_analysis, "predict", predict)
    body = ResponseAnalysisInput.model_validate(observed().model_dump())
    assert asyncio.run(response_analysis.analyze_response(body)) == output
    assert isinstance(predict.call_args.args[0], Sanitized)
    assert "Alice Smith" not in predict.call_args.args[0].value.sources[0].text


def test_real_presidio_cpu_recognizers(monkeypatch):
    pytest.importorskip("en_core_web_sm")
    monkeypatch.setattr(
        "socket.socket.connect",
        Mock(side_effect=AssertionError("Presidio must stay local")),
    )
    sanitizer = PresidioSanitizer(registry=ResourceRegistry())
    try:
        clean = sanitizer.sanitize_text(
            "Email alice@example.org or call 07700 900123. National Insurance number AB 12 34 56 C. Address: 12 High Street, SW1A 1AA. London council records for 2024."
        ).value
        assert "alice@example.org" not in clean
        assert "07700 900123" not in clean
        assert "12 High Street" not in clean
        assert "AB 12 34 56 C" not in clean
        assert "London" in clean and "2024" in clean
    finally:
        sanitizer.resource.release()
