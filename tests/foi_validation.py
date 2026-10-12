"""
Comparison and reporting helpers for the copied live FOI validation examples.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from time import monotonic

from httpx import Response
from pydantic import BaseModel
from starlette.testclient import TestClient

from llm_management.foi.examples import (
    AcceptedClassification,
    AcceptedExtraction,
    ExtractionExpectation,
)
from llm_management.foi.response_schemas import ExtractionOutput
from llm_management.foi.schemas import TopicOutput


def post_example(
    client: TestClient,
    endpoint: str,
    case_id: str,
    body: BaseModel,
    record_property: Callable[[str, object], None],
    *,
    backend: str | None = None,
) -> Response:
    """
    Record transport, latency and full output before any semantic assertions.
    """
    start = monotonic()
    response = client.post(
        endpoint,
        params={"backend": backend} if backend else None,
        json=body.model_dump(mode="json"),
    )
    record_property("case_id", case_id)
    record_property("endpoint", endpoint)
    record_property("backend", backend or "exoscale")
    record_property("latency_seconds", round(monotonic() - start, 3))
    record_property("http_status", response.status_code)
    record_property("actual_output", response.text)
    assert response.status_code == 200, (
        f"Transport failure ({case_id}): {response.text}"
    )
    return response


def assert_extraction_matches(
    actual: ExtractionExpectation,
    expected: ExtractionExpectation,
    *,
    accepted: list[AcceptedExtraction] | None = None,
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    """
    Require gold or an explicitly reviewed reconstruction, recording gold fidelity.
    """
    alternative = next((item for item in accepted or [] if actual == item.output), None)
    if record_property:
        record_property("exact_extraction_match", actual == expected)
        record_property(
            "extraction_match",
            "gold"
            if actual == expected
            else "reviewed"
            if alternative
            else "unrecognized",
        )
    if actual == expected:
        return
    if alternative:
        if record_property:
            record_property("extraction_review_reason", alternative.reason)
        return
    whitespace_only = (
        actual.extraction_status == expected.extraction_status
        and [q.question_id for q in actual.questions]
        == [q.question_id for q in expected.questions]
        and [" ".join(q.text.split()) for q in actual.questions]
        == [" ".join(q.text.split()) for q in expected.questions]
        and [" ".join(text.split()) for text in actual.additional_text]
        == [" ".join(text.split()) for text in expected.additional_text]
    )
    category = (
        "Whitespace-only text difference"
        if whitespace_only
        else "Semantic extraction mismatch"
    )
    assert actual == expected, f"{category}\nExpected: {expected}\nActual: {actual}"


def assert_response_matches(
    actual: ExtractionOutput, expected: ExtractionOutput
) -> None:
    """
    Match complete items irrespective of outcome/event order, retaining duplicates.
    """
    for field in ("outcomes", "events"):
        actual_items = Counter(
            json.dumps(item.model_dump(mode="json"), sort_keys=True)
            for item in getattr(actual, field)
        )
        expected_items = Counter(
            json.dumps(item.model_dump(mode="json"), sort_keys=True)
            for item in getattr(expected, field)
        )
        assert actual_items == expected_items, f"Semantic {field} mismatch: {actual}"
    assert actual.process_states == expected.process_states
    assert actual.process_references == expected.process_references


def assert_topics_match(
    actual: TopicOutput,
    expected: TopicOutput,
    record_property: Callable[[str, object], None],
    *,
    accepted: list[AcceptedClassification] | None = None,
) -> None:
    """
    Keep IDs/regimes strict and accept only gold or explicitly reviewed topic output.
    """
    assert [q.question_id for q in actual.questions] == [
        q.question_id for q in expected.questions
    ]
    assert [q.regime for q in actual.questions] == [
        q.regime for q in expected.questions
    ]
    record_property("exact_topic_match", actual == expected)
    record_property("expected_classification", expected.model_dump_json())
    alternative = next((item for item in accepted or [] if actual == item.output), None)
    record_property(
        "topic_match",
        "gold" if actual == expected else "reviewed" if alternative else "unrecognized",
    )
    if alternative:
        record_property("topic_review_reason", alternative.reason)
        return
    assert actual == expected, (
        "Topic wording mismatch; review for equivalent wording versus changed subject matter."
        f"\nExpected: {expected}\nActual: {actual}"
    )
