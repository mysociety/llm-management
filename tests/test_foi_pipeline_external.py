"""
Paid extraction and topic-classification validation using the R1–R5 gold examples.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from llm_management.bootstrap import assemble_runtime
from llm_management.deployments import get_catalog
from llm_management.foi.examples import ExtractionExpectation, validation_examples
from llm_management.foi.schemas import ExtractQuestionsResponse, TopicOutput
from llm_management.server import create_app
from tests.foi_validation import (
    assert_extraction_matches,
    assert_topics_match,
    post_example,
)

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def client():
    """
    Release model resources and scale touched deployments down after the batch.
    """
    with TestClient(create_app(runtime=assemble_runtime())) as connection:
        yield connection


@pytest.mark.parametrize("backend", ["cpu", "exoscale"])
@pytest.mark.parametrize(
    "example", validation_examples().requests, ids=lambda case: case.id
)
def test_full_pipeline_validation(client, backend, example, record_property):
    """
    Check reviewed extraction and topics while keeping IDs and regimes strict.
    """
    response = post_example(
        client,
        "/endpoint/foi/extract_questions",
        example.id,
        example.input,
        record_property,
        backend=backend,
    )
    result = ExtractQuestionsResponse.model_validate(response.json())
    assert_extraction_matches(
        ExtractionExpectation.from_result(result),
        example.expected,
        accepted=example.accepted_extractions,
        record_property=record_property,
    )
    assert (
        result.meta.classification_model
        == get_catalog().get(get_catalog().require_foi().topic_deployment).model
    )
    assert_topics_match(
        TopicOutput.model_validate(
            {
                "questions": [q.model_dump(exclude={"text"}) for q in result.questions],
            }
        ),
        example.classification,
        record_property,
        accepted=example.accepted_classifications,
    )


@pytest.mark.parametrize("backend", ["cpu", "exoscale"])
def test_no_questions_skips_classification(client, backend):
    """
    Retain the empty-question behaviour absent from the supplied gold examples.
    """
    response = client.post(
        "/endpoint/foi/extract_questions",
        params={"backend": backend},
        json={
            "body": "Dear Council,\n\nThank you for your response.\n\nKind regards,\nA resident"
        },
    )
    assert response.status_code == 200, response.text
    result = ExtractQuestionsResponse.model_validate(response.json())
    assert result.extraction_status == "no_questions_found"
    assert result.questions == []
    assert result.meta.classification_model is None
