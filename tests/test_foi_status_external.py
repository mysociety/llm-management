"""
Paid status-model validation using the copied S1–S5 dataset pairs.

Each input retains its own gold request context. Candidate semantic failures are
reported separately from transport/schema failures; no model output is repaired.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from llm_management.foi.examples import validation_examples
from llm_management.foi.response_schemas import ExtractionOutput
from llm_management.server import app
from tests.foi_validation import assert_response_matches, post_example

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def client():
    """
    Own the API lifecycle and scale touched deployments down after the batch.
    """
    with TestClient(app) as connection:
        yield connection


@pytest.mark.parametrize(
    "example", validation_examples().responses, ids=lambda case: case.id
)
def test_status_model_validation(client, example, record_property):
    """
    Require the complete gold output, including references, scopes and visibility.
    """
    response = post_example(
        client,
        "/endpoint/foi/response_status",
        example.id,
        example.input,
        record_property,
    )
    output = ExtractionOutput.model_validate(response.json())
    output.validate_against(example.input.model_input())
    assert_response_matches(output, example.expected)
