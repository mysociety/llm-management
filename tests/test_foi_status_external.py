"""
Paid status-model integration checks; run explicitly against Exoscale.

These semantic examples expose candidate quality and can fail even when transport
and schema checks pass. Startup may provision the configured GPU deployment.
"""

import pytest
from starlette.testclient import TestClient

from llm_management.foi.response_schemas import ExtractionOutput
from llm_management.server import app

pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as connection:
        yield connection


@pytest.mark.parametrize(
    "text,act,event",
    [
        (
            "The requested total number of requests received last year was 42.",
            "supplied",
            None,
        ),
        ("We do not hold the requested records.", "not_held", None),
        (
            "We refuse to disclose the requested records under section 40(2) of the Freedom of Information Act.",
            "withheld",
            None,
        ),
        ("We will send our substantive response next week.", "pending", None),
        ("We explicitly acknowledge receipt of your request.", None, "acknowledgement"),
    ],
    ids=["supplied", "not_held", "withheld", "pending", "acknowledgement"],
)
def test_status_model_semantics(client, text, act, event):
    body = {
        "request": {
            "questions": [
                {
                    "question_id": "q1",
                    "text": "Please provide the total number of requests received last year.",
                }
            ],
            "extraction_status": "questions_found",
            "additional_text": [],
        },
        "sources": [
            {
                "id": "reply",
                "kind": "email",
                "role": "current",
                "sender": "authority",
                "filename": None,
                "availability": "visible",
                "text": text,
            }
        ],
    }
    response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 200, response.text
    output = ExtractionOutput.model_validate(response.json())
    assert output.process_states is None
    if act:
        assert any(outcome.act == act for outcome in output.outcomes), output
    if event:
        assert any(item.kind == event for item in output.events), output
