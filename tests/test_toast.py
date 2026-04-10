"""
Integration test that exercises the immigration_detection agent endpoint.

Uses FastAPI's TestClient to run the server in-process and a session-scoped
fixture to ensure the deployment is started once for all tests.
"""

from collections.abc import Iterator

import pytest
from llm_management.server import app
from starlette.testclient import TestClient

DEPLOYMENT = "toast_llama"


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        resp = c.post(f"/deployments/{DEPLOYMENT}/ensure")
        resp.raise_for_status()
        yield c


def classify_request(client: TestClient, request_text: str) -> str:
    """
    Post a request to the immigration_detection endpoint and return the classification.
    """
    resp = client.post("/agents/immigration_detection", json={"request": request_text})
    resp.raise_for_status()
    return resp.json()["classification"]


def test_immigration_detection(client: TestClient):
    """
    An immigration-related request should be classified as IMM.
    """
    classification = classify_request(
        client,
        "Dear public authority, I would like an update on my application for leave to remain.",
    )
    assert classification == "IMM", f"Expected 'IMM', got '{classification}'"


def test_foi_detection(client: TestClient):
    """
    A general FOI request should be classified as FOI.
    """
    classification = classify_request(
        client,
        "Please provide all records of expenditure on office supplies by your department in the last financial year.",
    )
    assert classification == "FOI", f"Expected 'FOI', got '{classification}'"
