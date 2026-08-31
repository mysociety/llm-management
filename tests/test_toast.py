"""
Integration test that exercises the immigration_detection agent endpoint.

Uses FastAPI's TestClient to run the server in-process and a session-scoped
fixture to ensure the deployment is started once for all tests.
"""

from collections.abc import Iterator

import pytest
from llm_management.server import app
from starlette.testclient import TestClient

DEPLOYMENTS = (
    "toast_llama",
    # Add further compatible deployment slugs here when required.
)

pytestmark = pytest.mark.external


@pytest.fixture(scope="session", params=DEPLOYMENTS, ids=DEPLOYMENTS)
def deployment_client(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[TestClient, str]]:
    deployment = request.param
    with TestClient(app) as c:
        resp = c.post(f"/deployments/{deployment}/ensure")
        resp.raise_for_status()
        yield c, deployment


def classify_request(client: TestClient, deployment: str, request_text: str) -> str:
    """
    Post a request to the immigration_detection endpoint and return the classification.
    """
    resp = client.post(
        "/agents/immigration_detection",
        params={"deployment": deployment},
        json={"request": request_text},
    )
    resp.raise_for_status()
    return resp.json()["classification"]


def test_immigration_detection(deployment_client: tuple[TestClient, str]):
    """
    An immigration-related request should be classified as IMM.
    """
    client, deployment = deployment_client
    classification = classify_request(
        client,
        deployment,
        "Dear public authority, I would like an update on my application for leave to remain.",
    )
    assert classification == "IMM", f"Expected 'IMM', got '{classification}'"


def test_foi_detection(deployment_client: tuple[TestClient, str]):
    """
    A general FOI request should be classified as FOI.
    """
    client, deployment = deployment_client
    classification = classify_request(
        client,
        deployment,
        "Please provide all records of expenditure on office supplies by your department in the last financial year.",
    )
    assert classification == "FOI", f"Expected 'FOI', got '{classification}'"
