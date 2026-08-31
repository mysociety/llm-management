"""
Integration test that exercises the foi_structure agent endpoint.

Uses FastAPI's TestClient to run the server in-process and a session-scoped
fixture to ensure the deployment is started once for all tests.
"""

from collections.abc import Iterator

import pytest
from llm_management.agents.foi_structure import FOIRequest, normalize_text
from llm_management.server import app
from starlette.testclient import TestClient

DEPLOYMENTS = (
    # "olmo3_7b",
    # "granite4_h_tiny",
    "granite4_1_8b",
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


def extract_structure(
    client: TestClient, deployment: str, request_text: str
) -> FOIRequest:
    """
    Post a request to the foi_structure endpoint and return the parsed result.
    """
    resp = client.post(
        "/agents/foi_structure",
        params={"deployment": deployment},
        json={"request": request_text},
    )
    resp.raise_for_status()
    return FOIRequest.model_validate(resp.json())


def test_eir_foi_mixed(deployment_client: tuple[TestClient, str]):
    test_content = """
    Dear Tanbridge Council,

    I would like the following information:

    1. Latest report on air pollution monitoring in the council.
    2. Minutes of any discussion of this report in council meetings. 
    3. Any internal guidance on publication of council minutes

    Thank you,
    """

    client, deployment = deployment_client
    result = extract_structure(client, deployment, test_content)

    assert len(result.questions) == 3, (
        f"Expected 3 questions, got {len(result.questions)}: {result.questions}"
    )
    assert normalize_text(result.questions[0].text) == normalize_text(
        "Latest report on air pollution monitoring in the council."
    ), f"Question 1 text mismatch: got '{result.questions[0].text}'"
    assert result.questions[0].ir_type == "EIR", (
        f"Question 1 should be EIR (environmental), got '{result.questions[0].ir_type}'"
    )
    assert normalize_text(result.questions[1].text) == normalize_text(
        "Minutes of any discussion of this report in council meetings."
    ), f"Question 2 text mismatch: got '{result.questions[1].text}'"
    assert result.questions[1].ir_type == "FOI", (
        f"Question 2 should be FOI, got '{result.questions[1].ir_type}'"
    )
    assert normalize_text(result.questions[2].text) == normalize_text(
        "Any internal guidance on publication of council minutes"
    ), f"Question 3 text mismatch: got '{result.questions[2].text}'"
    assert result.questions[2].ir_type == "FOI", (
        f"Question 3 should be FOI, got '{result.questions[2].ir_type}'"
    )


def test_all_foi(deployment_client: tuple[TestClient, str]):
    test_content = """
    Dear Rushmoor Borough Council,

    I would like to make an FOI request about the following statistics in the previous annual period. If figures are recorded in calendar year this would be 2017, if financial year 2017-18. If available in both, I would prefer calendar for comparison to central government FOI statistics.

    1. What period do you record FOI statistics in? Financial Year/Calendar Year/Other?
    2. How many FOI requests have you received? (if this figure includes EIR requests, please state)
    3. The number of requests where the information was granted?
    4. The number of requests where the information was entirely withheld (no information provided)?
    5. The number of requests where the information was partially withheld (some, but not all information requested, provided)?
    6. How many requests were completed inside the statutory deadline?
    7. How many requests were appealed to internal review?
    8. How many decisions were upheld at internal review?
    9. How many FOI decisions have been appealed to the ICO?
    10. How many decisions were upheld by the ICO?
    11. How many vexatious requests were received?

    If not all information is available, please treat questions individually.
    """

    client, deployment = deployment_client
    result = extract_structure(client, deployment, test_content)

    # allow a range here, questions can be split up differently and it's *fine* as long as the content is preserved
    assert len(result.questions) > 10 and len(result.questions) < 13, (
        f"Expected 11-12 questions, got {len(result.questions)}: {result.questions}"
    )
    assert normalize_text(result.questions[0].text) == normalize_text(
        "What period do you record FOI statistics in? Financial Year/Calendar Year/Other?"
    ), f"Question 1 text mismatch: got '{result.questions[0].text}'"
    assert result.questions[0].ir_type == "FOI", (
        f"Question 1 should be FOI, got '{result.questions[0].ir_type}'"
    )
    # check all questions are classified as FOI
    for i, question in enumerate(result.questions, 1):
        assert question.ir_type == "FOI", (
            f"Question {i} should be FOI, got '{question.ir_type}': '{question.text}'"
        )
