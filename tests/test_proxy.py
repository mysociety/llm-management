"""
Integration tests that exercise the FastAPI proxy server.

Tests both the /agents/capital_city endpoint and the proxy endpoint
via pydantic-ai, using the olmo3_7b deployment.
"""

from collections.abc import Iterator

import httpx
import pytest
from llm_management.server import app
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from starlette.testclient import TestClient

DEPLOYMENT = "olmo3_7b"
BASE_URL = "http://test"


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        resp = c.post(f"/deployments/{DEPLOYMENT}/ensure")
        resp.raise_for_status()
        yield c


def test_capital_city_agent(client: TestClient):
    """
    Hit the /agents/capital_city endpoint and check the result.
    """
    resp = client.post("/agents/capital_city", json={"country": "France"})
    resp.raise_for_status()
    data = resp.json()
    assert data["city"].lower() == "paris", f"Expected 'Paris', got '{data['city']}'"


def test_proxy_with_pydantic_ai():
    """
    Use pydantic-ai pointed at the proxy's forwarding endpoint to run a
    structured capital-city query, the same way a real client would.

    Uses an ASGI transport so all requests go through the in-process app.
    """

    class CapitalCity(BaseModel):
        country: str
        city: str

    transport = httpx.ASGITransport(app=app)
    async_client = httpx.AsyncClient(transport=transport, base_url=BASE_URL)

    model = OpenAIChatModel(
        "allenai/Olmo-3-7B-Instruct",
        provider=OpenAIProvider(
            api_key="not-needed",
            base_url=f"{BASE_URL}/deployments/{DEPLOYMENT}/v1",
            http_client=async_client,
        ),
    )

    agent = Agent(
        model,
        system_prompt=(
            "Given the name of a country, return the capital city of that country. "
            "e.g. Germany -> Berlin"
        ),
        output_type=CapitalCity,
    )

    result = agent.run_sync("France")
    output = result.output
    assert output.city.lower() == "paris", f"Expected 'Paris', got '{output.city}'"
