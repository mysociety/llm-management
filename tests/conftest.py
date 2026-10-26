"""Unit tests use deterministic recognizers without downloading NLP weights."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from llm_management.deployments.catalog import DeploymentCatalog


class FakeAnalyzer:
    def analyze(self, text, **kwargs):
        results = []
        for entity, pattern in [
            ("EMAIL_ADDRESS", r"[\w.+-]+@[\w.-]+\.[A-Za-z]+"),
            ("PERSON", r"\bAlice Smith\b"),
            ("PHONE_NUMBER", r"\b07700 900123\b"),
        ]:
            results.extend(
                SimpleNamespace(
                    entity_type=entity, start=m.start(), end=m.end(), score=0.9
                )
                for m in re.finditer(pattern, text)
            )
        return results


@pytest.fixture(autouse=True)
def local_presidio_for_unit_tests(monkeypatch, request):
    if request.node.get_closest_marker("external"):
        return
    from llm_management.bootstrap import default_application

    monkeypatch.setattr(default_application().sanitizer(), "_analyzer", FakeAnalyzer())


@pytest.fixture(autouse=True)
def isolated_application(monkeypatch, request):
    if request.node.get_closest_marker("external") or not hasattr(
        request.module, "server"
    ):
        return
    from llm_management import server

    monkeypatch.setattr(server, "app", server.create_app())
    sanitizer = server.app.state.application.sanitizer()
    monkeypatch.setattr(sanitizer, "_analyzer", FakeAnalyzer())


@pytest.fixture
def api_catalog() -> DeploymentCatalog:
    """
    Small API catalog independent of operational provider/model choices.
    """
    from llm_management.deployments.catalog import DeploymentCatalog

    return DeploymentCatalog.model_validate(
        {
            "api": {
                "service": {
                    "openai": {"provider": "openai", "credential": "openai"},
                    "tavily": {"provider": "tavily", "credential": "tavily"},
                },
                "deployment": [
                    {
                        "slug": "research_test",
                        "service": "openai",
                        "model": "gpt-6.1-sol",
                        "reasoning_effort": "low",
                        "max_tokens": 1536,
                    }
                ],
                "search": [{"slug": "web_test", "service": "tavily"}],
            },
            "foi_email_finder": {"deployment": "research_test", "search": "web_test"},
        }
    )
