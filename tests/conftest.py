"""Unit tests use deterministic recognizers without downloading NLP weights."""

from __future__ import annotations

import re
from collections.abc import Generator
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from llm_management.bootstrap import Application
    from llm_management.deployments.catalog import DeploymentCatalog
    from llm_management.foi.backends import FOIResources
    from llm_management.sanitization import PresidioSanitizer


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


@pytest.fixture
def sanitizer() -> Generator[PresidioSanitizer, None, None]:
    """
    Provide an isolated sanitizer with deterministic recognizers and no model loads.
    """
    from llm_management.deployments import get_catalog
    from llm_management.inference.resources import ResourceRegistry
    from llm_management.sanitization import PresidioSanitizer

    owner = PresidioSanitizer(
        config=get_catalog().require_sanitizer(), registry=ResourceRegistry()
    )
    owner._analyzer = FakeAnalyzer()
    yield owner
    owner.resource.release()


@pytest.fixture
def application() -> Generator[Application, None, None]:
    """
    Assemble an isolated runtime and close its resources after the test.
    """
    import asyncio

    from llm_management.bootstrap import assemble_application

    runtime = assemble_application()
    runtime.sanitizer()._analyzer = FakeAnalyzer()
    yield runtime
    asyncio.run(runtime.close())


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
def foi_resources(application: Application) -> FOIResources:
    """
    Explicit FOI providers with the unit-test sanitizer and categories disabled.
    """
    return application.foi.model_copy(update={"category_matcher": None})


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


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Select paid model comparisons independently of frozen research fixtures.
    """
    group = parser.getgroup("FOI email model evaluations")
    group.addoption(
        "--foi-model",
        action="append",
        default=[],
        help="API model deployment slug; repeat to compare models.",
    )
    group.addoption(
        "--foi-eval-output",
        default=".state/foi-email-evaluations",
        help="Directory for complete model evaluation reports.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Run each requested model against each authority, or use the configured model.
    """
    if "foi_model_slug" in metafunc.fixturenames:
        slugs = metafunc.config.getoption("foi_model") or [None]
        metafunc.parametrize(
            "foi_model_slug", slugs, ids=[slug or "configured" for slug in slugs]
        )
