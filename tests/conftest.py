"""Unit tests use deterministic recognizers without downloading NLP weights."""

from __future__ import annotations

import re
from collections.abc import Generator
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi import FastAPI

    from llm_management.bootstrap import ApplicationRuntime
    from llm_management.deployments.catalog import DeploymentCatalog
    from llm_management.foi.backends import FOIResources
    from llm_management.sanitization import PresidioSanitizer


@pytest.fixture(scope="session", autouse=True)
def external_test_run(request: pytest.FixtureRequest):
    """
    Isolate opted-in live tests before any application fixtures are assembled.
    """
    if not request.config.getoption("run_external") or not any(
        item.get_closest_marker("external") for item in request.session.items
    ):
        yield None
        return

    from tests.external_safety import isolated_external_run

    with isolated_external_run() as role:
        yield role


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    A marker selection alone must never authorize paid or destructive tests.
    """
    if not config.getoption("run_external"):
        skip = pytest.mark.skip(reason="Live tests require --run-external")
        for item in items:
            if item.get_closest_marker("external"):
                item.add_marker(skip)


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
def runtime() -> Generator[ApplicationRuntime, None, None]:
    """
    Assemble an isolated runtime and close its resources after the test.
    """
    import asyncio

    from llm_management.bootstrap import assemble_runtime

    runtime = assemble_runtime()
    runtime.sanitizer()._analyzer = FakeAnalyzer()
    yield runtime
    asyncio.run(runtime.close())


@pytest.fixture
def http_app(
    runtime: ApplicationRuntime, monkeypatch: pytest.MonkeyPatch
) -> Generator[FastAPI, None, None]:
    """
    Own an HTTP app and close its runtime before test patches are restored.

    Tests request this fixture explicitly. TestClient may run the lifespan itself;
    runtime.close() is idempotent, so teardown also covers clients without lifespan.
    """
    import asyncio

    from llm_management.http_app import create_app

    app = create_app(runtime=runtime)
    try:
        yield app
    finally:
        asyncio.run(runtime.close())


@pytest.fixture
def foi_resources(runtime: ApplicationRuntime) -> FOIResources:
    """
    Explicit FOI providers with the unit-test sanitizer and categories disabled.
    """
    return runtime.foi.model_copy(update={"category_matcher": None})


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
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Allow paid live tests in a fresh, isolated test deployment namespace.",
    )
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
