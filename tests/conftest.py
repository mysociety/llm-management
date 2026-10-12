"""Unit tests use deterministic recognizers without downloading NLP weights."""

import re
from types import SimpleNamespace

import pytest


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
