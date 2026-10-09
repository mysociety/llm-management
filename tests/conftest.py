"""Unit tests use deterministic recognizers without downloading NLP weights."""

import re
from types import SimpleNamespace

import pytest

from llm_management.sanitization import presidio


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
    monkeypatch.setattr(presidio, "_analyzer", FakeAnalyzer())
