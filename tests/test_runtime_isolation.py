"""
Runtime assembly isolates mutable configuration and captures execution policy.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_management.bootstrap import assemble_runtime
from llm_management.inference.errors import ClassifierBusy
from llm_management.settings import settings


def test_default_runtimes_have_independent_catalogs() -> None:
    first, second = assemble_runtime(), assemble_runtime()
    first_catalog, second_catalog = (
        first.deployments.catalog,
        second.deployments.catalog,
    )
    assert first_catalog is not second_catalog
    slug = next(iter(first_catalog.local.deployment))
    first_catalog.local.deployment.pop(slug)
    assert slug in second_catalog.local.deployment
    model = next(iter(first_catalog.model))
    original = second_catalog.model[model].repo
    first_catalog.model[model].repo = "changed/checkpoint"
    assert second_catalog.model[model].repo == original
    assert first.management.service is first.deployments
    assert second.management.service is second.deployments
    asyncio.run(first.close())
    asyncio.run(second.close())


@pytest.mark.parametrize("kind", ["sar", "automatic"])
def test_classifier_admission_keeps_its_assembled_policy(monkeypatch, kind) -> None:
    monkeypatch.setattr(settings, "logistic_max_pending", 1)
    runtime = assemble_runtime()
    classifier = runtime.sar if kind == "sar" else runtime.automatic
    entered, finish = threading.Event(), threading.Event()

    def predict(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return SimpleNamespace(result=False, positive_score=0.1)

    classifier.logistic_provider = lambda: Mock(predict=predict)
    monkeypatch.setattr(settings, "logistic_max_pending", 64)

    async def run() -> None:
        first = asyncio.create_task(classifier.detect("first", fast=True))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(ClassifierBusy, match="queue is full"):
                await classifier.detect("second", fast=True)
        finally:
            finish.set()
            await first
            await runtime.close()

    asyncio.run(run())


def test_importing_http_factory_does_not_assemble_a_runtime() -> None:
    code = """
import sys
from unittest.mock import Mock
import llm_management.bootstrap as bootstrap
bootstrap.assemble_runtime = Mock(side_effect=AssertionError("Unexpected assembly"))
from llm_management.http_app import create_app
bootstrap.assemble_runtime.assert_not_called()
assert "llm_management.server" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
