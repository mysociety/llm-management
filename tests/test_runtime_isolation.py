"""
Runtime assembly isolates mutable configuration and captures execution policy.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

from llm_management.bootstrap import assemble_runtime


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
