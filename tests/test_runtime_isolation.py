"""
Runtime assembly isolates mutable configuration and captures execution policy.
"""

from __future__ import annotations

import subprocess
import sys


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
