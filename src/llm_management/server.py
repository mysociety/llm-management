"""
ASGI entry point: assemble one runtime and bind its HTTP interface.
"""

from __future__ import annotations

from .bootstrap import assemble_runtime
from .http_app import create_app

app = create_app(runtime=assemble_runtime())
