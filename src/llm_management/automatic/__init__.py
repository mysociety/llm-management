"""
Local automatic-response classification.
"""

from __future__ import annotations

from .pipeline import detect_automatic
from .schemas import AutomaticResult

__all__ = ["detect_automatic", "AutomaticResult"]
