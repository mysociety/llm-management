"""
Shared example data for FOI email agent and service tests.
"""

from __future__ import annotations

from llm_management.foi.email_finder.schemas import FOIFinding

URL = "https://council.example/foi"
QUOTE = "Send new FOI requests to foi@council.example."


def finding(**updates: object) -> FOIFinding:
    data = dict(
        status="found",
        email="foi@council.example",
        email_type="foi_eir",
        evidence={"source_url": URL, "quote": QUOTE},
        explanation="Published guidance",
    )
    return FOIFinding.model_validate(data | updates)
