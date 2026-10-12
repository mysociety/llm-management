"""
Authenticated Exoscale management client construction.
"""

from __future__ import annotations

from exoscale.api.v2 import Client

from ..settings import settings
from .errors import LLMManagementError


def get_client(zone: str) -> Client:
    if not settings.exoscale_api_key or not settings.exoscale_api_secret:
        raise LLMManagementError(
            "EXOSCALE_API_KEY and EXOSCALE_API_SECRET must be set "
            "(via environment variables or .env file)."
        )
    return Client(settings.exoscale_api_key, settings.exoscale_api_secret, zone=zone)
