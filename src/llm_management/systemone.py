"""Shared HTTP transport for an Exoscale System One deployment."""

from __future__ import annotations

import httpx


class SystemOneUnavailable(Exception):
    """The configured decision server is unavailable."""


class SystemOneTimeout(Exception):
    """The decision server timed out."""


async def send_systemone(
    body: bytes, *, deployment_url: str, api_key: str, params=None
) -> httpx.Response:
    """Send a decision request using server credentials, never caller headers."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            return await client.post(
                f"{deployment_url.rstrip('/')}/systemone",
                content=body,
                headers=headers,
                params=params,
            )
    except httpx.TimeoutException as exc:
        raise SystemOneTimeout("Clef server timed out") from exc
    except httpx.RequestError as exc:
        raise SystemOneUnavailable("Clef server could not be reached") from exc
