"""
Bearer authentication for API requests.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ..settings import settings

PUBLIC_PATHS = {"/", "/docs/oauth2-redirect", "/health", "/openapi.json", "/redoc"}


async def require_bearer_auth_when_enabled(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    Require a configured Authorization Bearer token for API requests.

    Health and API documentation routes are always public.
    """
    if request.url.path in PUBLIC_PATHS or not settings.auth_tokens:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    scheme, _, token = auth_header.partition(" ")
    token_is_valid = scheme.lower() == "bearer" and any(
        secrets.compare_digest(token, configured_token)
        for configured_token in settings.auth_tokens.values()
    )
    if not token_is_valid:
        return JSONResponse(
            status_code=401,
            content={"detail": "Not authenticated"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)
