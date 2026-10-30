"""
Bound request buffering and admission before parsing JSON or starting inference.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from ..settings import Settings


class HTTPRequestPolicy(BaseModel):
    """
    Immutable buffering and admission limits captured for one HTTP app.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_body_bytes: int = Field(ge=1)
    max_concurrent_requests: int = Field(ge=1)
    body_timeout_seconds: float = Field(gt=0)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """
        Capture supplied process settings when the HTTP app is constructed.
        """
        return cls(
            max_body_bytes=settings.http_max_body_bytes,
            max_concurrent_requests=settings.http_max_concurrent_requests,
            body_timeout_seconds=settings.http_body_timeout_seconds,
        )


class RequestLimits:
    """
    Enforce one app's captured request limits before downstream body parsing.
    """

    def __init__(self, app: ASGIApp, *, policy: HTTPRequestPolicy) -> None:
        """
        Retain explicit limits and start with no active requests.
        """
        self.app = app
        self.policy = policy
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if self.active >= self.policy.max_concurrent_requests:
            await JSONResponse(
                {"detail": "Server is busy"},
                status_code=429,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return
        self.active += 1
        try:
            limit = self.policy.max_body_bytes
            for name, value in scope.get("headers", []):
                if name.lower() == b"content-length":
                    try:
                        size = int(value)
                    except ValueError:
                        size = -1
                    if size < 0 or size > limit:
                        await JSONResponse(
                            {"detail": "Invalid or excessive Content-Length"},
                            status_code=413 if size > limit else 400,
                        )(scope, receive, send)
                        return
            body = bytearray()
            try:
                async with asyncio.timeout(self.policy.body_timeout_seconds):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(body) + len(chunk) > limit:
                            await JSONResponse(
                                {"detail": "Request body too large"}, status_code=413
                            )(scope, receive, send)
                            return
                        body.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await JSONResponse(
                    {"detail": "Request body timed out"}, status_code=408
                )(scope, receive, send)
                return

            delivered = False

            async def bounded_receive() -> Message:
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, bounded_receive, send)
        finally:
            self.active -= 1
