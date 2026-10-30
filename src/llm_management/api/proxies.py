"""
OpenAI and System One HTTP forwarding.
"""

from __future__ import annotations

import httpx2
from fastapi import APIRouter, HTTPException, Request, Response

from ..settings import settings
from .dependencies import DependencyService

router = APIRouter(tags=["Inference proxy"])


@router.api_route(
    "/deployments/{slug}/v1/{path:path}",
    methods=["POST"],
)
async def proxy_to_deployment(
    service: DependencyService, slug: str, path: str, request: Request
) -> Response:
    """
    Forward any request under /deployments/{slug}/v1/... to the
    corresponding Exoscale deployment endpoint, injecting the correct
    bearer token. Resets the idle-scaler timer for this deployment.
    """
    # ASGI has decoded the path once. Reject residual encoding as well, so
    # upstream servers cannot decode another traversal layer.
    if (
        not path
        or path.startswith("/")
        or any(c in path for c in "%\\?#")
        or any(part in (".", "..", "") for part in path.split("/"))
        or any(ord(c) < 32 or ord(c) == 127 for c in path)
    ):
        raise HTTPException(status_code=400, detail="Invalid inference path")
    async with service.use(slug) as running:
        state = running.state

        target_url = f"{state.deployment_url.rstrip('/')}/{path}"
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("authorization", None)
        if state.api_key:
            headers["authorization"] = f"Bearer {state.api_key}"
        for h in ("host", "content-length", "transfer-encoding"):
            headers.pop(h, None)

        params = httpx2.QueryParams(tuple(request.query_params.multi_items()))

        try:
            async with httpx2.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=params,
                    content=body,
                ) as resp:
                    content = bytearray()
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        if (
                            len(content) + len(chunk)
                            > settings.proxy_max_response_bytes
                        ):
                            raise HTTPException(
                                status_code=502, detail="Upstream response too large"
                            )
                        content.extend(chunk)
        except httpx2.TimeoutException as exc:
            raise HTTPException(
                status_code=504, detail="Deployment server timed out"
            ) from exc
        except httpx2.RequestError as exc:
            raise HTTPException(
                status_code=503, detail="Deployment server could not be reached"
            ) from exc

        return Response(
            content=bytes(content),
            status_code=resp.status_code,
            headers={
                name: value
                for name, value in resp.headers.items()
                if name.lower()
                not in (
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "connection",
                )
            },
        )
