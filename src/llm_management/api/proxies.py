"""
OpenAI and System One HTTP forwarding.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx2
from fastapi import APIRouter, HTTPException, Request, Response

from ..transports import systemone
from .dependencies import ServiceDependency

router = APIRouter()


@router.api_route(
    "/deployments/{slug}/v1/{path:path}",
    methods=["POST"],
)
async def proxy_to_deployment(
    service: ServiceDependency, slug: str, path: str, request: Request
) -> Response:
    """
    Forward any request under /deployments/{slug}/v1/... to the
    corresponding Exoscale deployment endpoint, injecting the correct
    bearer token. Resets the idle-scaler timer for this deployment.
    """
    state = (await service.ensure_running(slug)).state

    target_url = f"{state.deployment_url.rstrip('/')}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("authorization", None)
    if state.api_key:
        headers["authorization"] = f"Bearer {state.api_key}"
    for h in ("host", "content-length", "transfer-encoding"):
        headers.pop(h, None)

    params = dict(request.query_params)

    async with httpx2.AsyncClient(timeout=300.0) as client:
        resp = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=params,
            content=body,
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


@contextmanager
def systemone_http_errors() -> Iterator[None]:
    """
    Translate transport failures without exposing upstream credentials.
    """
    try:
        yield
    except systemone.SystemOneTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except systemone.SystemOneUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/v1/systemone")
async def proxy_to_systemone(
    service: ServiceDependency, request: Request, deployment: str = "clef"
) -> Response:
    """
    Forward System One JSON through the shared Exoscale lifecycle.
    """
    state = (await service.ensure_running(deployment)).state
    with systemone_http_errors():
        response = await systemone.send_systemone(
            await request.body(),
            deployment_url=state.deployment_url,
            api_key=state.api_key,
            params=[
                (k, v)
                for k, v in request.query_params.multi_items()
                if k != "deployment"
            ],
        )
    # httpx2 decodes compressed bodies, so do not copy encoding or length headers.
    headers = {
        name: response.headers[name]
        for name in ("content-type", "retry-after")
        if name in response.headers
    }
    return Response(
        content=response.content, status_code=response.status_code, headers=headers
    )
