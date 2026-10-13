"""
OpenAI and System One HTTP forwarding.
"""

from __future__ import annotations

import httpx2
from fastapi import APIRouter, HTTPException, Request, Response

from .dependencies import ServiceDependency

router = APIRouter(tags=["Inference proxy"])


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

    params = httpx2.QueryParams(tuple(request.query_params.multi_items()))

    try:
        async with httpx2.AsyncClient(timeout=300.0) as client:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=params,
                content=body,
            )
    except httpx2.TimeoutException as exc:
        raise HTTPException(
            status_code=504, detail="Deployment server timed out"
        ) from exc
    except httpx2.RequestError as exc:
        raise HTTPException(
            status_code=503, detail="Deployment server could not be reached"
        ) from exc

    return Response(
        content=resp.content,
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
