"""
FastAPI application serving as an intermediary between services and Exoscale.

Provides:
- Scaling management endpoints (check/create/resume, scale-to-zero)
- Request proxying to Exoscale deployment endpoints
- Agent endpoints with built-in prompts (e.g. /agents/capital_city)
- Automatic scale-to-zero after idle timeout
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from functools import lru_cache

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agents.capital_city import CapitalCityResponse, capital_city_agent
from .agents.immigration_detection import (
    ClassificationResponse,
    immigration_detection_agent,
)
from .cache import cache
from .models import ExoscaleConfig, ExoscaleDeploymentConfig, LLMManagementError

logger = logging.getLogger("llm_management.server")

IDLE_TIMEOUT_MINUTES: int = 15  # minutes
_IDLE_CHECK_INTERVAL: int = 60  # seconds


class DeploymentStatusResponse(BaseModel):
    slug: str
    exists: bool
    replicas: int


class EnsureResponse(BaseModel):
    slug: str
    action: str
    replicas: int


class ScaleToZeroResponse(BaseModel):
    slug: str
    action: str


class DeploymentOverview(BaseModel):
    """
    Summary of a single deployment's current state and idle timer.
    """

    slug: str
    deployment_name: str
    exists: bool
    replicas: int
    idle_seconds: float | None
    seconds_until_scale_to_zero: float | None


class AllDeploymentsResponse(BaseModel):
    """
    Overview of all configured deployments with their current state
    and time remaining before auto-scaling to zero.
    """

    idle_timeout_minutes: int
    deployments: list[DeploymentOverview]


@lru_cache
def load_config() -> ExoscaleConfig:
    """
    Load the ExoscaleConfig from the default config file.
    """
    return ExoscaleConfig.load()


@lru_cache
def get_deployment_config(slug: str) -> ExoscaleDeploymentConfig:
    """
    Look up a deployment by slug in the config, raising a 404 if not found.
    """
    config = load_config()
    try:
        return config.get(slug)
    except LLMManagementError:
        raise HTTPException(
            status_code=404, detail=f"Deployment '{slug}' not found in config."
        )


def chat_model_from_slug(slug: str) -> OpenAIChatModel:
    """
    Return an OpenAI-compatible chat model backed by the given deployment.
    Ensures the deployment is cached and running, touches the idle timer,
    and raises 503 if the deployment is not available.
    """
    cfg = get_deployment_config(slug)
    state = cache.ensure(cfg)
    if not state.exists or state.replicas == 0:
        raise HTTPException(
            status_code=503,
            detail=f"Deployment '{slug}' is not running. Ensure it first.",
        )
    cache.touch(slug)
    return OpenAIChatModel(
        cfg.model,
        provider=OpenAIProvider(api_key=state.api_key, base_url=state.deployment_url),
    )


async def idle_scaler():
    """
    Background task that runs on a fixed interval. Checks all deployments
    that have received traffic and scales any to zero if they have been
    idle longer than IDLE_TIMEOUT_MINUTES.
    """
    while True:
        await asyncio.sleep(_IDLE_CHECK_INTERVAL)
        try:
            timeout_seconds = IDLE_TIMEOUT_MINUTES * 60
            active = cache.all_active()
            for ds in active:
                elapsed = time.time() - ds.last_request_time
                if elapsed >= timeout_seconds:
                    logger.info(
                        "Deployment %s idle for %.0fs — scaling to zero.",
                        ds.slug,
                        elapsed,
                    )
                    try:
                        cfg = get_deployment_config(ds.slug)
                        await asyncio.to_thread(cfg.scale_to_zero)
                        cache.refresh(cfg)
                    except Exception:
                        logger.exception("Failed to auto-scale %s to zero.", ds.slug)
        except Exception:
            logger.exception("Error in idle scaler loop.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the idle-scaler background task on startup. On shutdown,
    cancel the scaler and scale to zero any deployments that received
    traffic during this session.
    """
    task = asyncio.create_task(idle_scaler())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    active = cache.all_active()
    for ds in active:
        logger.info("Shutdown: scaling %s to zero.", ds.slug)
        try:
            cfg = get_deployment_config(ds.slug)
            await asyncio.to_thread(cfg.scale_to_zero)
        except Exception:
            logger.exception("Failed to scale %s to zero on shutdown.", ds.slug)


app = FastAPI(title="LLM Management Proxy", docs_url="/", lifespan=lifespan)


@app.get("/deployments/{slug}/status")
def deployment_status(slug: str) -> DeploymentStatusResponse:
    """
    Check whether a specific deployment exists on Exoscale and return
    its current replica count.
    """
    cfg = get_deployment_config(slug)
    state = cache.ensure(cfg)
    return DeploymentStatusResponse(
        slug=slug, exists=state.exists, replicas=state.replicas
    )


@app.post("/deployments/{slug}/ensure")
def ensure_deployment(slug: str) -> EnsureResponse:
    """
    Ensure a deployment is running. If it doesn't exist it will be created;
    if it exists but is scaled to zero it will be resumed. Returns the
    action taken and the resulting replica count.
    """
    cfg = get_deployment_config(slug)
    state = cache.ensure(cfg)

    if state.exists and state.replicas > 0:
        cache.touch(slug)
        return EnsureResponse(
            slug=slug, action="already_running", replicas=state.replicas
        )

    cfg.create_or_resume()
    new_state = cache.refresh(cfg)
    cache.touch(slug)
    action = "created" if not state.exists else "resumed"
    return EnsureResponse(slug=slug, action=action, replicas=new_state.replicas)


@app.post("/deployments/{slug}/scale-to-zero")
def scale_to_zero(slug: str) -> ScaleToZeroResponse:
    """
    Scale a deployment down to zero replicas, effectively pausing it
    without destroying it.
    """
    cfg = get_deployment_config(slug)
    cfg.scale_to_zero()
    cache.refresh(cfg)
    return ScaleToZeroResponse(slug=slug, action="scaled_to_zero")


@app.get("/deployments")
def all_deployments_overview() -> AllDeploymentsResponse:
    """
    Return the current status of every configured deployment, including
    how long each has been idle and how many seconds remain before it
    will be automatically scaled to zero.
    """
    config = load_config()
    timeout_seconds = IDLE_TIMEOUT_MINUTES * 60
    now = time.time()
    overviews: list[DeploymentOverview] = []

    for cfg in config.deployment:
        state = cache.ensure(cfg)
        if state.last_request_time > 0 and state.replicas > 0:
            idle = now - state.last_request_time
            remaining = max(0.0, timeout_seconds - idle)
        else:
            idle = None
            remaining = None
        overviews.append(
            DeploymentOverview(
                slug=cfg.slug,
                deployment_name=cfg.deployment_name,
                exists=state.exists,
                replicas=state.replicas,
                idle_seconds=idle,
                seconds_until_scale_to_zero=remaining,
            )
        )

    return AllDeploymentsResponse(
        idle_timeout_minutes=IDLE_TIMEOUT_MINUTES,
        deployments=overviews,
    )


@app.api_route(
    "/deployments/{slug}/v1/{path:path}",
    methods=["POST"],
)
async def proxy_to_deployment(slug: str, path: str, request: Request):
    """
    Forward any request under /deployments/{slug}/v1/... to the
    corresponding Exoscale deployment endpoint, injecting the correct
    bearer token. Resets the idle-scaler timer for this deployment.
    """
    cfg = get_deployment_config(slug)
    state = cache.ensure(cfg)

    if not state.exists or state.replicas == 0:
        raise HTTPException(
            status_code=503,
            detail=f"Deployment '{slug}' is not running. POST /deployments/{slug}/ensure first.",
        )

    cache.touch(slug)

    target_url = f"{state.deployment_url.rstrip('/')}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers["authorization"] = f"Bearer {state.api_key}"
    for h in ("host", "content-length", "transfer-encoding"):
        headers.pop(h, None)

    params = dict(request.query_params)

    async with httpx.AsyncClient(timeout=300.0) as client:
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


class CapitalCityRequest(BaseModel):
    country: str


@app.post("/agents/capital_city")
async def capital_city_endpoint(
    body: CapitalCityRequest, deployment: str = "olmo3_7b"
) -> CapitalCityResponse:
    """
    Example agent endpoint. Takes a country name and returns its capital
    city as structured output via a pydantic-ai Agent running on the
    specified deployment.
    """
    model = chat_model_from_slug(deployment)
    return await capital_city_agent(model=model, country=body.country)


class ImmigrationDetectionRequest(BaseModel):
    request: str


@app.post("/agents/immigration_detection")
async def immigration_detection_endpoint(
    body: ImmigrationDetectionRequest, deployment: str = "toast_llama"
) -> ClassificationResponse:
    """
    Example agent endpoint. Takes a request and returns its classification
    as a plain text response ("IMM" or "FOI") via a pydantic-ai Agent
    running on the specified deployment.
    """
    model = chat_model_from_slug(deployment)
    return await immigration_detection_agent(model=model, request=body.request)
