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
import secrets
import time
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from contextvars import ContextVar

import httpx
import httpx2
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.system_one import SystemOneProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agents.capital_city import CapitalCityResponse, capital_city_agent
from .agents.immigration_detection import (
    ClassificationResponse,
    immigration_detection_agent,
    immigration_decision_agent,
)
from .cache import DeploymentState, cache
from .models import ExoscaleConfig, DeploymentConfig, LLMManagementError
from .settings import settings
from . import systemone
from .errors import ClassifierBusy, ClassifierUnavailable
from .local_resources import local_resources
from .foi import backends, pipeline, response_analysis
from .foi.response_schemas import ExtractionOutput, ResponseAnalysisInput
from .foi.schemas import (
    ExtractionBackend,
    InformationRequestResult,
    QuestionSliceResult,
)

logger = logging.getLogger("llm_management.server")

IDLE_TIMEOUT_MINUTES: int = 15  # minutes
_IDLE_CHECK_INTERVAL: int = 60  # seconds
AUTO_ENSURE_ON_REQUEST: bool = True
_request_deployments: ContextVar[set[str] | None] = ContextVar(
    "request_deployments", default=None
)


class DeploymentStatusResponse(BaseModel):
    slug: str
    exists: bool
    replicas: int


class EnsureResponse(BaseModel):
    slug: str
    action: str
    replicas: int | None


class GroupMemberResult(BaseModel):
    slug: str
    success: bool
    replicas: int | None = None
    error: str | None = None


class GroupEnsureResponse(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupMemberResult]


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
def get_deployment_config(slug: str) -> DeploymentConfig:
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


async def ensure_running(
    slug: str, *, allow_start: bool = False
) -> tuple[DeploymentConfig, DeploymentState]:
    """
    Return the config and a live deployment state for *slug*.

    If the deployment is not running and AUTO_ENSURE_ON_REQUEST is True,
    start it (with a per-slug lock so concurrent requests don't race).
    Otherwise raise 503. Explicit group warm-up sets allow_start=True.
    """
    cfg = get_deployment_config(slug)
    async with cache.ensure_lock(slug):
        state = await asyncio.to_thread(cache.ensure, cfg)
        if not state.exists or state.replicas == 0:
            if not (allow_start or AUTO_ENSURE_ON_REQUEST):
                raise HTTPException(
                    status_code=503,
                    detail=f"Deployment '{slug}' is not running. Ensure it first.",
                )
            logger.info("Auto-ensuring deployment %s.", slug)
            cache.touch(slug)
            # Let a startup finish under the lock if its initiating request is
            # cancelled, then track the VM so idle/shutdown cleanup can find it.
            startup = asyncio.create_task(asyncio.to_thread(cfg.create_or_resume))
            try:
                await asyncio.shield(startup)
            except asyncio.CancelledError:
                await startup
                await asyncio.to_thread(cache.refresh, cfg)
                cache.touch(slug)
                raise
            except Exception:
                if getattr(cfg, "backend", "exoscale_managed") == "exoscale_compute":
                    try:
                        await asyncio.to_thread(cache.refresh, cfg)
                        cache.touch(slug)
                    except Exception:
                        logger.exception(
                            "Failed to refresh Compute resources after startup failure."
                        )
                raise
            state = await asyncio.to_thread(cache.refresh, cfg)
            if not state.exists or state.replicas == 0:
                raise HTTPException(
                    status_code=503, detail=f"Deployment '{slug}' did not become ready."
                )
        cache.touch(slug)
        leases = _request_deployments.get()
        if leases is not None and slug not in leases:
            cache.begin_request(slug)
            leases.add(slug)
    return cfg, state


async def chat_model_from_slug(slug: str) -> OpenAIChatModel:
    """
    Return an OpenAI-compatible chat model backed by the given deployment.
    Ensures the deployment is cached and running, touches the idle timer,
    and raises 503 if the deployment is not available.
    """
    cfg = get_deployment_config(slug)
    if getattr(cfg, "protocol", "openai") != "openai":
        raise HTTPException(
            status_code=400, detail="This endpoint requires an OpenAI deployment."
        )
    cfg, state = await ensure_running(slug)
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
            await asyncio.to_thread(
                local_resources.expire, settings.cpu_idle_timeout_minutes * 60
            )
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
                        async with cache.ensure_lock(ds.slug):
                            current = cache.get(ds.slug)
                            if (
                                current is None
                                or current.requests_in_flight
                                or time.time() - current.last_request_time
                                < timeout_seconds
                            ):
                                continue
                            cfg = get_deployment_config(ds.slug)
                            await asyncio.to_thread(cfg.scale_to_zero)
                            await asyncio.to_thread(cache.refresh, cfg)
                    except Exception:
                        logger.exception("Failed to auto-scale %s to zero.", ds.slug)
        except Exception:
            logger.exception("Error in idle scaler loop.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the idle-scaler background task on startup. On shutdown,
    cancel the scaler and scale to zero any deployments that received
    traffic during this session. Local resources load only on demand or through
    explicit warm-up; startup does not preload models.
    """
    if not settings.auth_tokens:
        logger.warning(
            "AUTH_TOKENS is empty — the server API is open to unauthenticated "
            "requests. "
            "Ensure this instance is protected at the network level (e.g. behind a "
            "VPN, firewall, or authenticating reverse proxy) before exposing it to "
            "internet traffic."
        )

    task = asyncio.create_task(idle_scaler())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    try:
        await asyncio.to_thread(local_resources.close)
    except Exception:
        logger.exception("Failed to close local resources on shutdown.")
    active = cache.all_active()
    for ds in active:
        logger.info("Shutdown: scaling %s to zero.", ds.slug)
        try:
            cfg = get_deployment_config(ds.slug)
            await asyncio.to_thread(cfg.scale_to_zero)
        except Exception:
            logger.exception("Failed to scale %s to zero on shutdown.", ds.slug)


app = FastAPI(title="LLM Management Proxy", docs_url="/", lifespan=lifespan)


@app.exception_handler(LLMManagementError)
async def deployment_error(request: Request, exc: LLMManagementError):
    logger.error("Deployment lifecycle failed: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Deployment could not be prepared; check server logs."},
    )


@app.middleware("http")
async def track_deployment_requests(request: Request, call_next):
    leases: set[str] = set()
    token = _request_deployments.set(leases)
    try:
        return await call_next(request)
    finally:
        for slug in leases:
            cache.end_request(slug)
        _request_deployments.reset(token)


PUBLIC_PATHS = {"/", "/docs/oauth2-redirect", "/health", "/openapi.json", "/redoc"}


@app.middleware("http")
async def require_bearer_auth_when_enabled(request: Request, call_next):
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


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    """
    Report that the API process is running.
    """
    return {"status": "ok"}


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


async def warmup_resource(name: str) -> int | None:
    """Warm a registry match locally, otherwise ensure a remote deployment.

    Local success has no replica count. Both paths reset their usual idle timer;
    callers choose whether failures become HTTP errors or per-group results.
    """
    resource = local_resources.get(name)
    if resource is None:
        _, state = await ensure_running(name, allow_start=True)
        return state.replicas
    try:
        await asyncio.to_thread(resource.warmup)
    except ClassifierUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Local resource %s unavailable: %s", name, type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Local resource unavailable"
        ) from exc
    return None


@app.post("/deployments/{slug}/ensure")
async def ensure_deployment(slug: str) -> EnsureResponse:
    """Warm a local registry match or create/resume a remote deployment."""
    if local_resources.get(slug) is not None:
        replicas = await warmup_resource(slug)
        return EnsureResponse(slug=slug, action="warmed", replicas=replicas)
    cfg = get_deployment_config(slug)
    previous = await asyncio.to_thread(cache.ensure, cfg)
    replicas = await warmup_resource(slug)
    action = (
        "already_running"
        if previous.replicas > 0
        else "resumed"
        if previous.exists
        else "created"
    )
    return EnsureResponse(slug=slug, action=action, replicas=replicas)


@app.post("/deployment-groups/{slug}/ensure")
async def ensure_deployment_group(slug: str) -> GroupEnsureResponse:
    """Prepare group members concurrently; report failures without rolling back peers."""
    try:
        group = load_config().get_group(slug)
    except LLMManagementError:
        raise HTTPException(
            status_code=404, detail=f"Deployment group '{slug}' not found"
        )

    async def ensure_member(member: str) -> GroupMemberResult:
        try:
            replicas = await warmup_resource(member)
            return GroupMemberResult(slug=member, success=True, replicas=replicas)
        except Exception as exc:
            # Provider errors may contain credentials; do not return their raw text.
            logger.warning("Group ensure failed for %s: %s", member, type(exc).__name__)
            return GroupMemberResult(
                slug=member, success=False, error="Resource could not be warmed"
            )

    results = await asyncio.gather(
        *(ensure_member(member) for member in group.deployments)
    )
    return GroupEnsureResponse(
        slug=slug, success=all(r.success for r in results), deployments=results
    )


@app.post("/deployments/{slug}/scale-to-zero")
async def scale_to_zero(slug: str) -> ScaleToZeroResponse:
    """Pause managed inference or delete a Compute VM and its access resources."""
    cfg = get_deployment_config(slug)
    async with cache.ensure_lock(slug):
        state = cache.get(slug)
        if state and state.requests_in_flight:
            raise HTTPException(
                status_code=409, detail="Deployment has requests in flight."
            )
        await asyncio.to_thread(cfg.scale_to_zero)
        await asyncio.to_thread(cache.refresh, cfg)
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
    cfg, state = await ensure_running(slug)

    target_url = f"{state.deployment_url.rstrip('/')}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("authorization", None)
    if state.api_key:
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


@contextmanager
def systemone_http_errors():
    """Translate transport failures without exposing upstream credentials."""
    try:
        yield
    except systemone.SystemOneTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except systemone.SystemOneUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/systemone")
async def proxy_to_systemone(request: Request, deployment: str = "clef"):
    """Forward System One JSON through the shared Exoscale lifecycle."""
    _, state = await ensure_running(deployment)
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
    # httpx decodes compressed bodies, so do not copy encoding or length headers.
    headers = {
        name: response.headers[name]
        for name in ("content-type", "retry-after")
        if name in response.headers
    }
    return Response(
        content=response.content, status_code=response.status_code, headers=headers
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
    model = await chat_model_from_slug(deployment)
    return await capital_city_agent(model=model, country=body.country)


class FOiRequestContainer(BaseModel):
    request: str


@app.post("/agents/immigration_detection")
async def immigration_detection_endpoint(
    body: FOiRequestContainer, deployment: str = "toast_llama"
) -> ClassificationResponse:
    """
    Example agent endpoint. Takes a request and returns its classification
    as a plain text response ("IMM" or "FOI") via a pydantic-ai Agent
    running on the specified deployment.
    """
    model = await chat_model_from_slug(deployment)
    return await immigration_detection_agent(model=model, request=body.request)


class ClefImmigrationRequest(BaseModel):
    request: str = Field(min_length=1, max_length=100_000)


@app.post("/agents/immigration_detection/clef")
async def clef_immigration_detection_endpoint(
    body: ClefImmigrationRequest,
    deployment: str = "clef",
) -> ClassificationResponse:
    """Classify a request with one Clef choice question using existing labels."""
    cfg, state = await ensure_running(deployment)
    # The provider owns the /v1/systemone URL and response validation.
    async with httpx2.AsyncClient(timeout=300.0) as client:
        model = SystemOneModel(
            cfg.model,
            provider=SystemOneProvider(
                base_url=state.deployment_url,
                api_key=state.api_key,
                http_client=client,
            ),
        )
        try:
            return await immigration_decision_agent(model=model, request=body.request)
        except ModelHTTPError as exc:
            raise HTTPException(
                status_code=502, detail="Clef classification request failed"
            ) from exc
        except ModelAPIError as exc:
            if isinstance(exc.__cause__, httpx2.TimeoutException):
                raise HTTPException(
                    status_code=504, detail="Clef server timed out"
                ) from exc
            raise HTTPException(
                status_code=503, detail="Clef server could not be reached"
            ) from exc
        except UnexpectedModelBehavior as exc:
            raise HTTPException(
                status_code=502, detail="Clef returned an invalid classification"
            ) from exc


class QuestionSliceRequest(BaseModel):
    request: str = Field(min_length=1, max_length=100_000)


def foi_deployments() -> backends.DeploymentAccess:
    """Bind the shared deployment lifecycle to the HTTP-independent FOI pipeline."""
    return backends.DeploymentAccess(get_deployment_config, ensure_running, cache.touch)


@contextmanager
def foi_http_errors():
    """Translate domain/runtime failures only at the HTTP boundary."""
    try:
        yield
    except ClassifierBusy as exc:
        raise HTTPException(
            status_code=429, detail=str(exc), headers={"Retry-After": "1"}
        ) from exc
    except pipeline.PipelineInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except pipeline.PipelineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except pipeline.PipelineOutputError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/agents/foi_structure/extract")
async def question_slice_endpoint(
    body: QuestionSliceRequest,
    backend: ExtractionBackend = "cpu",
) -> QuestionSliceResult:
    """Extract source-grounded questions with either classifier backend.

    When no question starts exist, promote the first continuation locally.
    No OLMo escalation or Granite classification. CPU overload returns 429.
    """
    with foi_http_errors():
        return await pipeline.extract_questions(
            body.request, backend=backend, deployments=foi_deployments()
        )


@app.post("/agents/foi_structure")
async def information_request_endpoint(
    body: QuestionSliceRequest,
    backend: ExtractionBackend = "cpu",
) -> InformationRequestResult:
    """QuestionSlice extraction followed by fine-tuned Granite topics/regimes."""
    with foi_http_errors():
        return await pipeline.process_information_request(
            body.request, backend=backend, deployments=foi_deployments()
        )


@app.post("/agents/foi_response_analysis", response_model=ExtractionOutput)
async def foi_response_analysis_endpoint(
    body: ResponseAnalysisInput,
) -> ExtractionOutput:
    """Analyze response sources against the questions extracted from a request."""
    with foi_http_errors():
        try:
            return await response_analysis.analyze_response(body)
        except response_analysis.ResponseModelNotReady as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc


@app.get("/local-models")
async def local_model_status() -> list[dict]:
    """CPU readiness and idle timers in this API worker."""
    return [resource.status() for resource in local_resources.all()]


@app.post("/local-models/{name}/ensure")
async def ensure_local_model(name: str) -> dict:
    """Load a CPU resource and reset its idle timer."""
    resource = local_resources.get(name)
    if resource is None:
        raise HTTPException(status_code=404, detail="Unknown local model")
    await warmup_resource(name)
    return resource.status()
