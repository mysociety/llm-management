"""
Build the HTTP interface around an explicitly supplied runtime.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .api import agents, classifiers, deployments, foi, mocks, proxies
from .api.auth import bearer_auth, require_bearer_auth_when_enabled
from .api.limits import HTTPRequestPolicy, RequestLimits
from .bootstrap import ApplicationRuntime
from .deployments.errors import DeploymentServiceError, LLMManagementError
from .inference.errors import ClassifierBusy
from .settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Run idle cleanup and release the app's resources on shutdown.
    """
    if not settings.auth_tokens:
        if not settings.allow_unauthenticated:
            raise RuntimeError(
                "AUTH_TOKENS must be configured. For local testing only, "
                "set ALLOW_UNAUTHENTICATED=true."
            )
        logger.warning(
            "ALLOW_UNAUTHENTICATED is enabled with no AUTH_TOKENS. "
            "The API accepts unauthenticated requests; use only for local testing."
        )
    async with app.state.runtime:
        yield


async def deployment_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Deployment lifecycle failed: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Deployment could not be prepared; check server logs."},
    )


async def service_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DeploymentServiceError)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app(
    runtime: ApplicationRuntime, *, request_policy: HTTPRequestPolicy | None = None
) -> FastAPI:
    """
    Bind HTTP routes to an existing runtime without assembling dependencies.

    The FastAPI lifespan starts and closes this runtime. Assembly is explicit at
    the entry point so tests and other hosts can supply independent runtimes.
    Capture request limits from settings here unless an explicit policy is supplied;
    the middleware retains that policy for the lifetime of this HTTP app.
    """
    app = FastAPI(title="LLM Management Proxy", docs_url="/", lifespan=lifespan)
    app.state.runtime = runtime
    app.add_exception_handler(LLMManagementError, deployment_error)
    app.add_exception_handler(DeploymentServiceError, service_error)

    policy = (
        request_policy
        if request_policy is not None
        else HTTPRequestPolicy.from_settings(settings)
    )
    app.add_middleware(RequestLimits, policy=policy)
    app.middleware("http")(require_bearer_auth_when_enabled)

    @app.exception_handler(ClassifierBusy)
    async def classifier_busy(request: Request, exc: ClassifierBusy) -> JSONResponse:
        return JSONResponse(
            {"detail": "Classifier is busy"},
            status_code=429,
            headers={"Retry-After": "1"},
        )

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        """
        Report that the API process is running.
        """
        return {"status": "ok"}

    for routes in (deployments, proxies, agents, foi, mocks, classifiers):
        app.include_router(routes.router, dependencies=[Depends(bearer_auth)])
    return app
