"""
FastAPI entry point: assemble dependencies, install middleware and routers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .api import agents, classifiers, deployments, foi, mocks, proxies
from .api.auth import bearer_auth, require_bearer_auth_when_enabled
from .api.limits import RequestLimits
from .bootstrap import Application, assemble_application
from .deployments.errors import DeploymentServiceError, LLMManagementError
from .inference.errors import ClassifierBusy
from .settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Run idle cleanup and release the app's resources on shutdown.
    """
    service = app.state.service
    if not settings.auth_tokens:
        logger.warning(
            "AUTH_TOKENS is empty — the server API is open to unauthenticated "
            "requests. Ensure this instance is protected at the network level "
            "before exposing it to internet traffic."
        )
    task = asyncio.create_task(service.idle_scaler())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await app.state.application.sar.close()
        await app.state.application.automatic.close()
        await service.close()


async def deployment_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Deployment lifecycle failed: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Deployment could not be prepared; check server logs."},
    )


async def service_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, DeploymentServiceError)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app(application: Application | None = None) -> FastAPI:
    """
    Construct an API around explicit, cold application dependencies.
    """
    application = application if application is not None else assemble_application()
    service = application.deployments
    app = FastAPI(title="LLM Management Proxy", docs_url="/", lifespan=lifespan)
    app.state.service = service
    app.state.application = application
    app.add_exception_handler(LLMManagementError, deployment_error)
    app.add_exception_handler(DeploymentServiceError, service_error)

    @app.middleware("http")
    async def track_deployment_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        with service.track_requests():
            return await call_next(request)

    app.add_middleware(RequestLimits)
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


app = create_app()
