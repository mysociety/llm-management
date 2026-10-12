"""FastAPI entry point: assemble dependencies, install middleware and routers."""

import asyncio
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .api import agents, deployments, proxies, foi
from .api.auth import require_bearer_auth_when_enabled
from .bootstrap import Application, assemble_application
from .deployments.errors import DeploymentServiceError
from .deployments.errors import LLMManagementError
from .settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run idle cleanup and release the app's resources on shutdown."""
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
        await service.close()


async def deployment_error(request: Request, exc: LLMManagementError):
    logger.error("Deployment lifecycle failed: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Deployment could not be prepared; check server logs."},
    )


async def service_error(request: Request, exc: DeploymentServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app(application: Application | None = None) -> FastAPI:
    """Construct an API around explicit, cold application dependencies."""
    application = application if application is not None else assemble_application()
    service = application.deployments
    app = FastAPI(title="LLM Management Proxy", docs_url="/", lifespan=lifespan)
    app.state.service = service
    app.state.application = application
    app.add_exception_handler(LLMManagementError, deployment_error)
    app.add_exception_handler(DeploymentServiceError, service_error)

    @app.middleware("http")
    async def track_deployment_requests(request: Request, call_next):
        with service.track_requests():
            return await call_next(request)

    app.middleware("http")(require_bearer_auth_when_enabled)

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        """Report that the API process is running."""
        return {"status": "ok"}

    for routes in (deployments, proxies, agents):
        app.include_router(routes.build_router(service))
    app.include_router(foi.build_router(service, application.foi))
    return app


app = create_app()
