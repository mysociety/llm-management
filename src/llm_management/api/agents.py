"""
HTTP adapters for example agents and immigration classification.
"""

from __future__ import annotations

import httpx2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.system_one import SystemOneProvider

from ..agents.capital_city import CapitalCityResponse, capital_city_agent
from ..agents.immigration_detection import (
    ClassificationResponse,
    immigration_decision_agent,
    immigration_detection_agent,
)
from .dependencies import ServiceDependency


class CapitalCityRequest(BaseModel):
    country: str


class FOIRequestContainer(BaseModel):
    body: str


class ClefImmigrationRequest(BaseModel):
    body: str = Field(min_length=1, max_length=100_000)


router = APIRouter()


async def chat_model_from_slug(
    slug: str, service: ServiceDependency
) -> OpenAIChatModel:
    """
    Return an OpenAI-compatible chat model backed by the given deployment.
    Ensures the deployment is cached and running, touches the idle timer,
    and raises 503 if the deployment is not available.
    """
    cfg = service.get_config(slug)
    if getattr(cfg, "protocol", "openai") != "openai":
        raise HTTPException(
            status_code=400, detail="This endpoint requires an OpenAI deployment."
        )
    deployment = await service.ensure_running(slug)
    cfg = deployment.config
    state = deployment.state
    return OpenAIChatModel(
        cfg.model,
        provider=OpenAIProvider(api_key=state.api_key, base_url=state.deployment_url),
    )


@router.post("/endpoint/misc/capital_city", tags=["Misc endpoints"])
async def capital_city_endpoint(
    service: ServiceDependency, body: CapitalCityRequest, deployment: str = "olmo3_7b"
) -> CapitalCityResponse:
    """
    Example agent endpoint. Takes a country name and returns its capital
    city as structured output via a pydantic-ai Agent running on the
    specified deployment.
    """
    model = await chat_model_from_slug(deployment, service)
    return await capital_city_agent(model=model, country=body.country)


@router.post("/endpoint/foi/immigration_detection", include_in_schema=False)
async def immigration_detection_endpoint(
    service: ServiceDependency,
    body: FOIRequestContainer,
    deployment: str = "toast_llama",
) -> ClassificationResponse:
    """
    Example agent endpoint. Takes a request and returns its classification
    as a plain text response ("IMM" or "FOI") via a pydantic-ai Agent
    running on the specified deployment.
    """
    model = await chat_model_from_slug(deployment, service)
    return await immigration_detection_agent(model=model, request=body.body)


@router.post("/endpoint/foi/immigration_detection/clef", include_in_schema=False)
async def clef_immigration_detection_endpoint(
    service: ServiceDependency,
    body: ClefImmigrationRequest,
    deployment: str = "clef",
) -> ClassificationResponse:
    """
    Classify a request with one Clef choice question using existing labels.
    """
    running = await service.ensure_running(deployment)
    cfg = running.config
    state = running.state
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
            return await immigration_decision_agent(model=model, request=body.body)
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
