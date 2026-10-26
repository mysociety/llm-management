"""
Server-owned external API deployments, credentials, and model construction.

External APIs are deliberately separate from managed compute deployments: they
have no VM lifecycle, and permission to use them is decided by each pipeline.
"""

from __future__ import annotations

import re

import httpx2
from pydantic import SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from ..deployments.api_config import APIModelDeployment, APIModelService
from ..settings import Settings


class ExternalUnavailable(RuntimeError):
    """
    A server-owned provider or credential is missing or invalid.
    """


class ExternalFailure(RuntimeError):
    """
    An upstream API failed or exhausted its research budget.
    """


def credential(name: str, settings: Settings) -> str:
    """
    Resolve a service's credential reference to its environment-backed setting.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ExternalUnavailable("Invalid external credential reference")
    secret = getattr(settings, f"{name}_api_key", None)
    if not isinstance(secret, SecretStr):
        raise ExternalUnavailable("External API credential is unavailable")
    value = secret.get_secret_value().strip()
    if not value:
        raise ExternalUnavailable("External API credential is unavailable")
    return value


def build_model(
    deployment: APIModelDeployment,
    service: APIModelService,
    *,
    api_key: str,
    client: httpx2.AsyncClient,
) -> Model:
    """
    Construct a named API deployment without exposing vendor SDKs to agents.
    """
    provider = OpenAIProvider(
        api_key=api_key,
        base_url=str(service.base_url) if service.base_url else None,
        http_client=client,
    )
    if service.protocol == "responses":
        settings = OpenAIResponsesModelSettings(
            max_tokens=deployment.max_tokens,
            openai_store=False,
        )
        if deployment.reasoning_effort is not None:
            settings["openai_reasoning_effort"] = deployment.reasoning_effort
        return OpenAIResponsesModel(
            deployment.model, provider=provider, settings=settings
        )
    chat_settings = OpenAIChatModelSettings(max_tokens=deployment.max_tokens)
    if deployment.reasoning_effort is not None:
        chat_settings["openai_reasoning_effort"] = deployment.reasoning_effort
    return OpenAIChatModel(deployment.model, provider=provider, settings=chat_settings)
