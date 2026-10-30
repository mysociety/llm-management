"""
Cold configuration for externally hosted model and search services.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

from .collections import SlugMappedList
from .errors import LLMManagementError


class APIConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class APIModelService(APIConfigModel):
    """
    A model provider connection shared by named API deployments.
    """

    provider: Literal["openai", "openai_compatible"]
    credential: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    protocol: Literal["responses", "chat"] = "responses"
    base_url: HttpUrl | None = None

    @model_validator(mode="after")
    def validate_connection(self) -> Self:
        if self.provider == "openai_compatible":
            if self.base_url is None:
                raise ValueError("An OpenAI-compatible service needs a base URL")
        elif self.base_url is not None:
            raise ValueError("Custom base URLs require an OpenAI-compatible service")
        return self


class APISearchService(APIConfigModel):
    """
    Search provider credentials, without model protocol settings.
    """

    provider: Literal["tavily"]
    credential: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


APIService = Annotated[
    APIModelService | APISearchService, Field(discriminator="provider")
]


class APIModelDeployment(APIConfigModel):
    """
    A reproducible model configuration, selected by slug.
    """

    slug: str = Field(min_length=1, frozen=True)
    service: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    max_tokens: int = Field(default=2048, ge=1)


class APISearchDeployment(APIConfigModel):
    slug: str = Field(min_length=1, frozen=True)
    service: str = Field(min_length=1)


class APIDeployments(APIConfigModel):
    service: dict[str, APIService] = Field(default_factory=dict)
    deployment: SlugMappedList[APIModelDeployment] = Field(default_factory=dict)
    search: SlugMappedList[APISearchDeployment] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_services(self) -> Self:
        slugs = [*self.deployment, *self.search]
        if len(slugs) != len(set(slugs)):
            raise ValueError("API deployment and search slugs must be unique")
        for deployment in self.deployment.values():
            service = self.service.get(deployment.service)
            if service is None:
                raise ValueError(f"Unknown API service: {deployment.service}")
            if not isinstance(service, APIModelService):
                raise ValueError("Model deployments require a model service")
        for search in self.search.values():
            service = self.service.get(search.service)
            if service is None:
                raise ValueError(f"Unknown API service: {search.service}")
            if not isinstance(service, APISearchService):
                raise ValueError("Search deployments require a search service")
        return self

    def model_service(self, name: str) -> APIModelService:
        service = self.service[name]
        if not isinstance(service, APIModelService):
            raise LLMManagementError(f"API service {name!r} is not a model service")
        return service

    def search_service(self, name: str) -> APISearchService:
        service = self.service[name]
        if not isinstance(service, APISearchService):
            raise LLMManagementError(f"API service {name!r} is not a search service")
        return service

    def get(self, slug: str) -> APIModelDeployment:
        """
        Resolve one API model deployment by its configured identity.
        """
        try:
            return self.deployment[slug]
        except KeyError as exc:
            raise LLMManagementError(
                f"No API model deployment found with slug {slug!r}"
            ) from exc

    def get_search(self, slug: str) -> APISearchDeployment:
        """
        Resolve one search deployment by its configured identity.
        """
        try:
            return self.search[slug]
        except KeyError as exc:
            raise LLMManagementError(
                f"No API search deployment found with slug {slug!r}"
            ) from exc


class EmailFinderConfig(APIConfigModel):
    deployment: str
    search: str | None = None
    request_limit: int = Field(default=20, ge=1, le=50)
    tool_limit: int = Field(default=30, ge=1, le=100)
    timeout_seconds: float = Field(default=180, gt=0, le=600)
    http_timeout_seconds: float = Field(default=30, gt=0, le=60)
    max_page_chars: int = Field(default=40_000, ge=1, le=100_000)
    max_download_bytes: int = Field(default=5_000_000, ge=1, le=10_000_000)
