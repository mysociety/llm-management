"""
Validated remote deployment configuration without cloud operations.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field

from ..settings import settings
from .collections import SlugMappedList
from .errors import LLMManagementError

if TYPE_CHECKING:
    from ..templates.config import TemplateRecipe


class ExoscaleDeploymentConfig(BaseModel):
    model_config = {"extra": "forbid"}
    backend: Literal["exoscale_managed"] = "exoscale_managed"
    protocol: Literal["openai"] = "openai"
    slug: str = Field(frozen=True)
    model: str
    model_ref: str | None = None
    gpu_type: str
    gpu_count: int
    replicas: int
    zone: str
    inference_engine_params: list[str] = []
    inference_engine_version: str | None = Field(
        default=None, pattern=r"^\d+\.\d+\.\d+$"
    )

    @property
    def deployment_name(self) -> str:
        """
        Name used for the deployment on Exoscale (slug + server_role suffix).
        """
        if settings.server_role:
            return f"{self.slug}_{settings.server_role}"
        return self.slug


class ComputeDeploymentConfig(BaseModel):
    """
    One prepared GPU VM, with settings inherited from its template recipe.
    """

    model_config = {"extra": "forbid"}
    slug: str = Field(pattern=r"^[a-zA-Z0-9_-]+$", frozen=True)
    backend: Literal["exoscale_compute"]
    protocol: Literal["systemone"] = "systemone"
    template: str

    @cached_property
    def recipe(self) -> TemplateRecipe:
        """
        Resolve and cache the referenced recipe from local template configuration.

        First access validates the reference and may raise LLMManagementError.
        It does not contact a provider or provision compute.
        """
        from ..templates.config import TemplateConfig

        return TemplateConfig.load().get(self.template)

    @property
    def model(self) -> str:
        return self.recipe.model

    @property
    def zone(self) -> str:
        return self.recipe.zone

    @property
    def replicas(self) -> int:
        return 1

    @property
    def deployment_name(self) -> str:
        return (
            f"{self.slug}_{settings.server_role}" if settings.server_role else self.slug
        )


DeploymentConfig = ExoscaleDeploymentConfig | ComputeDeploymentConfig


class ExoscaleDeployments(BaseModel):
    """
    Remote deployment configuration and selection within the catalog.
    """

    model_config = {"extra": "forbid"}

    deployment: SlugMappedList[DeploymentConfig] = Field(default_factory=dict)

    def get(self, slug: str) -> DeploymentConfig:
        try:
            return self.deployment[slug]
        except KeyError as exc:
            raise LLMManagementError(
                f"No deployment config found with slug '{slug}'."
            ) from exc

    def resolve(self, slug: Optional[str], all_: bool) -> list[DeploymentConfig]:
        if all_:
            return list(self.deployment.values())
        if slug is None:
            raise LLMManagementError("Provide a slug or use --all.")
        return [self.get(slug)]
