"""Named, reproducible template recipes, separate from deployment settings."""

from pathlib import Path
import tomllib
from typing import Any, Self

from exoscale.api.v2 import Client
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import LLMManagementError

TEMPLATE_CONFIG_PATH = Path("conf/exoscale_templates.toml")


class SmokeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] = Field(min_length=2)
    expected: str

    @model_validator(mode="after")
    def validate_expected(self) -> Self:
        if self.expected not in self.criteria:
            raise ValueError("Smoke-test expected answer must appear in criteria")
        return self


class TemplateRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(
        default="Prepared System One image and cached model; offline boot",
        max_length=255,
    )
    zone: str
    image: str = Field(min_length=1)
    backend: str = "clef"
    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(default=4096, gt=0)
    instance_type: str = "gpua5000.small"
    base_template_name: str = "Linux Ubuntu 24.04 LTS 64-bit"
    disk_size_gib: int = Field(default=100, ge=10)
    nvidia_driver_package: str = Field(
        default="nvidia-driver-570", pattern=r"^nvidia-driver-[0-9]+$"
    )
    port: int = Field(default=8000, ge=1, le=65535)
    startup_timeout: int = Field(default=1800, gt=0)
    poll_interval: int = Field(default=5, gt=0)
    ssh_cidr: str | None = None
    smoke: list[SmokeCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_backend(self) -> Self:
        # Extend the bootstrap and probes before enabling another backend.
        if self.backend != "clef":
            raise ValueError("Only the clef backend is currently supported")
        return self

    def find(self, client: Client) -> dict[str, Any] | None:
        matches = [
            t
            for t in client.list_templates(visibility="private").get("templates", [])
            if t.get("name") == self.name
        ]
        if len(matches) > 1:
            raise LLMManagementError(
                f"Multiple private templates named {self.name!r} in {self.zone}"
            )
        return matches[0] if matches else None

    def resolve(self, client: Client) -> dict[str, Any]:
        template = self.find(client)
        if template is None:
            raise LLMManagementError(
                f"Template {self.name!r} not found in {self.zone}; run templates create {self.slug}"
            )
        return template


class TemplateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    template: list[TemplateRecipe]

    @model_validator(mode="after")
    def validate_unique(self) -> Self:
        slugs = [r.slug for r in self.template]
        names = [(r.zone, r.name) for r in self.template]
        if len(set(slugs)) != len(slugs) or len(set(names)) != len(names):
            raise ValueError("Template slugs and names within a zone must be unique")
        return self

    @classmethod
    def load(cls, path: Path = TEMPLATE_CONFIG_PATH) -> Self:
        try:
            return cls.model_validate(tomllib.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            raise LLMManagementError(
                f"Cannot load template configuration {path}: {exc}"
            ) from exc

    def get(self, slug: str) -> TemplateRecipe:
        for recipe in self.template:
            if recipe.slug == slug:
                return recipe
        raise LLMManagementError(f"Unknown template recipe {slug!r}")
