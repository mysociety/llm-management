"""
Typed deployment catalog; validation never imports or loads inference owners.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Self, TypeVar

import tomllib
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from ..settings import settings
from .api_config import APIDeployments, EmailFinderConfig
from .catalog_validation import (
    validate_checkpoints,
    validate_feature_bindings,
    validate_foi_bindings,
    validate_groups,
    validate_namespaces,
)
from .config import (
    ComputeDeploymentConfig,
    DeploymentConfig,
    ExoscaleDeployments,
)
from .errors import LLMManagementError
from .local_config import (
    AutoClassifierDeployment,
    AutoLogisticDeployment,
    CategoryMatcherDeployment,
    LocalDeployment,
    LocalModelDeployment,
    PresidioDeployment,
    QuestionSliceHeadDeployment,
    ResponseTokenizerDeployment,
    SARClassifierDeployment,
    SARLogisticDeployment,
    SequenceClassifierDeployment,
    TopicTokenizerDeployment,
)
from .resource_types import LocalDeploymentConfig

# Preserve established catalog imports; internal callers use the owning modules.
__all__ = [
    "AutoClassifierDeployment",
    "AutoLogisticDeployment",
    "BinaryCascadePipeline",
    "CatalogModel",
    "CategoryMatcherDeployment",
    "DeploymentCatalog",
    "DeploymentGroupConfig",
    "FOIPipeline",
    "LocalDeployment",
    "LocalDeploymentConfig",
    "LocalDeployments",
    "LocalModelDeployment",
    "ModelCheckpoint",
    "PresidioDeployment",
    "QuestionSliceHeadDeployment",
    "ResponseBudget",
    "ResponseTokenizerDeployment",
    "SARClassifierDeployment",
    "SARLogisticDeployment",
    "SanitizationConfig",
    "SequenceClassifierDeployment",
    "TopicBudget",
    "TopicTokenizerDeployment",
    "get_catalog",
]


class CatalogModel(BaseModel):
    """
    Reject unknown fields in catalog sections, bindings and budget settings.

    This base shares validation policy; it does not identify a resource type.
    """

    model_config = ConfigDict(extra="forbid")


class ModelCheckpoint(CatalogModel):
    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


L = TypeVar("L", bound=LocalDeployment)


class LocalDeployments(CatalogModel):
    deployment: list[LocalDeploymentConfig] = Field(default_factory=list)

    def get(self, slug: str, kind: type[L]) -> L:
        for deployment in self.deployment:
            if deployment.slug == slug:
                if not isinstance(deployment, kind):
                    raise LLMManagementError(
                        f"Local deployment {slug!r} requires {kind.__name__}"
                    )
                return deployment
        raise LLMManagementError(f"No local deployment found with slug {slug!r}")


class TopicBudget(CatalogModel):
    output_limit: int = Field(gt=0)
    min_output_tokens: int = Field(gt=0)
    output_overhead: int = Field(ge=0)
    output_tokens_per_question: int = Field(gt=0)

    @property
    def max_questions(self) -> int:
        return (
            self.output_limit - self.output_overhead
        ) // self.output_tokens_per_question

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        if self.min_output_tokens > self.output_limit:
            raise ValueError("Minimum topic output budget exceeds the output limit")
        if self.max_questions < 1:
            raise ValueError(
                "Topic output budget must accommodate at least one question"
            )
        return self


class ResponseBudget(CatalogModel):
    """
    Total context and reserved generation budget for FOI status inference.
    """

    context_tokens: int = Field(default=8192, gt=0)
    output_tokens: int = Field(default=2048, gt=0)

    @model_validator(mode="after")
    def validate_budget(self) -> Self:
        """
        Leave space for a prompt while reserving the complete output allowance.
        """
        if self.output_tokens >= self.context_tokens:
            raise ValueError("Response output budget must leave room for the prompt")
        return self


class FOIPipeline(CatalogModel):
    classifier: str
    head: str
    tokenizer: str
    extraction_deployment: str
    topic_deployment: str
    category_matcher: str | None = None
    response_deployment: str | None = Field(
        default=None,
        description="Remote FOI response/status checkpoint paired with its local tokenizer.",
    )
    response_tokenizer: str | None = None
    response: ResponseBudget = Field(default_factory=ResponseBudget)
    topic: TopicBudget

    @model_validator(mode="after")
    def validate_response_pair(self) -> Self:
        """
        Require the optional response deployment and tokenizer together.
        """
        if (self.response_deployment is None) != (self.response_tokenizer is None):
            raise ValueError(
                "FOI response deployment and tokenizer must be configured together"
            )
        return self


class BinaryCascadePipeline(CatalogModel):
    logistic: str
    classifier: str


class DeploymentGroupConfig(CatalogModel):
    slug: str = Field(pattern=r"^group_.+$")
    deployments: list[str] = Field(min_length=1)


class SanitizationConfig(CatalogModel):
    deployment: str


class DeploymentCatalog(CatalogModel):
    model: dict[str, ModelCheckpoint] = Field(default_factory=dict)
    exoscale: ExoscaleDeployments = Field(default_factory=ExoscaleDeployments)
    local: LocalDeployments = Field(default_factory=LocalDeployments)
    api: APIDeployments = Field(default_factory=APIDeployments)
    deployment_group: list[DeploymentGroupConfig] = Field(default_factory=list)
    foi: FOIPipeline | None = None
    foi_email_finder: EmailFinderConfig | None = None
    sar: BinaryCascadePipeline | None = None
    automatic: BinaryCascadePipeline | None = None
    sanitization: SanitizationConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_checkpoints(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        # Copy the input so validation also works with reused dictionaries.
        value = dict(value)
        value["model"] = TypeAdapter(dict[str, ModelCheckpoint]).validate_python(
            value.get("model", {})
        )
        remote = value.get("exoscale", {})
        if isinstance(remote, dict):
            remote = dict(remote)
            deployments = []
            for entry in remote.get("deployment", []):
                if isinstance(entry, dict) and entry.get("model_ref") is not None:
                    entry = dict(entry)
                    ref = entry["model_ref"]
                    checkpoint = value.get("model", {}).get(ref)
                    if checkpoint is None:
                        raise ValueError(f"Unknown model reference: {ref}")
                    repo = checkpoint.repo
                    if "model" in entry and entry["model"] != repo:
                        raise ValueError(
                            f"Deployment model disagrees with model reference: {ref}"
                        )
                    entry["model"] = repo
                deployments.append(entry)
            remote["deployment"] = deployments
            value["exoscale"] = remote
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """
        Check that configuration references resolve to compatible resources.

        Require unique deployment and group names, valid model and group member
        references, and correctly typed pipeline bindings. Paired resources must
        share checkpoints where the pipeline requires it. Validation does not
        load models, provision compute, or contact external providers.
        """
        validate_namespaces(self)
        validate_checkpoints(self)
        validate_groups(self)
        try:
            validate_foi_bindings(self)
            validate_feature_bindings(self)
        except LLMManagementError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @classmethod
    def load(cls, config_path: Path | None = None) -> DeploymentCatalog:
        path = config_path if config_path is not None else settings.deployment_config
        try:
            config = cls.model_validate(tomllib.loads(path.read_text()))
        except (OSError, ValueError) as exc:
            raise LLMManagementError(
                f"Cannot load deployment configuration {path}: {exc}"
            ) from exc
        for deployment in config.exoscale.deployment:
            if isinstance(deployment, ComputeDeploymentConfig):
                deployment.recipe  # Check template references without provisioning.
        return config

    def get(self, slug: str) -> DeploymentConfig:
        return self.exoscale.get(slug)

    def get_group(self, slug: str) -> DeploymentGroupConfig:
        for group in self.deployment_group:
            if group.slug == slug:
                return group
        raise LLMManagementError(f"No deployment group found with slug {slug!r}")

    def require_foi(self) -> FOIPipeline:
        if self.foi is None:
            raise LLMManagementError("FOI pipeline is not configured")
        return self.foi

    def require_sanitizer(self) -> PresidioDeployment:
        if self.sanitization is None:
            raise LLMManagementError("Sanitization is not configured")
        return self.local.get(self.sanitization.deployment, PresidioDeployment)

    def require_sar(self) -> BinaryCascadePipeline:
        if self.sar is None:
            raise LLMManagementError("SAR pipeline is not configured")
        return self.sar

    def require_automatic(self) -> BinaryCascadePipeline:
        """
        Resolve automatic classifier bindings when that feature is first used.

        Assembly can remain cold even when this optional pipeline is absent.
        """
        if self.automatic is None:
            raise ValueError("Application requires an automatic-response pipeline")
        return self.automatic


@lru_cache(maxsize=1)
def get_catalog() -> DeploymentCatalog:
    """
    Return cached validated configuration without registering resources.
    """
    return DeploymentCatalog.load()
