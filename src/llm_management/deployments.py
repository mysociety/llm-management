"""Typed deployment catalog; validation never imports or loads inference owners."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .models import (
    DeploymentConfig,
    ExoscaleDeployments,
    ComputeDeploymentConfig,
    LLMManagementError,
)
from .settings import settings


class CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCheckpoint(CatalogModel):
    repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class LocalDeployment(CatalogModel):
    slug: str = Field(min_length=1)


class LocalModelDeployment(LocalDeployment):
    model_ref: str = Field(min_length=1)


class QuestionSliceDeployment(LocalModelDeployment):
    loader: Literal["modernbert_classifier"]
    max_tokens: int = Field(gt=0)


class QuestionSliceHeadDeployment(LocalModelDeployment):
    loader: Literal["modernbert_head"]


class TopicTokenizerDeployment(LocalModelDeployment):
    loader: Literal["topic_tokenizer"]
    max_tokens: int = Field(gt=0)


class LogisticDeployment(LocalModelDeployment):
    loader: Literal["sar_logistic_v1"]
    artifact: str = Field(min_length=1)


class SARClassifierDeployment(LocalModelDeployment):
    loader: Literal["sar_deberta_v1"]
    max_tokens: int = Field(gt=0)


class PresidioDeployment(LocalDeployment):
    loader: Literal["presidio"]
    spacy_model: str = Field(min_length=1)
    score_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)


LocalDeploymentConfig = Annotated[
    QuestionSliceDeployment
    | QuestionSliceHeadDeployment
    | TopicTokenizerDeployment
    | LogisticDeployment
    | SARClassifierDeployment
    | PresidioDeployment,
    Field(discriminator="loader"),
]
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
    def validate_budget(self):
        if self.min_output_tokens > self.output_limit:
            raise ValueError("Minimum topic output budget exceeds the output limit")
        if self.max_questions < 1:
            raise ValueError(
                "Topic output budget must accommodate at least one question"
            )
        return self


class FOIPipeline(CatalogModel):
    classifier: str
    head: str
    tokenizer: str
    extraction_deployment: str
    topic_deployment: str
    topic: TopicBudget


class SARPipeline(CatalogModel):
    logistic: str
    classifier: str


class DeploymentGroupConfig(CatalogModel):
    slug: str = Field(min_length=1)
    deployments: list[str] = Field(min_length=1)


class SanitizationConfig(CatalogModel):
    deployment: str


class DeploymentCatalog(CatalogModel):
    model: dict[str, ModelCheckpoint] = Field(default_factory=dict)
    exoscale: ExoscaleDeployments = Field(default_factory=ExoscaleDeployments)
    local: LocalDeployments = Field(default_factory=LocalDeployments)
    deployment_group: list[DeploymentGroupConfig] = Field(default_factory=list)
    foi: FOIPipeline | None = None
    sar: SARPipeline | None = None
    sanitization: SanitizationConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_checkpoints(cls, value):
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
    def validate_references(self):
        remote_names = [d.slug for d in self.exoscale.deployment]
        local_names = [d.slug for d in self.local.deployment]
        if len(set(remote_names)) != len(remote_names) or len(set(local_names)) != len(
            local_names
        ):
            raise ValueError("Deployment slugs must be unique")
        if set(remote_names) & set(local_names):
            raise ValueError(
                "Remote deployments and local resources must have distinct names"
            )
        for deployment in [*self.exoscale.deployment, *self.local.deployment]:
            ref = getattr(deployment, "model_ref", None)
            if ref is not None:
                if ref not in self.model:
                    raise ValueError(f"Unknown model reference: {ref}")
                if isinstance(deployment, LocalModelDeployment):
                    continue
                if deployment.model != self.model[ref].repo:
                    raise ValueError(
                        f"Deployment model disagrees with model reference: {ref}"
                    )
        known = set(remote_names) | set(local_names)
        groups = set()
        for group in self.deployment_group:
            if group.slug in groups:
                raise ValueError(f"Duplicate deployment group: {group.slug}")
            groups.add(group.slug)
            if len(set(group.deployments)) != len(group.deployments):
                raise ValueError(f"Duplicate members in deployment group: {group.slug}")
            unknown = set(group.deployments) - known
            if unknown:
                raise ValueError(
                    f"Unknown resources in group {group.slug}: {sorted(unknown)}"
                )
        try:
            if self.foi:
                classifier = self.local.get(
                    self.foi.classifier, QuestionSliceDeployment
                )
                head = self.local.get(self.foi.head, QuestionSliceHeadDeployment)
                tokenizer = self.local.get(self.foi.tokenizer, TopicTokenizerDeployment)
                extraction = self.get(self.foi.extraction_deployment)
                topic = self.get(self.foi.topic_deployment)
                if (
                    classifier.model_ref != head.model_ref
                    or getattr(extraction, "model_ref", None) != classifier.model_ref
                ):
                    raise ValueError(
                        "QuestionSlice classifier, head and remote deployment must share a checkpoint"
                    )
                if getattr(topic, "model_ref", None) != tokenizer.model_ref:
                    raise ValueError(
                        "Topic deployment and tokenizer must share a checkpoint"
                    )
            if self.sanitization:
                self.local.get(self.sanitization.deployment, PresidioDeployment)
            if self.sar:
                self.local.get(self.sar.logistic, LogisticDeployment)
                self.local.get(self.sar.classifier, SARClassifierDeployment)
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

    def require_sar(self) -> SARPipeline:
        if self.sar is None:
            raise LLMManagementError("SAR pipeline is not configured")
        return self.sar


@lru_cache(maxsize=1)
def get_catalog() -> DeploymentCatalog:
    catalog = DeploymentCatalog.load()
    from .local_resources import register_configured_resources

    register_configured_resources(catalog)
    return catalog
