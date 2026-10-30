"""
Validate relationships between catalog entries without constructing runtime owners.

Schemas validate individual entries. These checks validate references across
entries: names, pinned checkpoint identities, warm-up groups and feature bindings.
Only configuration types are imported at runtime; importing this module is cold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ComputeDeploymentConfig, ExoscaleDeploymentConfig
from .local_config import (
    AutoClassifierDeployment,
    AutoLogisticDeployment,
    CategoryMatcherDeployment,
    PresidioDeployment,
    QuestionSliceHeadDeployment,
    ResponseTokenizerDeployment,
    SARClassifierDeployment,
    SARLogisticDeployment,
    SequenceClassifierDeployment,
    TopicTokenizerDeployment,
)

if TYPE_CHECKING:
    from .catalog import DeploymentCatalog


def validate_namespaces(catalog: DeploymentCatalog) -> None:
    """
    Require distinct deployment namespaces and reserve API group names.
    """
    remote_names = [d.slug for d in catalog.exoscale.deployment]
    local_names = [d.slug for d in catalog.local.deployment]
    if len(set(remote_names)) != len(remote_names) or len(set(local_names)) != len(
        local_names
    ):
        raise ValueError("Deployment slugs must be unique")
    if set(remote_names) & set(local_names):
        raise ValueError(
            "Remote deployments and local resources must have distinct names"
        )
    api_names = [d.slug for d in [*catalog.api.deployment, *catalog.api.search]]
    if (set(remote_names) | set(local_names)) & set(api_names):
        raise ValueError("API, remote and local deployments must have distinct names")
    if any(name.startswith("group_") for name in api_names):
        raise ValueError("The group_ prefix is reserved for deployment groups")


def validate_checkpoints(catalog: DeploymentCatalog) -> None:
    """
    Resolve local and remote checkpoint references and verify remote identities.
    """
    for deployment in [*catalog.exoscale.deployment, *catalog.local.deployment]:
        if isinstance(deployment, CategoryMatcherDeployment):
            if deployment.encoder_ref not in catalog.model:
                raise ValueError(f"Unknown encoder reference: {deployment.encoder_ref}")
        ref = getattr(deployment, "model_ref", None)
        if ref is not None:
            if ref not in catalog.model:
                raise ValueError(f"Unknown model reference: {ref}")
            if not isinstance(
                deployment, (ExoscaleDeploymentConfig, ComputeDeploymentConfig)
            ):
                continue
            if deployment.model != catalog.model[ref].repo:
                raise ValueError(
                    f"Deployment model disagrees with model reference: {ref}"
                )


def validate_groups(catalog: DeploymentCatalog) -> None:
    """
    Require unique warm-up groups with known, non-repeated resource members.
    """
    remote_names = [d.slug for d in catalog.exoscale.deployment]
    local_names = [d.slug for d in catalog.local.deployment]
    known = set(remote_names) | set(local_names)
    if any(name.startswith("group_") for name in known):
        raise ValueError("The group_ prefix is reserved for deployment groups")
    groups: set[str] = set()
    for group in catalog.deployment_group:
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


def validate_foi_bindings(catalog: DeploymentCatalog) -> None:
    """
    Check FOI resource types, optional response pairing and shared checkpoints.
    """
    if catalog.foi:
        if catalog.foi.category_matcher is not None:
            catalog.local.get(catalog.foi.category_matcher, CategoryMatcherDeployment)
        classifier = catalog.local.get(
            catalog.foi.classifier, SequenceClassifierDeployment
        )
        head = catalog.local.get(catalog.foi.head, QuestionSliceHeadDeployment)
        tokenizer = catalog.local.get(catalog.foi.tokenizer, TopicTokenizerDeployment)
        extraction = catalog.get(catalog.foi.extraction_deployment)
        topic = catalog.get(catalog.foi.topic_deployment)
        if (catalog.foi.response_deployment is None) != (
            catalog.foi.response_tokenizer is None
        ):
            raise ValueError(
                "FOI response deployment and tokenizer must be configured together"
            )
        if catalog.foi.response_deployment is not None:
            response = catalog.get(catalog.foi.response_deployment)
            if getattr(response, "protocol", "openai") != "openai":
                raise ValueError("FOI response analysis requires an OpenAI deployment")
            assert catalog.foi.response_tokenizer is not None
            response_tokenizer = catalog.local.get(
                catalog.foi.response_tokenizer, ResponseTokenizerDeployment
            )
            if getattr(response, "model_ref", None) != response_tokenizer.model_ref:
                raise ValueError(
                    "Response deployment and tokenizer must share a checkpoint"
                )
        if (
            classifier.model_ref != head.model_ref
            or getattr(extraction, "model_ref", None) != classifier.model_ref
        ):
            raise ValueError(
                "QuestionSlice classifier, head and remote deployment must share a checkpoint"
            )
        if getattr(topic, "model_ref", None) != tokenizer.model_ref:
            raise ValueError("Topic deployment and tokenizer must share a checkpoint")


def validate_feature_bindings(catalog: DeploymentCatalog) -> None:
    """
    Check email lookup, sanitization and classifier bindings against their schemas.
    """
    if catalog.foi_email_finder:
        catalog.api.get(catalog.foi_email_finder.deployment)
        if catalog.foi_email_finder.search:
            catalog.api.get_search(catalog.foi_email_finder.search)
    if catalog.sanitization:
        catalog.local.get(catalog.sanitization.deployment, PresidioDeployment)
    if catalog.automatic:
        catalog.local.get(catalog.automatic.logistic, AutoLogisticDeployment)
        catalog.local.get(catalog.automatic.classifier, AutoClassifierDeployment)
    if catalog.sar:
        catalog.local.get(catalog.sar.logistic, SARLogisticDeployment)
        catalog.local.get(catalog.sar.classifier, SARClassifierDeployment)
