"""
Validate relationships between catalog entries without constructing runtime owners.

Schemas validate individual entries. These checks validate references across
entries: names, pinned checkpoint identities, warm-up groups and feature bindings.
Only configuration types are imported at runtime; importing this module is cold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import ExoscaleDeploymentConfig
from .local_config import (
    AutoClassifierDeployment,
    AutoLogisticDeployment,
    CategoryMatcherDeployment,
    LocalModelDeployment,
    PresidioDeployment,
    QuestionSliceHeadDeployment,
    ResponseTokenizerDeployment,
    SARClassifierDeployment,
    SARLogisticDeployment,
    SequenceClassifierDeployment,
    TopicTokenizerDeployment,
)
from .slugs import is_group_slug

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
    if any(is_group_slug(name) for name in api_names):
        raise ValueError("The group_ prefix is reserved for deployment groups")


def validate_checkpoints(catalog: DeploymentCatalog) -> None:
    """
    Resolve local and remote checkpoint references and verify remote identities.
    """
    for deployment in [
        *catalog.exoscale.deployment,
        *catalog.local.deployment,
    ]:
        if isinstance(deployment, CategoryMatcherDeployment):
            if deployment.encoder_ref not in catalog.model:
                raise ValueError(f"Unknown encoder reference: {deployment.encoder_ref}")
        if not isinstance(deployment, (LocalModelDeployment, ExoscaleDeploymentConfig)):
            continue
        ref = deployment.model_ref
        if ref is not None:
            if ref not in catalog.model:
                raise ValueError(f"Unknown model reference: {ref}")
            if not isinstance(deployment, ExoscaleDeploymentConfig):
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
    if any(is_group_slug(name) for name in known):
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
    Check FOI references, resource types and shared checkpoints across entries.

    Pydantic validates each configuration entry, but pipeline bindings are slug
    strings. Resolve them here to check that they select compatible resources.
    """
    foi = catalog.foi
    if foi is None:
        return
    # A valid slug can still name the wrong kind of resource. Typed lookups check
    # both existence and suitability for each feature role without loading models.
    if foi.category_matcher is not None:
        catalog.local.get(foi.category_matcher, CategoryMatcherDeployment)
    classifier = catalog.local.get(foi.classifier, SequenceClassifierDeployment)
    head = catalog.local.get(foi.head, QuestionSliceHeadDeployment)
    tokenizer = catalog.local.get(foi.tokenizer, TopicTokenizerDeployment)
    extraction = catalog.get(foi.extraction_deployment)
    topic = catalog.get(foi.topic_deployment)
    # Response analysis is optional. FOIPipeline has already required its remote
    # deployment and local tokenizer together; now validate what those slugs select.
    if foi.response_deployment is not None:
        response = catalog.get(foi.response_deployment)
        if not isinstance(response, ExoscaleDeploymentConfig):
            raise ValueError(
                "FOI response analysis requires a managed OpenAI deployment"
            )
        assert foi.response_tokenizer is not None
        response_tokenizer = catalog.local.get(
            foi.response_tokenizer, ResponseTokenizerDeployment
        )
        # Token counting must use the checkpoint served by the remote model.
        if response.model_ref != response_tokenizer.model_ref:
            raise ValueError(
                "Response deployment and tokenizer must share a checkpoint"
            )
    # CPU inference uses the full classifier; Exoscale inference combines the
    # remote encoder with a local head. Both paths must use the same checkpoint.
    if (
        classifier.model_ref != head.model_ref
        or not isinstance(extraction, ExoscaleDeploymentConfig)
        or extraction.model_ref != classifier.model_ref
    ):
        raise ValueError(
            "QuestionSlice classifier, head and remote deployment must share a checkpoint"
        )
    # Topic prompt budgeting also depends on a tokenizer matching the served model.
    if (
        not isinstance(topic, ExoscaleDeploymentConfig)
        or topic.model_ref != tokenizer.model_ref
    ):
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
