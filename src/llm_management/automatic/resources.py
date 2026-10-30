"""
Lazy, bounded local classification of automatic email bodies.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from ..deployments import DeploymentCatalog, get_catalog
from ..deployments.local_config import AutoClassifierDeployment, AutoLogisticDeployment
from ..inference.classifiers import AuthenticatedSequenceClassifier
from ..inference.errors import ClassifierUnavailable
from ..inference.logistic_v1 import (
    BinaryPredictionV1,
    BinaryTextLogisticModelV1,
    load_model_v1,
    predict_binary,
)
from ..inference.resources import LocalResource, ResourceRegistry
from ..settings import settings
from .pipeline import Encoder

GATE_CUTOFF = 0.4292739638892871
POLICY_SHA256 = "47195de9b4cc57368a02b777dfc53dcb02a7c57c7de89f0f9cd5499b2f146d69"


class LogisticModels(NamedTuple):
    """
    Shared weights with separate cascade and standalone decision policies.
    """

    cascade: BinaryTextLogisticModelV1
    standalone: BinaryTextLogisticModelV1


class LogisticDetector:
    """
    Own shared logistic weights with distinct cascade and fast decision policies.

    Retain both validated policies behind one resource guard. Fast classification
    uses the standalone threshold, while normal classification uses the cascade gate.
    Their weight and feature caches are shared after release-policy validation.
    """

    def __init__(
        self,
        config: AutoLogisticDeployment,
        *,
        catalog: DeploymentCatalog | None = None,
        registry: ResourceRegistry | None = None,
    ) -> None:
        """
        Bind checkpoint metadata and register a cold guard without downloading files.
        """
        self.config = config
        self.checkpoint = (catalog if catalog is not None else get_catalog()).model[
            config.model_ref
        ]
        self.cascade: BinaryTextLogisticModelV1 | None = None
        self.standalone: BinaryTextLogisticModelV1 | None = None
        self.resource = (
            registry if registry is not None else ResourceRegistry()
        ).register(
            LocalResource(
                config.slug,
                self.load,
                self.unload,
            )
        )

    def load(self) -> LogisticModels:
        """
        Load both policies under a lease and verify their pinned release contract.

        Check policy hash, labels, Unicode version, inclusive decisions, gate cutoff and
        identical weights. Share cached feature structures, mark the guard ready and
        return the paired policies. Existing policies are reused on subsequent calls.
        """
        with self.resource.use():
            if self.cascade is None:
                from huggingface_hub import hf_hub_download

                models = []
                for artifact in (self.config.artifact, self.config.standalone_artifact):
                    path = hf_hub_download(
                        self.checkpoint.repo,
                        artifact,
                        revision=self.checkpoint.revision,
                        token=settings.huggingface_token or None,
                        cache_dir=settings.classifier_cache_dir,
                    )
                    models.append(
                        load_model_v1(Path(path), expected_sha256=POLICY_SHA256)
                    )
                gate, standalone = models
                for model in models:
                    if (
                        model.true_label != "automatic"
                        or model.false_label != "authored"
                        or not model.inclusive
                        or model.logistic.features.unicode_version != "16.0.0"
                    ):
                        raise ClassifierUnavailable(
                            "Unexpected automatic logistic policy"
                        )
                if (
                    gate.probability_threshold != GATE_CUTOFF
                    or gate.logistic != standalone.logistic
                ):
                    raise ClassifierUnavailable(
                        "Unexpected automatic logistic weights or gate"
                    )
                # Separate decision policies share weights and cached feature indices.
                standalone = standalone.model_copy(update={"logistic": gate.logistic})
                _ = gate.logistic.features.vocabulary_indices
                _ = gate.logistic.features.token_regex
                self.cascade, self.standalone = gate, standalone
            self.resource.mark_ready()
            assert self.cascade is not None and self.standalone is not None
            return LogisticModels(self.cascade, self.standalone)

    def predict(self, body: str, fast: bool = False) -> BinaryPredictionV1:
        """
        Lease shared weights and select standalone policy for fast mode or cascade gate.
        """
        with self.resource.use():
            gate, standalone = self.load()
            return predict_binary(body, standalone if fast else gate)

    def unload(self) -> None:
        """
        Drop both policies while retaining the controller and downloaded artifacts.
        """
        self.cascade = self.standalone = None


def logistic_detector(
    config: AutoLogisticDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> LogisticDetector:
    """
    Resolve the default application's retained owner or construct an explicit one.

    An explicit registry requires both configuration and catalog. Without a registry,
    use the default assembly's provider and reject explicit resource configuration.
    Construction registers a cold guard; loading remains a warm-up or inference operation.
    """
    if registry is not None:
        if config is None or catalog is None:
            raise ValueError(
                "Explicit resource construction requires config and catalog"
            )
        return LogisticDetector(config, catalog=catalog, registry=registry)
    from ..bootstrap import default_application

    if config is not None:
        raise ValueError(
            "Explicit resource configuration requires catalog and registry"
        )
    return default_application().automatic.logistic_provider()


def create_encoder_detector(
    config: AutoClassifierDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> AuthenticatedSequenceClassifier:
    """
    Construct a cold authenticated encoder with authored/automatic output order.

    Use the supplied registry and pinned checkpoint; loading is deferred until
    warm-up or classification through its lifecycle guard.
    """
    checkpoint = catalog.model[config.model_ref]
    encoder = AuthenticatedSequenceClassifier(
        model=checkpoint.repo,
        revision=checkpoint.revision,
        labels=("authored", "automatic"),
        max_tokens=config.max_tokens,
        max_units=settings.deberta_batch_size,
        batch_size=settings.deberta_batch_size,
        threads=settings.cpu_inference_threads,
        token=settings.huggingface_token or None,
        cache_dir=settings.classifier_cache_dir,
        model_kwargs={"dtype": "float32"},
        resource_name=config.slug,
        registry=registry,
    )
    return encoder


def encoder_detector(
    config: AutoClassifierDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> Encoder:
    """
    Resolve the default application's retained owner or construct an explicit one.

    An explicit registry requires both configuration and catalog. Without a registry,
    use the default assembly's provider and reject explicit resource configuration.
    Construction registers a cold guard; loading remains a warm-up or inference operation.
    """
    if registry is not None:
        if config is None or catalog is None:
            raise ValueError(
                "Explicit resource construction requires config and catalog"
            )
        return create_encoder_detector(config, catalog=catalog, registry=registry)
    from ..bootstrap import default_application

    if config is not None:
        raise ValueError(
            "Explicit resource configuration requires catalog and registry"
        )
    return default_application().automatic.encoder_provider()
