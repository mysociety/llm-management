"""
Lazy local SAR model resources and release policy checks.
"""

from __future__ import annotations

from ..deployments import SARLogisticDeployment, get_catalog
from ..deployments.catalog import DeploymentCatalog, SARClassifierDeployment
from ..inference.classifiers import (
    AuthenticatedSequenceClassifier as SARSequenceClassifier,
)
from ..inference.errors import ClassifierUnavailable
from ..inference.logistic_v1 import (
    BinaryPredictionV1,
    BinaryTextLogisticModelV1,
    load_model_v1,
    predict_binary,
)
from ..inference.resources import LocalResource, ResourceRegistry
from ..settings import settings

LOGISTIC_THRESHOLD = 0.0018603434604847564
UNICODE_VERSION = "16.0.0"


class LogisticDetector:
    """
    Own the pinned SAR logistic artifact behind a reusable local resource guard.

    Construction resolves metadata and registers a cold guard. Loading may download
    the artifact; unloading clears memory while keeping the controller and disk cache.
    """

    def __init__(
        self,
        config: SARLogisticDeployment | None = None,
        *,
        catalog: DeploymentCatalog | None = None,
        registry: ResourceRegistry | None = None,
    ) -> None:
        """
        Resolve checkpoint metadata and establish ownership without loading weights.
        """
        catalog = catalog if catalog is not None else get_catalog()
        self.config = config or catalog.local.get(
            catalog.require_sar().logistic, SARLogisticDeployment
        )
        self.checkpoint = catalog.model[self.config.model_ref]
        self._model = None
        self.resource = (
            registry if registry is not None else ResourceRegistry()
        ).register(LocalResource(self.config.slug, self.load, self.unload))

    def load(self) -> BinaryTextLogisticModelV1:
        """
        Load or reuse the pinned artifact under a lease and validate its SAR policy.

        Authenticate Hub downloads and check label order, inclusive threshold and Unicode
        version before caching the predictor and marking the guard ready.
        """
        with self.resource.use():
            if self._model is None:
                from pathlib import Path

                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    self.checkpoint.repo,
                    self.config.artifact,
                    revision=self.checkpoint.revision,
                    token=settings.huggingface_token or None,
                    cache_dir=settings.classifier_cache_dir,
                )
                model = load_model_v1(Path(path))
                if (
                    model.true_label != "sar"
                    or model.false_label != "not-sar"
                    or model.probability_threshold != LOGISTIC_THRESHOLD
                    or not model.inclusive
                    or model.logistic.features.unicode_version != UNICODE_VERSION
                ):
                    raise ClassifierUnavailable(
                        "Unexpected SAR logistic artifact policy"
                    )
                self._model = model
            self.resource.mark_ready()
            return self._model

    def unload(self) -> None:
        """
        Clear the predictor reference when its inactive guard releases it.
        """
        self._model = None

    def predict(self, raw_text: str) -> BinaryPredictionV1:
        """
        Hold a lease through artifact loading and logistic prediction on raw text.
        """
        with self.resource.use():
            return predict_binary(raw_text, self.load())


def logistic_detector(
    config: SARLogisticDeployment | None = None,
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
    return default_application().sar.logistic_provider()


def create_deberta_detector(
    config: SARClassifierDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> SARSequenceClassifier:
    """
    Construct a cold authenticated SAR encoder controller in the supplied registry.

    Use the pinned checkpoint and not-sar/sar output order. Weight and tokenizer
    loading happen during warm-up or classification, not in this factory.
    """
    checkpoint = catalog.model[config.model_ref]
    return SARSequenceClassifier(
        model=checkpoint.repo,
        revision=checkpoint.revision,
        labels=("not-sar", "sar"),
        max_tokens=config.max_tokens,
        max_units=1,
        batch_size=1,
        threads=settings.cpu_inference_threads,
        token=settings.huggingface_token or None,
        cache_dir=settings.classifier_cache_dir,
        model_kwargs={"dtype": "float32"},
        resource_name=config.slug,
        registry=registry,
    )


def deberta_detector(
    config: SARClassifierDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> SARSequenceClassifier:
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
        return create_deberta_detector(config, catalog=catalog, registry=registry)
    from ..bootstrap import default_application

    if config is not None:
        raise ValueError(
            "Explicit resource configuration requires catalog and registry"
        )
    return default_application().sar.encoder_provider()
