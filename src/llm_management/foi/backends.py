"""QuestionSlice runtime setup and CPU/Exoscale backend selection.

CPU runs the complete classifier. Exoscale serves mean-pooled encoder embeddings;
the original small classification head runs locally because the gateway does not
expose native classification. Generic model execution lives in the inference package.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict

from ..deployments import (
    QuestionSliceHeadDeployment,
    SequenceClassifierDeployment,
    get_catalog,
)
from ..deployments.catalog import DeploymentCatalog
from ..deployments.types import DeploymentAccess
from ..inference.classifiers import LocalSequenceClassifier
from ..inference.errors import ClassifierOutputError, ClassifierUnavailable
from ..inference.remote import classify_remote
from ..inference.resources import LocalResource, ResourceRegistry
from ..sanitization import PresidioSanitizer, Sanitized, require_sanitized
from ..settings import settings
from .schemas import UNIT_LABELS, ExtractionBackend


class ModernBertClassificationHead:
    """
    Own the original local head used with remote mean-pooled encoder embeddings.

    Retain only head weights and checkpoint configuration, not the remote encoder.
    Its resource lease protects loading and computation; validate pooling, labels,
    required tensors and embedding dimensions before producing probability rows.
    """

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        labels: tuple[str, ...],
        token: str | None = None,
        cache_dir: str | None = None,
        threads: int = 1,
        resource_name: str = "classification_head",
        registry: ResourceRegistry | None = None,
    ) -> None:
        """
        Register a cold head owner with pinned metadata and deferred ML imports.
        """
        self.model = model
        self.revision = revision
        self.labels = labels
        self.token = token
        self.cache_dir = cache_dir
        self.threads = threads
        self._lock = threading.RLock()
        self._weights = None
        self._config: Any = None
        self._torch: Any = None
        self.resource = (
            registry if registry is not None else ResourceRegistry()
        ).register(LocalResource(resource_name, self.load, self.unload))

    def unload(self) -> None:
        """
        Clear head tensors, configuration and runtime references when idle.
        """
        with self._lock:
            self._weights = self._config = self._torch = None

    def load(self) -> None:
        """
        Load the pinned head tensors under a lease and validate checkpoint compatibility.

        Download through authenticated Hub access when needed. Verify mean pooling, GELU,
        label order and required weight tensors before marking the resource ready.
        Reuse retained weights; initialization failures raise ClassifierUnavailable.
        """
        with self.resource.use(), self._lock:
            if self._weights is not None:
                return
            try:
                import torch
                from huggingface_hub import hf_hub_download
                from safetensors import safe_open
                from transformers import AutoConfig
            except ImportError as exc:
                raise ClassifierUnavailable(
                    "Run poetry install to install classification head dependencies"
                ) from exc
            try:
                kwargs: dict[str, Any] = dict(
                    revision=self.revision, token=self.token, cache_dir=self.cache_dir
                )
                config = AutoConfig.from_pretrained(self.model, **kwargs)
                if (
                    config.model_type != "modernbert"
                    or config.classifier_pooling != "mean"
                    or config.classifier_activation != "gelu"
                    or tuple(
                        (config.id2label or {}).get(i) for i in range(config.num_labels)
                    )
                    != self.labels
                ):
                    raise ClassifierUnavailable(
                        "Checkpoint is incompatible with mean-pooled ModernBERT head inference"
                    )
                path = hf_hub_download(self.model, "model.safetensors", **kwargs)
                with safe_open(path, framework="pt", device="cpu") as source:
                    weights = {
                        key: source.get_tensor(key).float()
                        for key in source.keys()
                        if key.startswith(("head.", "classifier."))
                    }
                required = {
                    "head.dense.weight",
                    "head.norm.weight",
                    "classifier.weight",
                    "classifier.bias",
                }
                if not required.issubset(weights):
                    raise ClassifierUnavailable(
                        "Checkpoint classification head is incomplete"
                    )
                torch.set_num_threads(self.threads)
                self._torch = torch
                self._config = config
                self._weights = weights
                self.resource.mark_ready()
            except ClassifierUnavailable:
                raise
            except Exception as exc:
                raise ClassifierUnavailable(
                    "Could not load the fine-tuned classification head"
                ) from exc

    def classify(self, embeddings: list[list[float]]) -> list[list[float]]:
        """
        Apply the original dense, normalization and classification layers to embeddings.

        Hold the resource lease throughout loading and inference. Reject empty,
        non-finite or incorrectly sized vectors and return rows in checkpoint label order.
        """
        with self.resource.use(), self._lock:
            self.load()
            assert self._weights is not None
            if not embeddings or any(
                len(row) != self._config.hidden_size
                or any(not math.isfinite(v) for v in row)
                for row in embeddings
            ):
                raise ClassifierOutputError(
                    "Invalid encoder embedding dimensions or values"
                )
            torch = self._torch
            weights = self._weights
            with torch.inference_mode():
                x = torch.tensor(embeddings, dtype=torch.float32)
                x = torch.nn.functional.linear(
                    x, weights["head.dense.weight"], weights.get("head.dense.bias")
                )
                x = torch.nn.functional.gelu(x)
                x = torch.nn.functional.layer_norm(
                    x,
                    (self._config.hidden_size,),
                    weights["head.norm.weight"],
                    weights.get("head.norm.bias"),
                    eps=self._config.norm_eps,
                )
                x = torch.nn.functional.linear(
                    x, weights["classifier.weight"], weights.get("classifier.bias")
                )
                return x.softmax(-1).tolist()


def create_question_classification_head(
    config: QuestionSliceHeadDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> ModernBertClassificationHead:
    """
    Construct the registry-owned cold head for the catalog's pinned checkpoint.
    """
    checkpoint = catalog.model[config.model_ref]
    return ModernBertClassificationHead(
        model=checkpoint.repo,
        revision=checkpoint.revision,
        labels=UNIT_LABELS,
        token=settings.huggingface_token or None,
        cache_dir=settings.classifier_cache_dir,
        threads=settings.cpu_inference_threads,
        resource_name=config.slug,
        registry=registry,
    )


def create_question_classifier(
    config: SequenceClassifierDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> LocalSequenceClassifier:
    """
    Construct the registry-owned cold CPU classifier with QuestionSlice label order.

    Bind application token, unit, batch and thread limits without loading weights.
    The complete model is needed only when this controller performs CPU inference.
    """
    checkpoint = catalog.model[config.model_ref]
    return LocalSequenceClassifier(
        model=checkpoint.repo,
        revision=checkpoint.revision,
        labels=UNIT_LABELS,
        max_tokens=config.max_tokens,
        max_units=settings.classifier_max_units,
        batch_size=settings.classifier_batch_size,
        threads=settings.cpu_inference_threads,
        cache_dir=settings.classifier_cache_dir,
        token=settings.huggingface_token or None,
        resource_name=config.slug,
        registry=registry,
        model_kwargs={"attn_implementation": "sdpa", "reference_compile": False},
    )


def question_classification_head(
    config: QuestionSliceHeadDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> ModernBertClassificationHead:
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
        return create_question_classification_head(
            config, catalog=catalog, registry=registry
        )
    from ..bootstrap import default_application

    if config is not None:
        raise ValueError(
            "Explicit resource configuration requires catalog and registry"
        )
    return default_application().foi.head()


def question_classifier(
    config: SequenceClassifierDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> LocalSequenceClassifier:
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
        return create_question_classifier(config, catalog=catalog, registry=registry)
    from ..bootstrap import default_application

    if config is not None:
        raise ValueError(
            "Explicit resource configuration requires catalog and registry"
        )
    return default_application().foi.classifier()


class FOIResources(BaseModel):
    """
    Cold model providers supplied by the application assembly.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    catalog: DeploymentCatalog
    sanitizer: Callable[[], PresidioSanitizer]
    classifier: Callable[[], LocalSequenceClassifier]
    head: Callable[[], ModernBertClassificationHead]
    tokenizer: Callable[[], LocalResource[Any]]


class ClassificationRows(NamedTuple):
    """
    Ordered prediction rows and provenance for one extraction backend.

    A missing revision indicates that the remote encoder's exact weight revision
    cannot be verified, even though its local head is loaded from a pinned checkpoint.
    """

    probabilities: list[list[float]]
    model: str
    revision: str | None


async def classify_question_units(
    texts: Sanitized[list[str]],
    *,
    backend: ExtractionBackend,
    deployments: DeploymentAccess,
    resources: FOIResources | None = None,
) -> ClassificationRows:
    """
    Classify attested contextual text using CPU weights or remote encoder embeddings.

    CPU execution returns the pinned classifier revision. Remote execution checks the
    same input budgets, ensures the configured encoder and applies its pinned head
    locally. Record deployment activity even on failure; report no verified encoder
    revision because repository-name serving does not establish its weight SHA.
    """
    clean_texts = require_sanitized(texts)
    classifier = (
        resources.classifier() if resources is not None else question_classifier()
    )
    if backend == "cpu":
        rows = await asyncio.to_thread(classifier.classify, clean_texts)
        return ClassificationRows(rows, classifier.model_name, classifier.revision)
    if not settings.question_slice_gpu_enabled:
        raise ClassifierUnavailable(
            "Exoscale extraction is disabled by QUESTION_SLICE_GPU_ENABLED."
        )
    await asyncio.to_thread(classifier.validate_inputs, clean_texts)
    # The managed gateway exposes embeddings, so the trained head runs locally.
    # A custom Compute template could serve the encoder and head together.
    head = resources.head() if resources is not None else question_classification_head()
    await asyncio.to_thread(head.load)
    catalog = resources.catalog if resources is not None else get_catalog()
    slug = catalog.require_foi().extraction_deployment
    cfg = deployments.get_config(slug)
    if cfg.model != classifier.model_name:
        raise ClassifierUnavailable(
            "Extraction deployment must use the configured QuestionSlice model"
        )
    deployment = await deployments.ensure_running(slug)
    cfg = deployment.config
    state = deployment.state
    try:
        rows = await classify_remote(
            texts=texts,
            model=cfg.model,
            deployment_url=state.deployment_url,
            api_key=state.api_key,
            batch_size=classifier.batch_size,
            # A remote classification endpoint would return these probabilities directly.
            embedding_head=head.classify,
        )
    finally:
        deployments.touch(slug)
    # Exoscale imports by repository name; the remote weight SHA is unverified.
    return ClassificationRows(rows, classifier.model_name, None)
