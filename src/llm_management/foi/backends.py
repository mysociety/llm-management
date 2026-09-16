"""QuestionSlice runtime setup and CPU/Exoscale backend selection.

CPU runs the complete classifier. Exoscale serves mean-pooled encoder embeddings;
the original small classification head runs locally because the gateway does not
expose native classification. Generic model execution lives in inference.py.
"""

from __future__ import annotations

import math
import threading
from functools import lru_cache

from ..errors import ClassifierOutputError, ClassifierUnavailable
from ..inference import LocalSequenceClassifier, classify_remote
from ..cache import DeploymentState
from ..models import ExoscaleDeploymentConfig
from .schemas import UNIT_LABELS, ExtractionBackend
from .model_spec import QUESTION_SLICE_INPUT_LIMIT
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
import asyncio
from ..settings import settings


class ModernBertClassificationHead:
    def __init__(
        self,
        *,
        model: str,
        revision: str,
        labels: tuple[str, ...],
        token: str | None = None,
        cache_dir: str | None = None,
        threads: int = 1,
    ):
        self.model = model
        self.revision = revision
        self.labels = labels
        self.token = token
        self.cache_dir = cache_dir
        self.threads = threads
        self._lock = threading.RLock()
        self._weights = None
        self._config = None
        self._torch = None

    def load(self):
        with self._lock:
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
                kwargs = dict(
                    revision=self.revision, token=self.token, cache_dir=self.cache_dir
                )
                config = AutoConfig.from_pretrained(self.model, **kwargs)
                if (
                    config.model_type != "modernbert"
                    or config.classifier_pooling != "mean"
                    or config.classifier_activation != "gelu"
                    or tuple(config.id2label.get(i) for i in range(config.num_labels))
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
            except ClassifierUnavailable:
                raise
            except Exception as exc:
                raise ClassifierUnavailable(
                    "Could not load the fine-tuned classification head"
                ) from exc

    def classify(self, embeddings: list[list[float]]) -> list[list[float]]:
        self.load()
        with self._lock:
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


@lru_cache(maxsize=1)
def question_classification_head() -> ModernBertClassificationHead:
    return ModernBertClassificationHead(
        model=settings.question_slice_model,
        revision=settings.question_slice_revision,
        labels=UNIT_LABELS,
        token=settings.huggingface_token or None,
        cache_dir=settings.classifier_cache_dir,
        threads=settings.cpu_inference_threads,
    )


@lru_cache(maxsize=1)
def question_classifier() -> LocalSequenceClassifier:
    return LocalSequenceClassifier(
        model=settings.question_slice_model,
        revision=settings.question_slice_revision,
        labels=UNIT_LABELS,
        max_tokens=QUESTION_SLICE_INPUT_LIMIT,
        max_units=settings.classifier_max_units,
        batch_size=settings.classifier_batch_size,
        threads=settings.cpu_inference_threads,
        cache_dir=settings.classifier_cache_dir,
        token=settings.huggingface_token or None,
        model_kwargs={"attn_implementation": "sdpa", "reference_compile": False},
    )


@dataclass(frozen=True)
class DeploymentAccess:
    """Deployment operations supplied by the application; no HTTP dependency here."""

    get_config: Callable[[str], ExoscaleDeploymentConfig]
    ensure_running: Callable[
        [str], Awaitable[tuple[ExoscaleDeploymentConfig, DeploymentState]]
    ]
    touch: Callable[[str], None]


@dataclass(frozen=True)
class ClassificationRows:
    probabilities: list[list[float]]
    model: str
    revision: str | None


async def classify_question_units(
    texts: list[str], *, backend: ExtractionBackend, deployments: DeploymentAccess
) -> ClassificationRows:
    """Run the full CPU classifier or the Exoscale encoder plus original local head."""
    classifier = question_classifier()
    if backend == "cpu":
        rows = await asyncio.to_thread(classifier.classify, texts)
        return ClassificationRows(rows, classifier.model_name, classifier.revision)
    if not settings.question_slice_gpu_enabled:
        raise ClassifierUnavailable(
            "Exoscale extraction is disabled by QUESTION_SLICE_GPU_ENABLED."
        )
    await asyncio.to_thread(classifier.validate_inputs, texts)
    head = question_classification_head()
    await asyncio.to_thread(head.load)
    slug = settings.question_slice_deployment
    cfg = deployments.get_config(slug)
    if cfg.model != classifier.model_name:
        raise ClassifierUnavailable(
            "Extraction deployment must use the configured QuestionSlice model"
        )
    cfg, state = await deployments.ensure_running(slug)
    try:
        rows = await classify_remote(
            texts=texts,
            model=cfg.model,
            deployment_url=state.deployment_url,
            api_key=state.api_key,
            batch_size=classifier.batch_size,
            embedding_head=head.classify,
        )
    finally:
        deployments.touch(slug)
    # Exoscale imports by repository name; the remote weight SHA is unverified.
    return ClassificationRows(rows, classifier.model_name, None)
