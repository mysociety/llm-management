"""Reusable, lazy sequence-classification backends without eager ML imports.

CPU work runs in a caller-supplied worker thread. A nonblocking per-model lock bounds
in-flight CPU work even if an HTTP caller disconnects while its thread is running.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from .errors import (
    ClassifierBusy,
    ClassifierInputTooLong,
    ClassifierUnavailable,
)
from .resources import LocalResource, ResourceRegistry

if TYPE_CHECKING:
    from transformers import BatchEncoding


class LocalSequenceClassifier:
    """
    Own a lazy tokenizer and CPU model with bounded classification work.

    Validate every input without truncation and preserve configured label order in
    probability rows. Use a resource lease for model lifetime, a loading lock for
    initialization, and a nonblocking work lock to reject overlapping classifications.
    Blocking operations belong in a worker thread.
    """

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        labels: tuple[str, ...],
        max_tokens: int = 768,
        max_units: int = 256,
        batch_size: int = 8,
        threads: int = 1,
        cache_dir: str | None = None,
        token: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        resource_name: str | None = None,
        registry: ResourceRegistry | None = None,
    ) -> None:
        """
        Validate limits and register a cold owner without importing or loading ML code.

        Use an independent guard when no registry is supplied. threads controls PyTorch's
        process-wide thread setting at load time, so owners should share one policy.
        """
        if min(max_tokens, max_units, batch_size, threads) < 1 or not labels:
            raise ValueError("Classifier limits must be positive and labels nonempty")
        self.model_name = model
        self.revision = revision
        self.labels = labels
        self.max_tokens = max_tokens
        self.max_units = max_units
        self.batch_size = batch_size
        self.threads = threads
        self.cache_dir = cache_dir
        self.token = token
        self.model_kwargs = dict(model_kwargs or {})
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._load_lock = threading.RLock()
        self._work_lock = threading.Lock()
        self.resource = (
            registry if registry is not None else ResourceRegistry()
        ).register(LocalResource(resource_name or model, self.warmup, self.unload))

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """
        Load the tokenizer lazily and reject unsupported unit counts or token lengths.

        Retain full text without truncation. The public wrappers hold a resource lease
        while using these tokenized inputs or loading the corresponding model.
        """
        if not texts or len(texts) > self.max_units:
            raise ValueError(f"Request must contain 1–{self.max_units} semantic units")
        with self._load_lock:
            if self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer
                except ImportError as exc:
                    raise ClassifierUnavailable(
                        "Run poetry install to install extraction dependencies"
                    ) from exc
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        revision=self.revision,
                        token=self.token,
                        cache_dir=self.cache_dir,
                    )
                except Exception as exc:
                    raise ClassifierUnavailable(
                        "Could not load classifier tokenizer"
                    ) from exc
            encoded = self._tokenizer(texts, padding=False, truncation=False)
        for index, ids in enumerate(encoded["input_ids"]):
            if len(ids) > self.max_tokens:
                raise ClassifierInputTooLong(
                    f"Classifier context window {index} has {len(ids)} tokens; limit is {self.max_tokens}. Input was not truncated."
                )
        return encoded

    def validate_inputs(self, texts: list[str]) -> None:
        """
        Check token and unit budgets under a lease without loading classifier weights.
        """
        with self.resource.use():
            self._tokenize(texts)

    def unload(self) -> None:
        """
        Clear model, tokenizer and runtime references after the resource becomes idle.
        """
        with self._load_lock:
            self._model = self._tokenizer = self._torch = None

    def warmup(self) -> None:
        """
        Load and validate CPU weights under the resource and initialization locks.

        Check checkpoint label mapping against configured output order and mark the guard
        ready on success. Reuse a loaded model; translate initialization failures to
        ClassifierUnavailable. Tokenizer validation alone does not mark the model ready.
        """
        with self.resource.use(), self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification
            except ImportError as exc:
                raise ClassifierUnavailable(
                    "Run poetry install to install CPU inference dependencies"
                ) from exc
            self._tokenize(["Warmup."])
            try:
                # PyTorch's thread count is process-wide; use the same policy for
                # every local classifier in this API worker.
                torch.set_num_threads(self.threads)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    revision=self.revision,
                    token=self.token,
                    cache_dir=self.cache_dir,
                    **self.model_kwargs,
                ).eval()
                if tuple(
                    model.config.id2label.get(i) for i in range(len(self.labels))
                ) != self.labels or model.config.num_labels != len(self.labels):
                    raise ClassifierUnavailable(
                        "Checkpoint label mapping does not match the configured classifier"
                    )
                self._torch = torch
                self._model = model
                self.resource.mark_ready()
            except ClassifierUnavailable:
                raise
            except Exception as exc:
                raise ClassifierUnavailable(
                    "Could not load CPU classifier model"
                ) from exc

    def classify(self, texts: list[str]) -> list[list[float]]:
        """
        Return ordered probability rows while holding admission and resource leases.

        Raise ClassifierBusy immediately when another classification owns the work lock.
        Keep the guard leased through tokenization, loading and batched inference, so
        cancelling an async caller cannot unload weights still used by its worker.
        """
        if not self._work_lock.acquire(blocking=False):
            raise ClassifierBusy("CPU classifier is busy; retry the request")
        try:
            with self.resource.use():
                return self._classify(texts)
        finally:
            self._work_lock.release()

    def _classify(self, texts: list[str]) -> list[list[float]]:
        """
        Execute full-input batched CPU inference within the public classifier's locks.
        """
        encoded = self._tokenize(texts)
        self.warmup()
        rows = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = {
                    key: value[start : start + self.batch_size]
                    for key, value in encoded.items()
                }
                with self._load_lock:
                    batch = self._tokenizer.pad(
                        batch, padding=True, return_tensors="pt"
                    )
                rows.extend(self._model(**batch).logits.softmax(-1).tolist())
        return rows


class AuthenticatedSequenceClassifier(LocalSequenceClassifier):
    """
    Load private pinned tokenizers locally with v4/v5 artifact compatibility.
    """

    def _tokenize(self, texts: list[str]) -> BatchEncoding:
        """
        Fetch private pinned tokenizer files explicitly, then load from the local cache.

        Adapt serialized special-token metadata for supported Transformers versions.
        Delegate unit and token-budget checks to the common tokenizer path afterward.
        """
        with self._load_lock:
            if self._tokenizer is None:
                try:
                    from huggingface_hub import snapshot_download
                    from transformers import AutoTokenizer

                    # Like the Granite tokenizer, fetch with explicit authentication
                    # then load locally to avoid unauthenticated v4 metadata probes.
                    snapshot = snapshot_download(
                        self.model_name,
                        revision=self.revision,
                        token=self.token,
                        cache_dir=self.cache_dir,
                        allow_patterns=[
                            "config.json",
                            "tokenizer*",
                            "special_tokens_map.json",
                        ],
                    )
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        snapshot,
                        local_files_only=True,
                        fix_mistral_regex=False,
                        # v5 serialized a list; v4 expects a mapping. PAD/CLS/SEP
                        # already have their standard named roles in this artifact.
                        extra_special_tokens={},
                    )
                except Exception as exc:
                    raise ClassifierUnavailable(
                        "Could not load classifier tokenizer"
                    ) from exc
        return super()._tokenize(texts)
