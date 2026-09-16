"""Reusable, lazy sequence-classification backends without eager ML imports.

CPU work runs in a caller-supplied worker thread. A nonblocking per-model lock bounds
in-flight CPU work even if an HTTP caller disconnects while its thread is running.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from collections.abc import Callable

import httpx

from .errors import ClassifierUnavailable, ClassifierBusy, ClassifierOutputError


class LocalSequenceClassifier:
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
    ):
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
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_lock = threading.RLock()
        self._work_lock = threading.Lock()

    def _tokenize(self, texts: list[str]):
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
                raise ValueError(
                    f"Classifier context window {index} has {len(ids)} tokens; limit is {self.max_tokens}. Input was not truncated."
                )
        return encoded

    def validate_inputs(self, texts: list[str]) -> None:
        self._tokenize(texts)

    def warmup(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification
            except ImportError as exc:
                raise ClassifierUnavailable(
                    "Run poetry install to install CPU inference dependencies"
                ) from exc
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
            except ClassifierUnavailable:
                raise
            except Exception as exc:
                raise ClassifierUnavailable(
                    "Could not load CPU classifier model"
                ) from exc

    def classify(self, texts: list[str]) -> list[list[float]]:
        if not self._work_lock.acquire(blocking=False):
            raise ClassifierBusy("CPU classifier is busy; retry the request")
        try:
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
        finally:
            self._work_lock.release()


async def classify_remote(
    *,
    texts: list[str],
    model: str,
    deployment_url: str,
    api_key: str,
    batch_size: int = 8,
    embedding_head: Callable[[list[list[float]]], list[list[float]]],
) -> list[list[float]]:
    base = deployment_url.rstrip("/")
    url = base + "/embeddings"
    rows = []
    async with httpx.AsyncClient(
        timeout=120, headers={"Authorization": f"Bearer {api_key}"}
    ) as client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            payload = {"model": model, "input": batch, "encoding_format": "float"}
            response = await client.post(url, json=payload)
            response.raise_for_status()
            try:
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise ClassifierOutputError(
                        "Remote classifier returned missing or duplicate indices"
                    )
                vectors = [[float(p) for p in item["embedding"]] for item in data]
                rows.extend(await asyncio.to_thread(embedding_head, vectors))
            except (KeyError, TypeError, ValueError) as exc:
                raise ClassifierOutputError(
                    "Remote classifier returned an invalid response"
                ) from exc
    return rows
