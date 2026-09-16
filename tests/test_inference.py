import asyncio
import threading

import httpx
import pytest

from llm_management import inference
from llm_management.inference import (
    ClassifierBusy,
    ClassifierOutputError,
    LocalSequenceClassifier,
)


def test_cpu_work_is_bounded_and_lock_released_on_error(monkeypatch):
    model = LocalSequenceClassifier(
        model="unused", revision="unused", labels=("A", "B")
    )
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def block(texts):
        entered.set()
        assert release.wait(5)
        raise ValueError("invalid input")

    monkeypatch.setattr(model, "_tokenize", block)

    def run():
        try:
            model.classify(["first"])
        except ValueError as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(ClassifierBusy):
            model.classify(["second"])
    finally:
        release.set()
        worker.join(5)
    assert errors
    with pytest.raises(ValueError, match="invalid input"):
        model.classify(["third"])


def test_remote_reorders_results_and_rejects_duplicate_indices(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        data = [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
        if len(calls) > 1:
            data[0]["index"] = 0
        return httpx.Response(200, json={"data": data})

    original = httpx.AsyncClient
    monkeypatch.setattr(
        inference.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )
    kwargs = dict(
        embedding_head=lambda vectors: vectors,
        texts=["a", "b"],
        model="test",
        deployment_url="https://example.test/v1/",
        api_key="secret",
    )
    assert asyncio.run(inference.classify_remote(**kwargs)) == [[1.0, 0.0], [0.0, 1.0]]
    assert str(calls[0].url) == "https://example.test/v1/embeddings"
    assert calls[0].headers["Authorization"] == "Bearer secret"
    with pytest.raises(ClassifierOutputError):
        asyncio.run(inference.classify_remote(**kwargs))


def test_remote_embeddings_preserve_raw_vectors_for_head(monkeypatch):
    import json

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [2.0, -3.0]}]}
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        inference.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )

    def head(vectors):
        assert vectors == [[2.0, -3.0]]  # Do not normalise trained encoder features.
        return [[0.1, 0.9]]

    output = asyncio.run(
        inference.classify_remote(
            texts=["request"],
            model="test",
            deployment_url="https://example.test/v1",
            api_key="secret",
            embedding_head=head,
        )
    )
    assert output == [[0.1, 0.9]]
    assert str(calls[0].url) == "https://example.test/v1/embeddings"
    assert json.loads(calls[0].content)["encoding_format"] == "float"
