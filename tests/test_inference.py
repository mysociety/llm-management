import asyncio
import threading

import httpx2
import pytest

from llm_management.inference import remote as inference
from llm_management.inference.classifiers import LocalSequenceClassifier
from llm_management.inference.errors import ClassifierBusy, ClassifierOutputError


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


def test_remote_reorders_results_and_rejects_duplicate_indices(sanitizer, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        data = [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
        if len(calls) > 1:
            data[0]["index"] = 0
        return httpx2.Response(200, json={"data": data})

    original = httpx2.AsyncClient
    monkeypatch.setattr(
        inference.httpx2,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx2.MockTransport(handler), **kwargs),
    )
    kwargs = dict(
        embedding_head=lambda vectors: vectors,
        texts=sanitizer.sanitize_texts(["a", "b"]),
        model="test",
        deployment_url="https://example.test/v1/",
        api_key="secret",
    )
    assert asyncio.run(inference.classify_remote(**kwargs)) == [[1.0, 0.0], [0.0, 1.0]]
    assert str(calls[0].url) == "https://example.test/v1/embeddings"
    assert calls[0].headers["Authorization"] == "Bearer secret"
    with pytest.raises(ClassifierOutputError):
        asyncio.run(inference.classify_remote(**kwargs))


def test_remote_embeddings_preserve_raw_vectors_for_head(sanitizer, monkeypatch):
    import json

    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(
            200, json={"data": [{"index": 0, "embedding": [2.0, -3.0]}]}
        )

    original = httpx2.AsyncClient
    monkeypatch.setattr(
        inference.httpx2,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx2.MockTransport(handler), **kwargs),
    )

    def head(vectors):
        assert vectors == [[2.0, -3.0]]  # Do not normalise trained encoder features.
        return [[0.1, 0.9]]

    output = asyncio.run(
        inference.classify_remote(
            texts=sanitizer.sanitize_texts(["request"]),
            model="test",
            deployment_url="https://example.test/v1",
            api_key="secret",
            embedding_head=head,
        )
    )
    assert output == [[0.1, 0.9]]
    assert str(calls[0].url) == "https://example.test/v1/embeddings"
    assert json.loads(calls[0].content)["encoding_format"] == "float"


@pytest.mark.parametrize(
    ("status", "body", "attempts"),
    [
        (
            404,
            {"error": {"message": "deployment not found", "type": "not_found_error"}},
            6,
        ),
        (404, {"error": {"message": "model not found", "type": "not_found_error"}}, 1),
        (404, {"error": "deployment not found"}, 1),
        (404, [], 1),
        (404, "deployment not found", 1),
        (401, {"error": {"message": "unauthorized"}}, 1),
        (422, {"error": {"message": "invalid input"}}, 1),
        (503, {"error": {"message": "unavailable"}}, 1),
    ],
)
def test_encoder_retry_is_bounded_and_specific(
    monkeypatch, caplog, status, body, attempts
):
    from unittest.mock import AsyncMock

    calls = []
    sleep = AsyncMock()
    monkeypatch.setattr(inference.asyncio, "sleep", sleep)

    def respond(request):
        calls.append(request)
        if isinstance(body, str):
            return httpx2.Response(status, text=body)
        return httpx2.Response(status, json=body)

    async def run():
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            with pytest.raises(httpx2.HTTPStatusError) as error:
                await inference.fetch_embeddings(
                    client,
                    url="https://example.test/v1/embeddings",
                    model="test",
                    texts=["private correspondence"],
                )
            assert error.value.response.status_code == status

    asyncio.run(run())
    assert len(calls) == attempts
    assert [call.args[0] for call in sleep.await_args_list] == (
        [1, 2, 4, 8, 15] if attempts == 6 else []
    )
    assert f"HTTP {status}" in caplog.text
    assert "private correspondence" not in caplog.text


def test_encoder_retry_preserves_batch_and_applies_head_once(sanitizer, monkeypatch):
    from unittest.mock import AsyncMock, Mock

    calls = []
    sleep = AsyncMock()
    monkeypatch.setattr(inference.asyncio, "sleep", sleep)

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx2.Response(
                404,
                json={
                    "error": {
                        "message": "deployment not found",
                        "type": "not_found_error",
                    }
                },
            )
        return httpx2.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    original = httpx2.AsyncClient
    monkeypatch.setattr(
        inference.httpx2,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx2.MockTransport(respond), **kwargs),
    )
    head = Mock(return_value=[[0.1, 0.9]])
    result = asyncio.run(
        inference.classify_remote(
            texts=sanitizer.sanitize_texts(["request"]),
            model="test",
            deployment_url="https://example.test/v1",
            api_key="secret",
            embedding_head=head,
        )
    )
    assert result == [[0.1, 0.9]]
    assert calls[0].content == calls[1].content
    head.assert_called_once_with([[1.0]])
    sleep.assert_awaited_once_with(1)


def test_encoder_retry_does_not_swallow_cancellation(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        inference.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)
    )
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(
            404,
            json={
                "error": {"message": "deployment not found", "type": "not_found_error"}
            },
        )

    async def run():
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            with pytest.raises(asyncio.CancelledError):
                await inference.fetch_embeddings(
                    client,
                    url="https://example.test/v1/embeddings",
                    model="test",
                    texts=["request"],
                )

    asyncio.run(run())
    assert len(calls) == 1
