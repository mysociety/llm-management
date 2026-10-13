"""Queue safety and real cascade integration without model downloads."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_management.automatic.pipeline import AutomaticClassifier
from llm_management.inference.batching import AsyncClassifierBatcher
from llm_management.inference.errors import (
    ClassifierBusy,
    ClassifierInputTooLong,
    ClassifierUnavailable,
)
from llm_management.sar.pipeline import SARClassifier


def test_eight_requests_flush_immediately_and_preserve_order():
    calls = []

    def classify(texts):
        calls.append(texts)
        return [[float(text)] for text in texts]

    async def run():
        batcher = AsyncClassifierBatcher(classify, delay_seconds=1)
        results = await asyncio.wait_for(
            asyncio.gather(*(batcher.predict(str(i)) for i in range(8))), 0.5
        )
        assert results == [[float(i)] for i in range(8)]
        assert calls == [[str(i) for i in range(8)]]
        await batcher.close()

    asyncio.run(run())


def test_single_request_flushes_without_waiting_for_full_batch():
    async def run():
        batcher = AsyncClassifierBatcher(lambda texts: [[1.0] for _ in texts])
        assert await asyncio.wait_for(batcher.predict("single"), 1) == [1.0]
        await batcher.close()

    asyncio.run(run())


def test_invalid_input_does_not_poison_other_members():
    def classify(texts):
        if "bad" in texts:
            raise ClassifierInputTooLong("too long")
        return [[0.1, 0.9] for _ in texts]

    async def run():
        batcher = AsyncClassifierBatcher(classify, batch_size=3)
        results = await asyncio.gather(
            *(batcher.predict(text) for text in ("good", "bad", "good2")),
            return_exceptions=True,
        )
        assert results[0] == results[2] == [0.1, 0.9]
        assert isinstance(results[1], ClassifierInputTooLong)
        await batcher.close()

    asyncio.run(run())


def test_cancellation_and_queue_limit_do_not_overlap_active_batches():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def classify(texts):
        calls.append(texts)
        if texts == ["active"]:
            entered.set()
            assert release.wait(5)
        return [[1.0] for _ in texts]

    async def run():
        batcher = AsyncClassifierBatcher(classify, batch_size=1, max_pending=1)
        active = asyncio.create_task(batcher.predict("active"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            cancelled = asyncio.create_task(batcher.predict("cancelled"))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            queued = asyncio.create_task(batcher.predict("queued"))
            await asyncio.sleep(0)
            with pytest.raises(ClassifierBusy, match="full"):
                await batcher.predict("overflow")
            assert calls == [["active"]]
        finally:
            release.set()
        assert await queued == [1.0]
        await batcher.close()
        assert calls == [["active"], ["queued"]]

    asyncio.run(run())


def test_timeout_does_not_release_active_worker():
    entered, release = threading.Event(), threading.Event()

    def classify(texts):
        entered.set()
        assert release.wait(5)
        return [[1.0] for _ in texts]

    async def run():
        batcher = AsyncClassifierBatcher(classify, batch_size=1, timeout_seconds=0.05)
        task = asyncio.create_task(batcher.predict("active"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(ClassifierBusy, match="deadline"):
                await task
            closing = asyncio.create_task(batcher.close())
            await asyncio.sleep(0.01)
            assert not closing.done()
        finally:
            release.set()
        await closing
        with pytest.raises(ClassifierUnavailable):
            await batcher.predict("closed")

    asyncio.run(run())


def test_shutdown_rejects_pending_and_waits_for_active_work():
    entered, release = threading.Event(), threading.Event()

    def classify(texts):
        entered.set()
        assert release.wait(5)
        return [[1.0] for _ in texts]

    async def run():
        batcher = AsyncClassifierBatcher(classify, batch_size=1)
        task = asyncio.create_task(batcher.predict("active"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            pending = asyncio.create_task(batcher.predict("pending"))
            await asyncio.sleep(0)
            closing = asyncio.create_task(batcher.close())
            with pytest.raises(ClassifierUnavailable):
                await pending
            assert not closing.done()
        finally:
            release.set()
        assert await task == [1.0]
        await closing

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["sar", "automatic"])
def test_cascades_batch_positives_and_bypass_fast_and_negative(kind):
    logistic = Mock()
    logistic.predict.side_effect = lambda text, **kwargs: SimpleNamespace(
        result=text != "negative", positive_score=0.1
    )
    entered, release = threading.Event(), threading.Event()
    calls = []

    def classify(texts):
        calls.append(texts)
        entered.set()
        assert release.wait(5)
        return [[0.1, 0.9] for _ in texts]

    encoder = SimpleNamespace(classify=classify)
    owner = (
        SARClassifier(lambda: logistic, lambda: encoder)
        if kind == "sar"
        else AutomaticClassifier(lambda: logistic, lambda: encoder, max_concurrency=4)
    )
    # Give logistic worker threads time to finish all eight gates deterministically.
    owner.batcher.delay_seconds = 1

    async def run():
        tasks = [asyncio.create_task(owner.detect(f"positive {i}")) for i in range(8)]
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            fast = await asyncio.wait_for(owner.detect("fast", fast=True), 1)
            negative = await asyncio.wait_for(owner.detect("negative"), 1)
            assert fast.stage == negative.stage == "logistic"
            assert len(calls) == 1 and len(calls[0]) == 8
        finally:
            release.set()
        results = await asyncio.gather(*tasks)
        assert all(result.stage == "deberta" for result in results)
        assert all(
            result.status == "complete" if kind == "sar" else not result.unresolved
            for result in results
        )
        await owner.batcher.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["automatic", "sar"])
def test_logistic_admission_survives_caller_cancellation(monkeypatch, kind):
    from llm_management.settings import settings

    monkeypatch.setattr(settings, "logistic_max_pending", 1)
    started, release = threading.Event(), threading.Event()

    def gate(*args, **kwargs):
        started.set()
        assert release.wait(5)
        if kind == "automatic":
            from llm_management.automatic.schemas import AutomaticResult

            return AutomaticResult(
                is_automatic=False, stage="logistic", logistic_score=0.0
            )
        from llm_management.sar.schemas import SARResult

        return SARResult(
            is_sar=False, stage="logistic", status="complete", logistic_score=0.0
        )

    async def run():
        if kind == "automatic":
            classifier = AutomaticClassifier(
                lambda: None, lambda: None, max_concurrency=1
            )
            monkeypatch.setattr(classifier, "classify_gate", gate)
        else:
            classifier = SARClassifier(lambda: None, lambda: None)
            monkeypatch.setattr("llm_management.sar.pipeline._detect_sar", gate)
        request = asyncio.create_task(classifier.detect("test", fast=True))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            with pytest.raises(ClassifierBusy):
                await classifier.detect("second", fast=True)
        finally:
            release.set()
            await classifier.close()
        assert len(classifier._gates) == 0

    asyncio.run(run())
