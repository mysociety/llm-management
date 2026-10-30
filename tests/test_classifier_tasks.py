"""
Admission and shutdown contracts for work retained after caller cancellation.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_management.automatic.pipeline import AutomaticClassifier
from llm_management.inference.errors import ClassifierBusy, ClassifierUnavailable
from llm_management.inference.policy import ClassifierExecutionPolicy
from llm_management.inference.tasks import ClassifierTasks
from llm_management.sar.pipeline import SARClassifier
from llm_management.settings import settings


def test_rejected_work_is_not_created_and_cancellation_keeps_its_slot():
    async def run():
        owner = ClassifierTasks[int]("Logistic")
        entered, finish = asyncio.Event(), asyncio.Event()

        async def work() -> int:
            entered.set()
            await finish.wait()
            return 7

        pending = owner.submit(work, max_pending=1)
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        rejected = Mock(side_effect=AssertionError("Rejected work must not be created"))
        with pytest.raises(ClassifierBusy, match="Logistic queue is full"):
            owner.submit(rejected, max_pending=1)
        rejected.assert_not_called()
        assert owner.pending_count == 1
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        assert not closing.done()
        finish.set()
        await closing
        assert owner.pending_count == 0

    asyncio.run(run())


def test_task_failures_propagate_and_release_admission():
    async def run():
        owner = ClassifierTasks[str]("Logistic")

        async def fail() -> str:
            raise ValueError("invalid output")

        async def succeed() -> str:
            return "accepted"

        with pytest.raises(ValueError, match="invalid output"):
            await owner.submit(fail, max_pending=1)
        assert owner.pending_count == 0
        assert await owner.submit(succeed, max_pending=1) == "accepted"
        await owner.close()

    asyncio.run(run())


def test_cancelled_shutdown_finishes_draining_before_propagating_cancellation():
    async def run():
        owner = ClassifierTasks[int]("Logistic")
        entered, finish = asyncio.Event(), asyncio.Event()

        async def work() -> int:
            entered.set()
            await finish.wait()
            return 3

        pending = owner.submit(work, max_pending=1)
        await entered.wait()
        closing = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        other_close = asyncio.create_task(owner.close())
        for _ in range(2):
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
            assert not pending.done()
        rejected = Mock()
        with pytest.raises(ClassifierUnavailable, match="Logistic queue is closed"):
            owner.submit(rejected, max_pending=1)
        rejected.assert_not_called()
        finish.set()
        assert await pending == 3
        with pytest.raises(asyncio.CancelledError):
            await closing
        await other_close
        await owner.close()
        assert owner.pending_count == 0

    asyncio.run(run())


def test_drain_waits_for_other_tasks_after_one_fails():
    async def run():
        owner = ClassifierTasks[int]("Logistic")
        finish = asyncio.Event()

        async def fail() -> int:
            raise ValueError("failed")

        async def succeed() -> int:
            await finish.wait()
            return 2

        failed = owner.submit(fail, max_pending=2)
        pending = owner.submit(succeed, max_pending=2)
        closing = asyncio.create_task(owner.close())
        with pytest.raises(ValueError):
            await failed
        assert not closing.done()
        finish.set()
        assert await pending == 2
        await closing

    asyncio.run(run())


def test_construction_is_cold_and_task_ownership_is_independent():
    first = ClassifierTasks[int]("Logistic")
    second = ClassifierTasks[int]("Logistic")
    assert first.pending_count == second.pending_count == 0

    async def work() -> int:
        return 1

    async def run():
        await first.close()
        with pytest.raises(ClassifierUnavailable):
            first.submit(work, max_pending=1)
        assert await second.submit(work, max_pending=1) == 1
        await second.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["sar", "automatic"])
def test_classifier_shutdown_stops_new_gates_and_preserves_late_positive_policy(kind):
    entered, finish = threading.Event(), threading.Event()

    def predict(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return SimpleNamespace(result=True, positive_score=0.9)

    logistic = Mock(predict=predict)
    encoder_provider = Mock(side_effect=AssertionError("Closed encoder must not load"))
    classifier = (
        SARClassifier(
            lambda: logistic,
            encoder_provider,
            policy=ClassifierExecutionPolicy.from_settings(settings),
        )
        if kind == "sar"
        else AutomaticClassifier(
            lambda: logistic,
            encoder_provider,
            max_concurrency=1,
            policy=ClassifierExecutionPolicy.from_settings(settings),
        )
    )

    async def run():
        request = asyncio.create_task(classifier.detect("positive"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(classifier.close())
            await asyncio.sleep(0)
            assert not closing.done()
            with pytest.raises(ClassifierUnavailable, match="Logistic queue is closed"):
                await classifier.detect("new", fast=True)
        finally:
            finish.set()
        result = await request
        await closing
        if kind == "sar":
            assert result.is_sar and result.status == "unclear"
            assert result.reason == "deberta_unavailable"
        else:
            assert result.unresolved
            assert result.reason == "encoder_unavailable"
        encoder_provider.assert_not_called()
        assert classifier.gate_tasks.pending_count == 0

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["sar", "automatic"])
def test_encoder_cleanup_failure_does_not_skip_gate_drain(kind, monkeypatch):
    entered, finish = threading.Event(), threading.Event()

    def predict(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return SimpleNamespace(result=False, positive_score=0.1)

    logistic = Mock(predict=predict)
    classifier = (
        SARClassifier(
            lambda: logistic,
            Mock(),
            policy=ClassifierExecutionPolicy.from_settings(settings),
        )
        if kind == "sar"
        else AutomaticClassifier(
            lambda: logistic,
            Mock(),
            max_concurrency=1,
            policy=ClassifierExecutionPolicy.from_settings(settings),
        )
    )
    monkeypatch.setattr(
        classifier.batcher,
        "close",
        AsyncMock(side_effect=RuntimeError("cleanup failed")),
    )

    async def run():
        request = asyncio.create_task(classifier.detect("negative", fast=True))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(classifier.close())
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            finish.set()
        await request
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await closing
        assert classifier.gate_tasks.pending_count == 0

    asyncio.run(run())


def test_shutdown_drains_accepted_work_waiting_for_automatic_capacity():
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def predict(body, **kwargs):
        calls.append(body)
        if body == "first":
            entered.set()
            assert finish.wait(5)
        return SimpleNamespace(result=False, positive_score=0.1)

    classifier = AutomaticClassifier(
        lambda: Mock(predict=predict),
        Mock(),
        max_concurrency=1,
        policy=ClassifierExecutionPolicy.from_settings(settings),
    )

    async def run():
        first = asyncio.create_task(classifier.detect("first", fast=True))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(classifier.detect("second", fast=True))
            await asyncio.sleep(0)
            assert classifier.gate_tasks.pending_count == 2
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            closing = asyncio.create_task(classifier.close())
            await asyncio.sleep(0)
            assert not closing.done()
            assert calls == ["first"]
        finally:
            finish.set()
        await first
        await closing
        assert calls == ["first", "second"]
        assert classifier.gate_tasks.pending_count == 0

    asyncio.run(run())
