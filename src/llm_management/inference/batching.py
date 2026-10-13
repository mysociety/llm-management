"""Bounded, process-local async batching around a synchronous classifier."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable

from .errors import ClassifierBusy, ClassifierOutputError, ClassifierUnavailable


class AsyncClassifierBatcher:
    """Collect requests without occupying threads; run one batch at a time.

    Caller cancellation removes queued work but does not cancel an active model
    call. The model's resource lease remains held until its worker finishes.
    Idle drainers exit, so construction creates no task or event-loop dependency.
    """

    def __init__(
        self,
        classify: Callable[[list[str]], list[list[float]]],
        *,
        batch_size: int = 8,
        delay_seconds: float = 0.01,
        max_pending: int = 64,
        timeout_seconds: float = 30,
    ) -> None:
        self.classify = classify
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
        self.max_pending = max_pending
        self.timeout_seconds = timeout_seconds
        self._pending: deque[tuple[str, asyncio.Future[list[float]]]] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._wake: asyncio.Event | None = None
        self._closed = False

    async def predict(self, text: str) -> list[float]:
        if self._closed:
            raise ClassifierUnavailable("Classifier queue is closed")
        self._pending = deque(item for item in self._pending if not item[1].done())
        if len(self._pending) >= self.max_pending:
            raise ClassifierBusy("Classifier queue is full")
        future = asyncio.get_running_loop().create_future()
        self._pending.append((text, future))
        if self._worker is None or self._worker.done():
            self._wake = asyncio.Event()
            self._worker = asyncio.create_task(self._drain())
        assert self._wake is not None
        if len(self._pending) >= self.batch_size:
            self._wake.set()
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_seconds)
        except TimeoutError as exc:
            raise ClassifierBusy("Classifier queue deadline exceeded") from exc

    def _classify(self, texts: list[str]) -> list[list[float] | Exception]:
        try:
            rows = self.classify(texts)
            if len(rows) != len(texts):
                raise ClassifierOutputError("Invalid classifier batch dimensions")
            return list(rows)
        except ValueError:
            # An overlong/invalid document must not poison its otherwise valid peers.
            results: list[list[float] | Exception] = []
            for text in texts:
                try:
                    rows = self.classify([text])
                    if len(rows) != 1:
                        raise ClassifierOutputError(
                            "Invalid classifier batch dimensions"
                        )
                    results.append(rows[0])
                except Exception as exc:
                    results.append(exc)
            return results

    async def _drain(self) -> None:
        assert self._wake is not None
        try:
            while self._pending:
                if len(self._pending) < self.batch_size:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), self.delay_seconds)
                    except TimeoutError:
                        pass
                batch = []
                while self._pending and len(batch) < self.batch_size:
                    item = self._pending.popleft()
                    if not item[1].done():
                        batch.append(item)
                if not batch:
                    continue
                try:
                    results = await asyncio.to_thread(
                        self._classify, [item[0] for item in batch]
                    )
                except Exception as exc:
                    results = [exc] * len(batch)
                for (_, future), result in zip(batch, results, strict=True):
                    if not future.done():
                        if isinstance(result, Exception):
                            future.set_exception(result)
                        else:
                            future.set_result(result)
        finally:
            # Event-loop shutdown must not leave queued callers waiting forever.
            while self._pending:
                _, future = self._pending.popleft()
                if not future.done():
                    future.set_exception(
                        ClassifierUnavailable("Classifier queue stopped")
                    )

    async def close(self) -> None:
        """Reject queued/new work and wait for any active model call to finish."""
        self._closed = True
        while self._pending:
            _, future = self._pending.popleft()
            if not future.done():
                future.set_exception(
                    ClassifierUnavailable("Classifier queue is closed")
                )
        if self._wake is not None:
            self._wake.set()
        if self._worker is not None:
            await asyncio.shield(self._worker)
