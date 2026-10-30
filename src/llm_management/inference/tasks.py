"""
Application-owned classifier tasks that outlive cancellation of their callers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar

from .errors import ClassifierBusy, ClassifierUnavailable

Result = TypeVar("Result")


class ClassifierTasks(Generic[Result]):
    """
    Bound admission and retain accepted work until its coroutine actually finishes.

    Each classifier owns one instance, used in its application's event loop.
    Pending work includes both running tasks and tasks waiting for feature-specific
    capacity. Cancelling a returned future stops that caller waiting; it does not
    cancel the owned task or free its admission slot prematurely.

    This object owns task lifetime, not model lifetime or classification policy.
    Features supply the work and handle its result or exception. Model resource
    guards still protect weights while worker threads use them. Construction creates
    no tasks, futures or event-loop-bound synchronization objects.
    """

    def __init__(self, queue_name: str) -> None:
        """
        Name the admission queue for errors and create empty task ownership.
        """
        self.queue_name = queue_name
        self._tasks: set[asyncio.Task[Result]] = set()
        self._closed = False

    @property
    def pending_count(self) -> int:
        """
        Count accepted tasks whose completion callbacks have not released their slots.
        """
        return len(self._tasks)

    def submit(
        self,
        operation: Callable[[], Coroutine[Any, Any, Result]],
        *,
        max_pending: int,
    ) -> asyncio.Future[Result]:
        """
        Admit work synchronously and return a cancellation-shielded result future.

        Check admission before invoking the factory, so rejected work never creates
        an unawaited coroutine or resolves a model provider. The caller supplies its
        configured limit on each submission; this helper reads no global settings.
        Queue errors are raised here, separately from failures awaiting the result.
        """
        if self._closed:
            raise ClassifierUnavailable(f"{self.queue_name} queue is closed")
        if self.pending_count >= max_pending:
            raise ClassifierBusy(f"{self.queue_name} queue is full")
        task = asyncio.create_task(operation())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return asyncio.shield(task)

    def stop_accepting(self) -> None:
        """
        Reject new work immediately, before the feature begins asynchronous cleanup.

        Already accepted tasks keep running, including tasks waiting for capacity.
        Calling this repeatedly has no additional effect.
        """
        self._closed = True

    async def close(self) -> None:
        """
        Stop admission and drain accepted work without cancelling its coroutines.

        Concurrent or repeated closes can safely wait for the same tasks. If a close
        caller is cancelled, finish draining before propagating cancellation, including
        repeated cancellation requests. Individual task failures belong to the feature
        awaiting their results and do not prevent other tasks from draining.
        """
        self.stop_accepting()
        if not self._tasks:
            return
        drain = asyncio.gather(*self._tasks, return_exceptions=True)
        cancelled = False
        while True:
            try:
                await asyncio.shield(drain)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
