"""Coordinate loading, use, and idle unloading of local CPU resources.

This module supplies synchronous lifecycle primitives, independent of HTTP or an
async event loop. Resource owners retain their model/tokenizer objects and provide
callbacks to load or clear them. The registry groups those owners so the server can
periodically expire idle resources and attempt cleanup on shutdown.

Run loading and inference in a worker thread, and hold ``LocalResource.use()`` for
the entire operation. Cancelling an async caller does not end that worker's lease,
so idle cleanup cannot unload a resource while its inference is still running.

There is no background timer here: callers must invoke ``ResourceRegistry.expire``.
Each process has its own registry and loaded objects. Unloading clears the owner's
references; Python and native allocators may retain memory afterward.
"""

from collections.abc import Callable
from contextlib import contextmanager
import logging
import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


class LocalResource(Generic[T]):
    """Protect one named resource and track its readiness and recent activity.

    ``T`` is the value returned by the owner's load callback, such as a model or
    tokenizer. The owner controls caching: ``warmup()`` calls ``load`` every time,
    so that callback should reuse an already loaded instance. ``unload`` should
    clear all owned model/tokenizer references and tolerate repeated cleanup.
    This wrapper itself does not retain the value returned by ``load``.

    A reentrant lock serializes loading and work for this resource. Nested leases
    on the same thread are supported, which lets a load callback or inference
    adapter use the same lifecycle guard. Different resources have separate locks.
    Idle release never waits for that lock; it skips busy resources for a later
    sweep. All code accessing the owner's objects must follow this lease contract.

    Typical use, with callbacks that cache and clear the owner's model::

        resource = LocalResource("classifier", load_model, clear_model)
        with resource.use():
            model = resource.warmup()
            result = model.predict(text)

    Keep the outer lease until inference finishes. ``warmup()`` alone protects
    loading, but does not reserve the returned object for subsequent work. These
    synchronous operations belong in a worker thread when called from async code.
    """

    def __init__(self, name: str, load: Callable[[], T], unload: Callable[[], None]):
        """Create a cold lifecycle guard without calling either callback.

        Args:
            name: Identifier used for registry lookup and status reporting.
            load: Owner's synchronous, repeatable loader; returns a value of type
                ``T`` and propagates failures to the caller.
            unload: Owner's synchronous cleanup callback. It must release its
                references without relying on this wrapper to store the model.
        """
        self.name = name
        self._load = load
        self._unload = unload
        self._lock = threading.RLock()
        self._ready = False
        self._last_used: float | None = None
        self._active = 0

    @contextmanager
    def use(self):
        """Lease the resource for the complete duration of synchronous work.

        Waits for other threads using this resource, then prevents unloading
        until the context exits. This does not load the resource or mark it ready.
        Nested contexts are counted individually; the count is lease depth, not
        the number of independent requests.

        Every exit, including an exception, records activity using a monotonic
        clock. Idle expiry therefore measures from the end of work. Keep this
        context in the worker thread that actually performs inference, so caller
        cancellation cannot release it prematurely.
        """
        # The worker thread retains this lease even if its async caller cancels.
        with self._lock:
            self._active += 1
            try:
                yield
            finally:
                self._active -= 1
                self._last_used = time.monotonic()

    def warmup(self):
        """Call the loader under a lease, mark success, and return its value.

        Explicit warm-up refreshes the idle timer even if the callback reuses a
        cached object. Loader exceptions propagate, and this method does not mark
        a failed load ready or automatically roll back partially loaded objects.
        Activity is still recorded when the lease exits after a failure.
        """
        with self.use():
            result = self._load()
            self._ready = True
            return result

    def mark_ready(self):
        """Record successful loading performed directly by the resource owner.

        Call this while holding ``use()`` when an existing adapter loads its own
        objects instead of going through ``warmup()``. It only updates readiness;
        it does not acquire a lease or update the idle timer on its own.
        """
        self._ready = True

    def release(self, idle_seconds: float = 0) -> bool:
        """Attempt nonblocking cleanup once the resource is idle long enough.

        Args:
            idle_seconds: Minimum elapsed seconds since the last lease ended.
                Zero requests immediate cleanup of an inactive resource.

        Returns:
            True when the unload callback completed and lifecycle state was
            reset. False when another thread holds the lock, a nested lease is
            active, the resource has no recorded use, or its timeout has not
            elapsed. A later sweep can retry a skipped resource.

        Cleanup also applies to failed or partial loads with recorded activity;
        readiness is not a prerequisite. Unload exceptions propagate without
        resetting readiness or activity, and the lock is always released. Dropping
        references does not guarantee immediate memory reclamation by the OS.
        """
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._active or self._last_used is None:
                return False
            if time.monotonic() - self._last_used < idle_seconds:
                return False
            self._unload()
            self._ready = False
            self._last_used = None
            return True
        finally:
            self._lock.release()

    def status(self) -> dict:
        """Return a lightweight, approximate status snapshot without waiting.

        ``ready`` records successful loading, rather than a live health probe.
        ``busy`` means at least one lease is active. ``idle_seconds`` measures
        from the last completed lease, or is None before use and after cleanup.
        While busy, that elapsed value is not eligibility for idle unloading.

        Reads intentionally avoid the work lock so status remains available
        during slow loading or inference. Fields can reflect slightly different
        moments during concurrent work; use ``release()`` for cleanup decisions.
        """
        # Avoid waiting for a long-running inference merely to report status.
        return {
            "name": self.name,
            "ready": self._ready,
            "busy": bool(self._active),
            "idle_seconds": (
                max(0, time.monotonic() - self._last_used)
                if self._last_used is not None
                else None
            ),
        }


class ResourceRegistry:
    """Group named lifecycle guards within a single process.

    The registry lock protects membership only. Sweeps take a snapshot and invoke
    each resource outside that lock, leaving its own lease lock to protect work.
    Registering a guard does not load it, and the registry schedules no timers.
    """

    def __init__(self):
        self._resources: dict[str, LocalResource] = {}
        self._factories: dict[str, Callable[[], LocalResource]] = {}
        self._lock = threading.RLock()

    def register(self, resource: LocalResource):
        """Store a guard by name and return it for convenient owner setup.

        An existing entry with the same name is replaced without unloading it.
        Owners should therefore register stable, unique names and reuse their
        guards; replacing a live entry removes it from subsequent cleanup sweeps.
        """
        with self._lock:
            self._resources[resource.name] = resource
        return resource

    def register_factory(self, name: str, factory: Callable[[], LocalResource]):
        """Register a cold resource without constructing its owner or loading it.

        The factory must return a guard with the registered name. It runs at most
        once after successful lookup; failures propagate and can be retried. A
        reentrant registry lock permits the owner to register its guard during
        construction. Factories should only construct controllers; model loading
        belongs in the guard's load callback.
        """
        with self._lock:
            self._factories[name] = factory

    def get(self, name: str) -> LocalResource | None:
        """Resolve a registered name, constructing a cold controller if needed.

        Returns None for unknown names. Construction is serialized so concurrent
        lookups reuse one guard, without invoking its model-loading callback.
        """
        with self._lock:
            resource = self._resources.get(name)
            if resource is None and name in self._factories:
                resource = self._factories[name]()
                if resource.name != name:
                    raise ValueError("Local resource factory returned a different name")
                self._resources[name] = resource
            return resource

    def names(self) -> set[str]:
        """Return known names, including cold factories, without constructing them."""
        with self._lock:
            return self._resources.keys() | self._factories.keys()

    def all(self):
        """Return a membership snapshot containing the registered guard objects."""
        with self._lock:
            return list(self._resources.values())

    def expire(self, idle_seconds: float):
        """Attempt idle release for each guard in the current snapshot.

        Busy resources are skipped. A callback exception propagates and stops
        this sweep; the hosting application decides how to log it and retry.
        """
        for resource in self.all():
            resource.release(idle_seconds)

    def close(self):
        """Attempt immediate release of inactive resources during shutdown.

        This neither waits for active work nor unregisters guards. Busy resources
        remain loaded. Callback failures are logged and cleanup continues for
        the remaining resources.
        The method is a best-effort cleanup pass, not a permanent closed state.
        """
        for resource in self.all():
            try:
                resource.release()
            except Exception:
                logger.exception("Failed to release local resource %s", resource.name)


local_resources = ResourceRegistry()


def register_builtin_resources():
    """Import resource owners so their registrations exist for config validation.

    Owners register their own guards or lightweight factories at module import.
    No model is loaded here. This also supports CLI configuration loading before
    the HTTP server has imported its inference components.
    """
    from importlib import import_module

    for module in ("foi.backends", "foi.question_extractor"):
        import_module(f"llm_management.{module}")
