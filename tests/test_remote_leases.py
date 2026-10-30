"""
Remote lifetime and independent status/activity ownership without cloud calls.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import NamedTuple
from unittest.mock import Mock

import pytest

from llm_management.bootstrap import assemble_service
from llm_management.deployments.errors import DeploymentServiceError
from llm_management.deployments.service import DeploymentService
from llm_management.deployments.status import DeploymentStatusCache
from llm_management.deployments.types import DeploymentQueryResult, DeploymentSnapshot


class RemoteFixture(NamedTuple):
    """
    Real orchestration with only the cloud backend replaced.
    """

    service: DeploymentService
    backend: Mock
    slug: str


def cached_state(cache: DeploymentStatusCache, slug: str) -> DeploymentSnapshot:
    """
    Require a cached state for assertions about a configured test deployment.
    """
    state = cache.get(slug)
    assert state is not None
    return state


@pytest.fixture
def remote() -> RemoteFixture:
    service = assemble_service()
    slug = service.catalog.require_foi().topic_deployment
    backend = Mock()
    backend.config = service.get_config(slug)
    backend.query_status.return_value = DeploymentQueryResult(
        exists=True,
        replicas=1,
        deployment_url="https://inference.test/v1",
        api_key="test",
    )
    service.backends[slug] = backend
    service.status.refresh(backend)
    return RemoteFixture(service, backend, slug)


def test_nested_leases_block_management_until_last_exit(remote):
    service, backend, slug = remote

    async def run():
        async with service.use(slug) as running:
            assert running.config.slug == slug
            assert service.activity.get(slug).requests_in_flight == 1
            async with service.use(slug):
                assert service.activity.get(slug).requests_in_flight == 2
                for action in (service.cooldown, service.scale_to_zero):
                    with pytest.raises(DeploymentServiceError) as caught:
                        await action(slug)
                    assert caught.value.status_code == 409
            assert service.activity.get(slug).requests_in_flight == 1
        assert service.activity.get(slug).requests_in_flight == 0
        await service.scale_to_zero(slug)

    asyncio.run(run())
    backend.create_or_resume.assert_not_called()
    backend.scale_to_zero.assert_called_once()


def test_concurrent_lease_cancellation_does_not_release_another_user(remote):
    service, _, slug = remote

    async def run():
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def other_user():
            async with service.use(slug):
                entered.set()
                await finish.wait()

        async with service.use(slug):
            task = asyncio.create_task(other_user())
            await entered.wait()
            assert service.activity.get(slug).requests_in_flight == 2
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert service.activity.get(slug).requests_in_flight == 1
        assert service.activity.get(slug).requests_in_flight == 0

    asyncio.run(run())


def test_inference_failure_releases_lease_and_restarts_idle_timer(remote, monkeypatch):
    service, _, slug = remote
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])

    async def run():
        with pytest.raises(ValueError, match="invalid output"):
            async with service.use(slug):
                now[0] += 100
                raise ValueError("invalid output")
        activity = service.activity.get(slug)
        assert activity.requests_in_flight == 0
        assert activity.last_request_time == 1100

    asyncio.run(run())


def test_slow_status_refresh_cannot_replace_live_activity(remote):
    service, backend, slug = remote
    entered, finish = threading.Event(), threading.Event()
    status = backend.query_status.return_value

    def query():
        entered.set()
        assert finish.wait(5)
        return status

    backend.query_status.side_effect = query

    async def run():
        refresh = asyncio.create_task(
            asyncio.to_thread(service.status.refresh, backend)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            async with service.use(slug):
                before = service.activity.get(slug)
                finish.set()
                await refresh
                assert service.activity.get(slug) == before
                assert service.activity.get(slug).requests_in_flight == 1
        finally:
            finish.set()
            await refresh
        assert service.activity.get(slug).requests_in_flight == 0

    asyncio.run(run())


def test_status_eviction_preserves_live_lease_and_lifecycle_lock(remote):
    service, backend, slug = remote

    async def run():
        async with service.use(slug):
            lock = service.activity.ensure_lock(slug)
            service.status.remove(slug)
            assert service.status.get(slug) is None
            assert service.activity.get(slug).requests_in_flight == 1
            assert service.activity.ensure_lock(slug) is lock
            service.status.refresh(backend)
            with pytest.raises(DeploymentServiceError):
                await service.scale_to_zero(slug)
        assert service.activity.get(slug).requests_in_flight == 0

    asyncio.run(run())


def test_cached_views_are_detached_and_refresh_keeps_cooldown(remote, monkeypatch):
    service, backend, slug = remote
    monkeypatch.setattr(time, "time", lambda: 10000.0)
    service.activity.touch(slug)
    service.activity.cooldown(slug, 900)
    before = service.activity.get(slug)
    state = cached_state(service.status, slug)
    activity = service.activity.get(slug)
    activity.requests_in_flight = 99
    state.replicas = 99
    service.status.refresh(backend)
    assert service.activity.get(slug) == before
    assert cached_state(service.status, slug).replicas == 1
    assert service.activity.get(slug).requests_in_flight == 0


def test_cancelled_startup_finishes_and_remains_tracked(remote):
    service, backend, slug = remote
    entered, finish = threading.Event(), threading.Event()
    backend.query_status.return_value = DeploymentQueryResult(
        exists=False, replicas=0, deployment_url="", api_key=""
    )
    service.status.refresh(backend)

    def start():
        entered.set()
        assert finish.wait(5)
        backend.query_status.return_value = DeploymentQueryResult(
            exists=True,
            replicas=1,
            deployment_url="https://inference.test/v1",
            api_key="test",
        )

    backend.create_or_resume.side_effect = start

    async def run():
        async def inference():
            async with service.use(slug):
                pytest.fail("Cancelled startup must not enter inference")

        task = asyncio.create_task(inference())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish.set()
        assert service.activity.get(slug).requests_in_flight == 0
        assert service.active_deployments()[0].slug == slug

    asyncio.run(run())


def test_startup_failure_does_not_leak_lease(remote):
    service, backend, slug = remote
    backend.query_status.return_value = DeploymentQueryResult(
        exists=False, replicas=0, deployment_url="", api_key=""
    )
    service.status.refresh(backend)
    backend.create_or_resume.side_effect = RuntimeError("startup failed")

    async def run():
        with pytest.raises(RuntimeError, match="startup failed"):
            async with service.use(slug):
                pytest.fail("Failed startup must not enter inference")
        assert service.activity.get(slug).requests_in_flight == 0
        assert not service.activity.ensure_lock(slug).locked()

    asyncio.run(run())


def test_activity_is_independent_of_cache_ttl_and_other_runtimes(remote):
    service, backend, slug = remote
    other = assemble_service()
    other.backends[slug] = backend
    other.status.refresh(backend)
    service.activity.begin_request(slug)
    assert other.activity.get(slug).requests_in_flight == 0
    assert service.activity.ensure_lock(slug) is not other.activity.ensure_lock(slug)
    service.status.cache_ttl_seconds = -1
    service.status.ensure(backend)
    assert service.activity.get(slug).requests_in_flight == 1
    assert service.status.is_stale(slug)
    service.activity.end_request(slug)


def test_independent_status_replacement_preserves_activity(remote):
    from llm_management.deployments.status import DeploymentStatusCache

    service, backend, slug = remote
    tracker = service.activity
    service.activity.begin_request(slug)
    service.status = DeploymentStatusCache()
    service.status.refresh(backend)
    assert service.activity is tracker
    assert service.activity.get(slug).requests_in_flight == 1
    assert service.status.get(slug) is not None
    service.activity.end_request(slug)


def test_status_cache_returns_detached_snapshots():
    from llm_management.deployments.status import DeploymentStatusCache
    from llm_management.deployments.types import DeploymentSnapshot

    status = DeploymentStatusCache()
    original = DeploymentSnapshot(slug="test", replicas=1)
    status.set(original)
    original.replicas = 99
    snapshot = status.get("test")
    assert snapshot.replicas == 1
    assert type(snapshot) is DeploymentSnapshot
    assert "requests_in_flight" not in snapshot.model_dump()
    snapshot.replicas = 99
    snapshots = status.all()
    assert snapshots[0].replicas == 1
    snapshots[0].replicas = 99
    assert status.all()[0].replicas == 1
