"""
Bound crawler state without losing active locks, pacing or cooldowns.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx2
import pytest
from protego import Protego
from pydantic import HttpUrl

from llm_management.web import crawl
from llm_management.web.crawl import (
    HOST_IDLE_SECONDS,
    CrawlBudgetExceeded,
    CrawlPolicy,
    CrawlRestricted,
    RobotsRules,
)
from llm_management.web.pages import PageReader
from tests.web_helpers import Clock


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def policy(clock: Clock) -> CrawlPolicy:
    return CrawlPolicy(clock=clock, sleep=clock.sleep)


def test_robots_cache_evicts_least_recently_used_responses(
    policy: CrawlPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(crawl, "ROBOTS_CACHE_SIZE", 2)
    downloads: dict[str, int] = {}

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            host = request.url.host
            downloads[host] = downloads.get(host, 0) + 1
            return httpx2.Response(404)
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            reader = PageReader(client, policy=policy)
            for host in [
                "a.example",
                "b.example",
                "a.example",
                "c.example",
                "a.example",
                "b.example",
            ]:
                assert (
                    await reader.fetch(HttpUrl(f"https://{host}/contact"))
                ).error is None
                assert len(policy.robots) <= 2
                assert not policy.robots_locks
        assert downloads == {"a.example": 1, "b.example": 2, "c.example": 1}

    asyncio.run(run())


def test_later_activity_prunes_expired_rules_and_idle_hosts(
    policy: CrawlPolicy, clock: Clock
) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404, headers={"Cache-Control": "max-age=10"})
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            reader = PageReader(client, policy=policy)
            assert (
                await reader.fetch(HttpUrl("https://old.example/contact"))
            ).error is None
            clock.now += HOST_IDLE_SECONDS + 60
            assert (
                await reader.fetch(HttpUrl("https://new.example/contact"))
            ).error is None
        assert "old.example" not in policy.hosts
        assert all(key.origin != "https://old.example" for key in policy.robots)
        assert not policy.robots_locks

    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 429])
def test_cleanup_preserves_pacing_and_cooldowns_until_they_expire(
    policy: CrawlPolicy, clock: Clock, status: int
) -> None:
    rules = RobotsRules(
        Protego.parse("User-agent: *\nCrawl-delay: 600" if status == 200 else ""),
        expires_at=3600,
    )
    requests = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        return httpx2.Response(status, headers={"Retry-After": "600"})

    async def fetch(client: httpx2.AsyncClient) -> None:
        async with policy.request(
            client, httpx2.URL("https://council.example/"), None, rules=rules
        ):
            pass

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            if status == 429:
                with pytest.raises(CrawlRestricted, match="HTTP 429"):
                    await fetch(client)
            else:
                await fetch(client)
            clock.now = HOST_IDLE_SECONDS + 60
            policy.cleanup()
            assert "council.example" in policy.hosts
            with pytest.raises(CrawlBudgetExceeded):
                await fetch(client)
            assert requests == 1
            clock.now = 1000
            policy.cleanup()
            assert not policy.hosts

    asyncio.run(run())


def test_cleanup_preserves_active_and_queued_host_requests(
    policy: CrawlPolicy, clock: Clock
) -> None:
    paths: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(200)

    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:

            async def queued_request() -> None:
                async with policy.request(
                    client, httpx2.URL("https://council.example/queued"), None
                ):
                    started.set()
                    await release.wait()

            async with policy.request(
                client, httpx2.URL("https://council.example/held"), None
            ):
                state = policy.hosts["council.example"]
                queued = asyncio.create_task(queued_request())
                await asyncio.sleep(0)
                clock.now += HOST_IDLE_SECONDS + 60
                policy.cleanup()
                assert policy.hosts["council.example"] is state
                assert state.lock.users == 2
            # The first holder has released the lock; the queued caller has not
            # resumed yet. Cleanup must not create a second lock in this gap.
            assert not state.lock.lock.locked()
            clock.now += HOST_IDLE_SECONDS + 60
            policy.cleanup()
            assert policy.hosts["council.example"] is state
            assert state.lock.users == 1
            await asyncio.wait_for(started.wait(), 1)
            release.set()
            await queued
        assert paths == ["/held", "/queued"]
        clock.now += HOST_IDLE_SECONDS + 60
        policy.cleanup()
        assert not policy.hosts

    asyncio.run(run())


def test_cancelled_robots_discovery_releases_holders_and_waiters(
    policy: CrawlPolicy, clock: Clock
) -> None:
    async def run() -> None:
        started = asyncio.Event()
        requests = 0

        class HeldRobots(httpx2.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await asyncio.Event().wait()
                yield b""

        def respond(request: httpx2.Request) -> httpx2.Response:
            nonlocal requests
            if request.url.path == "/robots.txt":
                requests += 1
                if requests == 1:
                    return httpx2.Response(200, stream=HeldRobots())
                return httpx2.Response(404)
            return httpx2.Response(
                200, text="contact", headers={"Content-Type": "text/plain"}
            )

        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            reader = PageReader(client, policy=policy)
            holder = asyncio.create_task(
                reader.fetch(HttpUrl("https://council.example/first"))
            )
            await asyncio.wait_for(started.wait(), 1)
            waiter = asyncio.create_task(
                reader.fetch(HttpUrl("https://council.example/second"))
            )
            await asyncio.sleep(0)
            lock = next(iter(policy.robots_locks.values()))
            assert lock.users == 2
            clock.now += HOST_IDLE_SECONDS + 60
            policy.cleanup()
            assert next(iter(policy.robots_locks.values())) is lock
            for task in [waiter, holder]:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert not policy.robots_locks
            assert policy.hosts["council.example"].lock.users == 0
            clock.now += HOST_IDLE_SECONDS + 60
            policy.cleanup()
            assert not policy.hosts
            assert (
                await reader.fetch(HttpUrl("https://council.example/third"))
            ).error is None
            assert requests == 2

    asyncio.run(run())
