"""
Crawler courtesy across concurrent lookups, robots discovery and redirects.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import NamedTuple

import httpx2
import pytest
from pydantic import HttpUrl

from llm_management.web.crawl import ROBOT_TOKEN, USER_AGENT, CrawlPolicy
from llm_management.web.pages import PageReader
from tests.web_helpers import Clock


class Visit(NamedTuple):
    """
    Keep request observations named and independent of the underlying HTTP client.
    """

    host: str
    path: str
    started_at: float


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def policy(clock: Clock) -> CrawlPolicy:
    return CrawlPolicy(clock=clock, sleep=clock.sleep, wall_clock=lambda: 1_800_000_000)


def test_robots_rules_apply_before_requests_and_are_shared(
    policy: CrawlPolicy,
    clock: Clock,
) -> None:
    visits: list[Visit] = []
    robots = f"""
User-agent: {ROBOT_TOKEN}
Disallow: /private/
User-agent: {ROBOT_TOKEN.upper()}
Allow: /private/contact$
"""

    def respond(request: httpx2.Request) -> httpx2.Response:
        visits.append(Visit(request.url.host, request.url.path, clock()))
        assert request.headers["User-Agent"] == USER_AGENT
        assert "https://www.mysociety.org/" in USER_AGENT
        return httpx2.Response(
            200,
            text=robots if request.url.path == "/robots.txt" else "contact",
            headers={"Content-Type": "text/plain"},
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            first, second = (
                PageReader(client, policy=policy),
                PageReader(client, policy=policy),
            )
            blocked = await first.fetch(
                HttpUrl("https://council.example/private/other")
            )
            assert blocked.error == "Blocked by robots.txt"
            allowed = await second.fetch(
                HttpUrl("https://council.example/private/contact")
            )
            assert allowed.error is None
        assert [visit.path for visit in visits] == ["/robots.txt", "/private/contact"]
        assert visits[1].started_at - visits[0].started_at >= 2

    asyncio.run(run())


def test_same_host_serializes_full_responses_across_readers(
    policy: CrawlPolicy,
    clock: Clock,
) -> None:
    active = 0
    peak = 0
    visits: list[Visit] = []

    class PageBody(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            await asyncio.sleep(0)
            yield b"contact"

        async def aclose(self) -> None:
            nonlocal active
            active -= 1

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal active, peak
        visits.append(Visit(request.url.host, request.url.path, clock()))
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        active += 1
        peak = max(peak, active)
        return httpx2.Response(
            200, stream=PageBody(), headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            readers = [PageReader(client, policy=policy) for _ in range(2)]
            results = await asyncio.gather(
                *[
                    reader.fetch(HttpUrl(f"https://council.example/page{i}"))
                    for i, reader in enumerate(readers)
                ]
            )
            assert all(result.error is None for result in results)
        assert peak == 1
        assert active == 0
        assert [visit.path for visit in visits].count("/robots.txt") == 1
        assert [visit.started_at for visit in visits] == [0, 2, 4]

    asyncio.run(run())


def test_other_hosts_can_progress_while_a_response_is_open(policy: CrawlPolicy) -> None:
    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()

        class HeldBody(httpx2.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await release.wait()
                yield b"contact"

        def respond(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/robots.txt":
                return httpx2.Response(404)
            if request.url.host == "first.example":
                return httpx2.Response(
                    200, stream=HeldBody(), headers={"Content-Type": "text/plain"}
                )
            return httpx2.Response(
                200, text="other", headers={"Content-Type": "text/plain"}
            )

        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            first = asyncio.create_task(
                PageReader(client, policy=policy).fetch(
                    HttpUrl("https://first.example/contact")
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            other = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://other.example/contact")
            )
            assert other.error is None
            assert not first.done()
            release.set()
            assert (await first).error is None

    asyncio.run(run())


def test_longer_crawl_delay_is_honoured(policy: CrawlPolicy, clock: Clock) -> None:
    visits: list[Visit] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        visits.append(Visit(request.url.host, request.url.path, clock()))
        text = (
            "User-agent: *\nCrawl-delay: 5\n"
            if request.url.path == "/robots.txt"
            else "contact"
        )
        return httpx2.Response(200, text=text, headers={"Content-Type": "text/plain"})

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            for path in ["first", "second"]:
                assert (
                    await PageReader(client, policy=policy).fetch(
                        HttpUrl(f"https://council.example/{path}")
                    )
                ).error is None
        assert [visit.started_at for visit in visits] == [0, 5, 10]

    asyncio.run(run())


@pytest.mark.parametrize("status", [404, 410, 500, 503, 429])
def test_robots_unavailable_and_unreachable_behaviour(
    policy: CrawlPolicy,
    status: int,
) -> None:
    paths: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx2.Response(status)
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            result = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/contact")
            )
        if status in {404, 410}:
            assert result.error is None
            assert paths == ["/robots.txt", "/contact"]
        else:
            assert result.error is not None and "Deferred" in result.error
            assert paths == ["/robots.txt"]

    asyncio.run(run())


def test_robots_network_errors_fail_closed_and_are_cached(policy: CrawlPolicy) -> None:
    attempts = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        raise httpx2.ConnectError("upstream-secret", request=request)

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            for path in ["first", "second"]:
                result = await PageReader(client, policy=policy).fetch(
                    HttpUrl(f"https://council.example/{path}")
                )
                assert result.error == "Deferred: robots.txt could not be fetched"
                assert "upstream-secret" not in result.model_dump_json()
        assert attempts == 1

    asyncio.run(run())


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("date_header", [False, True])
def test_retry_after_cooldown_is_shared_and_long_waits_are_deferred(
    policy: CrawlPolicy,
    clock: Clock,
    status: int,
    date_header: bool,
) -> None:
    visits: list[Visit] = []
    retry_after = (
        format_datetime(
            datetime.fromtimestamp(policy.wall_clock() + 120, timezone.utc), usegmt=True
        )
        if date_header
        else "120"
    )

    def respond(request: httpx2.Request) -> httpx2.Response:
        visits.append(Visit(request.url.host, request.url.path, clock()))
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        if request.url.path == "/busy":
            return httpx2.Response(status, headers={"Retry-After": retry_after})
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            busy = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/busy")
            )
            assert busy.error is not None and f"HTTP {status}" in busy.error
            reader = PageReader(client, policy=policy, deadline=clock() + 10)
            deferred = await reader.fetch(HttpUrl("https://council.example/other"))
            assert deferred.error is not None and "time budget" in deferred.error
            assert [visit.path for visit in visits] == ["/robots.txt", "/busy"]
            clock.now += 120
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/other")
                )
            ).error is None

    asyncio.run(run())


def test_retry_after_short_wait_is_observed(policy: CrawlPolicy, clock: Clock) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        if request.url.path == "/busy":
            return httpx2.Response(429, headers={"Retry-After": "3"})
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/busy")
            )
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/other")
                )
            ).error is None
        assert clock.waits == [2, 3]

    asyncio.run(run())


def test_redirect_destination_is_checked_against_its_own_robots(
    policy: CrawlPolicy,
) -> None:
    visits: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        visits.append(request)
        if request.url.path == "/robots.txt":
            text = (
                "User-agent: *\nDisallow: /blocked"
                if request.url.host == "other.example"
                else ""
            )
            return httpx2.Response(200, text=text)
        return httpx2.Response(
            302, headers={"Location": "https://other.example/blocked"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            result = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/start")
            )
            assert result.error == "Blocked by robots.txt"
        assert [str(request.url) for request in visits] == [
            "https://council.example/robots.txt",
            "https://council.example/start",
            "https://other.example/robots.txt",
        ]

    asyncio.run(run())


def test_robots_redirect_rules_apply_to_original_origin(policy: CrawlPolicy) -> None:
    urls: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        urls.append(str(request.url))
        if request.url.host == "council.example":
            return httpx2.Response(
                302, headers={"Location": "https://cdn.example/rules.txt"}
            )
        return httpx2.Response(200, text="User-agent: *\nDisallow: /private")

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            result = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/private")
            )
            assert result.error == "Blocked by robots.txt"
        assert urls == [
            "https://council.example/robots.txt",
            "https://cdn.example/rules.txt",
        ]

    asyncio.run(run())


def test_robots_cache_expires_and_honours_cache_control(
    policy: CrawlPolicy, clock: Clock
) -> None:
    downloads = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal downloads
        if request.url.path == "/robots.txt":
            downloads += 1
            rules = "" if downloads == 1 else "User-agent: *\nDisallow: /"
            return httpx2.Response(
                200, text=rules, headers={"Cache-Control": "max-age=10"}
            )
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/first")
                )
            ).error is None
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/second")
                )
            ).error is None
            assert downloads == 1
            clock.now += 11
            blocked = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/third")
            )
            assert blocked.error == "Blocked by robots.txt"
            assert downloads == 2

    asyncio.run(run())


def test_cancelled_fetch_releases_the_host_slot(policy: CrawlPolicy) -> None:
    async def run() -> None:
        started, closed = asyncio.Event(), asyncio.Event()

        class HeldBody(httpx2.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await asyncio.Event().wait()
                yield b""

            async def aclose(self) -> None:
                closed.set()

        def respond(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/robots.txt":
                return httpx2.Response(404)
            if request.url.path == "/held":
                return httpx2.Response(
                    200, stream=HeldBody(), headers={"Content-Type": "text/plain"}
                )
            return httpx2.Response(
                200, text="contact", headers={"Content-Type": "text/plain"}
            )

        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            task = asyncio.create_task(
                PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/held")
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closed.is_set()
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/other")
                )
            ).error is None

    asyncio.run(run())


def test_concurrent_lookups_share_inflight_robots_discovery(
    policy: CrawlPolicy,
) -> None:
    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()
        robots_requests = 0

        class RobotsBody(httpx2.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await release.wait()
                yield b"User-agent: *\nDisallow: /private\n"

        def respond(request: httpx2.Request) -> httpx2.Response:
            nonlocal robots_requests
            if request.url.path == "/robots.txt":
                robots_requests += 1
                return httpx2.Response(200, stream=RobotsBody())
            return httpx2.Response(
                200, text="contact", headers={"Content-Type": "text/plain"}
            )

        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            first = asyncio.create_task(
                PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/first")
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(
                PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/second")
                )
            )
            await asyncio.sleep(0)
            assert robots_requests == 1
            release.set()
            results = await asyncio.gather(first, second)
            assert all(result.error is None for result in results)
            assert robots_requests == 1

    asyncio.run(run())


def test_crawl_delay_exceeding_remaining_budget_is_reported_without_a_wait(
    policy: CrawlPolicy,
    clock: Clock,
) -> None:
    paths: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        return httpx2.Response(200, text="User-agent: *\nCrawl-delay: 600\n")

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            result = await PageReader(
                client, policy=policy, deadline=clock() + 10
            ).fetch(HttpUrl("https://council.example/contact"))
        assert result.error is not None and "time budget" in result.error
        assert paths == ["/robots.txt"]
        assert clock.waits == []

    asyncio.run(run())


def test_oversized_robots_preserves_complete_rules_within_the_limit(
    policy: CrawlPolicy,
) -> None:
    from llm_management.web.crawl import ROBOTS_MAX_BYTES

    closed: list[bool] = []

    class LargeRobots(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"User-agent: *\nDisallow: /private\n" + b"x" * ROBOTS_MAX_BYTES
            raise AssertionError("Robots file exceeded the bounded parser input")

        async def aclose(self) -> None:
            closed.append(True)

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/robots.txt"
        return httpx2.Response(200, stream=LargeRobots())

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            result = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/private")
            )
        assert result.error == "Blocked by robots.txt"
        assert closed == [True]

    asyncio.run(run())


def test_short_lookup_does_not_cache_an_unfetched_robots_denial(
    policy: CrawlPolicy,
    clock: Clock,
) -> None:
    paths: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            # An earlier request on this host creates a two-second wait before
            # a different origin can fetch its own robots file.
            async with policy.request(
                client, httpx2.URL("http://council.example/robots.txt"), None
            ):
                pass
            short = await PageReader(client, policy=policy, deadline=clock() + 1).fetch(
                HttpUrl("https://council.example/contact")
            )
            assert short.error is not None and "time budget" in short.error
            normal = await PageReader(client, policy=policy).fetch(
                HttpUrl("https://council.example/contact")
            )
            assert normal.error is None
        assert paths == ["/robots.txt", "/robots.txt", "/contact"]

    asyncio.run(run())


def test_updated_robots_can_reduce_a_previously_longer_delay(
    policy: CrawlPolicy,
    clock: Clock,
) -> None:
    visits: list[Visit] = []
    robots_requests = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal robots_requests
        visits.append(Visit(request.url.host, request.url.path, clock()))
        if request.url.path == "/robots.txt":
            robots_requests += 1
            text = "User-agent: *\nCrawl-delay: 5" if robots_requests == 1 else ""
            return httpx2.Response(
                200, text=text, headers={"Cache-Control": "max-age=10"}
            )
        return httpx2.Response(
            200, text="contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/first")
                )
            ).error is None
            clock.now += 20
            assert (
                await PageReader(client, policy=policy).fetch(
                    HttpUrl("https://council.example/second")
                )
            ).error is None
        assert [visit.started_at for visit in visits] == [0, 5, 25, 27]

    asyncio.run(run())


class CrawlerVisit(NamedTuple):
    """
    Capture project identities and timing across robots and page requests.
    """

    user_agent: str
    path: str
    started_at: float


def test_project_identities_use_their_own_rules_and_shared_host_pacing(
    policy: CrawlPolicy, clock: Clock
) -> None:
    first_agent = "FirstProject/1.0 (+https://first.example/bot)"
    second_agent = "SecondProject/2.0 (+https://second.example/bot)"
    visits: list[CrawlerVisit] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        user_agent = request.headers["User-Agent"]
        visits.append(CrawlerVisit(user_agent, request.url.path, clock()))
        if request.url.path == "/robots.txt":
            return httpx2.Response(302, headers={"Location": "/rules"})
        if request.url.path == "/rules":
            return httpx2.Response(
                200,
                text=(
                    "User-agent: FirstProject\nDisallow: /blocked\nCrawl-delay: 7\n"
                    if user_agent == first_agent
                    else "User-agent: SecondProject\nAllow: /blocked\nCrawl-delay: 3\n"
                ),
            )
        if request.url.path == "/start":
            return httpx2.Response(302, headers={"Location": "/contact"})
        return httpx2.Response(
            200, text="Contact", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            first = PageReader(client, policy=policy, user_agent=first_agent)
            second = PageReader(client, policy=policy, user_agent=second_agent)
            blocked = await first.fetch(HttpUrl("https://council.example/blocked"))
            assert blocked.error == "Blocked by robots.txt"
            allowed = await second.fetch(HttpUrl("https://council.example/blocked"))
            assert allowed.error is None
            redirected = await first.fetch(HttpUrl("https://council.example/start"))
            assert redirected.error is None
        assert visits == [
            CrawlerVisit(first_agent, "/robots.txt", 0),
            CrawlerVisit(first_agent, "/rules", 2),
            CrawlerVisit(second_agent, "/robots.txt", 4),
            CrawlerVisit(second_agent, "/rules", 6),
            CrawlerVisit(second_agent, "/blocked", 9),
            CrawlerVisit(first_agent, "/start", 16),
            CrawlerVisit(first_agent, "/contact", 23),
        ]

    asyncio.run(run())


def test_host_cooldowns_apply_across_project_identities(
    policy: CrawlPolicy, clock: Clock
) -> None:
    visits: list[CrawlerVisit] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        visits.append(
            CrawlerVisit(request.headers["User-Agent"], request.url.path, clock())
        )
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        return httpx2.Response(429, headers={"Retry-After": "60"})

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            first = PageReader(client, policy=policy, user_agent="FirstProject/1.0")
            second = PageReader(client, policy=policy, user_agent="SecondProject/1.0")
            limited = await first.fetch(HttpUrl("https://council.example/busy"))
            assert limited.error and "HTTP 429" in limited.error
            deferred = await second.fetch(HttpUrl("https://council.example/contact"))
            assert deferred.error and "time budget" in deferred.error
        assert visits == [
            CrawlerVisit("FirstProject/1.0", "/robots.txt", 0),
            CrawlerVisit("FirstProject/1.0", "/busy", 2),
        ]

    asyncio.run(run())
