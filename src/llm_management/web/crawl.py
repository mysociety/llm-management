"""
Shared robots rules, request spacing and cooldowns for public-web navigation.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import NamedTuple

import httpx2
from protego import Protego

ROBOT_TOKEN = "mySociety-LLMManagement"
USER_AGENT = f"{ROBOT_TOKEN}/0.1 (+https://www.mysociety.org/)"
ROBOTS_MAX_BYTES = 512 * 1024
ROBOTS_CACHE_SIZE = 256
CLEANUP_INTERVAL = 60
HOST_IDLE_SECONDS = 300


class CrawlRestricted(ValueError):
    """
    A known crawling restriction that can safely be reported to the caller.
    """


class CrawlBudgetExceeded(CrawlRestricted):
    """
    One lookup cannot wait long enough; this must not poison shared robots rules.
    """


@dataclass
class SharedLock:
    """
    Count holders and queued callers so cleanup never replaces an in-use lock.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        self.users += 1
        try:
            async with self.lock:
                yield
        finally:
            self.users -= 1


@dataclass
class HostState:
    """
    Serialize requests and retain timing independently of lookup-owned clients.
    """

    lock: SharedLock = field(default_factory=SharedLock)
    last_started: float | None = None
    last_used: float = 0
    delay: float = 0
    cooldown_until: float = 0

    def ready_at(self, delay: float) -> float:
        """
        Preserve both the previous spacing requirement and any active cooldown.
        """
        ready_at = self.cooldown_until
        if self.last_started is not None:
            ready_at = max(ready_at, self.last_started + max(self.delay, delay))
        return ready_at


@dataclass
class RobotsRules:
    """
    Parsed rules or a temporary denial for an origin and crawler identity.
    """

    parser: Protego
    expires_at: float
    error: str | None = None


class RobotsCacheKey(NamedTuple):
    """
    Keep robots responses separate when servers vary them by crawler identity.
    """

    origin: str
    user_agent: str


@dataclass
class CrawlPolicy:
    """
    Coordinate all lookups served by one application instance and event loop.
    """

    interval: float = 2
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    hosts: dict[str, HostState] = field(default_factory=dict, init=False)
    robots: OrderedDict[RobotsCacheKey, RobotsRules] = field(
        default_factory=OrderedDict, init=False
    )
    robots_locks: dict[RobotsCacheKey, SharedLock] = field(
        default_factory=dict, init=False
    )
    next_cleanup: float = field(default=0, init=False)

    def cleanup(self) -> None:
        """
        Lazily remove expired rules and safely idle hosts, at most once a minute.
        """
        now = self.clock()
        if now < self.next_cleanup:
            return
        self.next_cleanup = now + CLEANUP_INTERVAL
        for key, rules in list(self.robots.items()):
            if rules.expires_at <= now:
                del self.robots[key]
        for host, state in list(self.hosts.items()):
            if (
                state.lock.users == 0
                and state.last_used + HOST_IDLE_SECONDS <= now
                and state.ready_at(self.interval) <= now
            ):
                del self.hosts[host]

    def cache_rules(self, key: RobotsCacheKey, rules: RobotsRules) -> None:
        """
        Bound cached robots responses by evicting the least recently used entry.
        """
        if rules.expires_at <= self.clock():
            self.robots.pop(key, None)
            return
        self.robots[key] = rules
        self.robots.move_to_end(key)
        while len(self.robots) > ROBOTS_CACHE_SIZE:
            self.robots.popitem(last=False)

    async def check(
        self,
        client: httpx2.AsyncClient,
        url: httpx2.URL,
        deadline: float | None,
        *,
        user_agent: str = USER_AGENT,
    ) -> RobotsRules:
        """
        Cache robots discovery per origin and user agent, then check the URL.
        """
        self.cleanup()
        origin = f"{url.scheme}://{url.netloc.decode('ascii')}"
        key = RobotsCacheKey(origin, user_agent)
        lock = self.robots_locks.setdefault(key, SharedLock())
        try:
            async with lock.hold():
                rules = self.robots.get(key)
                if rules is None or rules.expires_at <= self.clock():
                    rules = await self.fetch_robots(
                        client, url, deadline, user_agent=user_agent
                    )
                    self.cache_rules(key, rules)
                else:
                    self.robots.move_to_end(key)
        finally:
            if lock.users == 0:
                del self.robots_locks[key]
        if rules.error:
            raise CrawlRestricted(rules.error)
        if url.path != "/robots.txt" and not rules.parser.can_fetch(
            str(url), user_agent
        ):
            raise CrawlRestricted("Blocked by robots.txt")
        return rules

    async def fetch_robots(
        self,
        client: httpx2.AsyncClient,
        url: httpx2.URL,
        deadline: float | None,
        *,
        user_agent: str = USER_AGENT,
    ) -> RobotsRules:
        """
        Follow up to five redirects; apply resulting rules to the initial origin.
        """
        current = url.copy_with(path="/robots.txt", query=None, fragment=None)
        try:
            for redirect in range(6):
                # Robots discovery is exempt from robots checks, including its
                # redirects, but still uses host pacing and the safe transport.
                async with self.request(
                    client, current, deadline, user_agent=user_agent
                ) as response:
                    if response.has_redirect_location:
                        if redirect == 5:
                            raise CrawlRestricted(
                                "Deferred: robots.txt redirect limit reached"
                            )
                        assert response.next_request is not None
                        current = response.next_request.url
                        continue
                    if 400 <= response.status_code < 500:
                        return RobotsRules(
                            Protego.parse(""), self.clock() + self.cache_ttl(response)
                        )
                    if not response.is_success:
                        raise CrawlRestricted(
                            "Deferred: robots.txt is temporarily unavailable"
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        remaining = ROBOTS_MAX_BYTES - len(content)
                        content.extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            # Discard a partially downloaded directive; parse
                            # complete rules within the bounded prefix.
                            content = content[: content.rfind(b"\n") + 1]
                            break
                    parser = Protego.parse(content.decode("utf-8", errors="replace"))
                    return RobotsRules(parser, self.clock() + self.cache_ttl(response))
        except CrawlBudgetExceeded:
            raise
        except (httpx2.HTTPError, OSError, ValueError) as exc:
            error = (
                str(exc)
                if isinstance(exc, CrawlRestricted)
                else "Deferred: robots.txt could not be fetched"
            )
            return RobotsRules(
                Protego.parse("User-agent: *\nDisallow: /"), self.clock() + 60, error
            )
        raise AssertionError("Robots redirect loop did not terminate")

    @asynccontextmanager
    async def request(
        self,
        client: httpx2.AsyncClient,
        url: httpx2.URL,
        deadline: float | None,
        *,
        rules: RobotsRules | None = None,
        user_agent: str = USER_AGENT,
    ) -> AsyncIterator[httpx2.Response]:
        """
        Hold one host slot until the response closes, including body consumption.
        """
        self.cleanup()
        host = url.raw_host.decode("ascii").lower()
        state = self.hosts.setdefault(host, HostState())
        try:
            async with state.lock.hold():
                delay = rules.parser.crawl_delay(user_agent) if rules else None
                delay = max(self.interval, delay or 0)
                ready_at = state.ready_at(delay)
                remaining = ready_at - self.clock()
                if remaining > 0:
                    wait_deadline = (
                        deadline if deadline is not None else self.clock() + 30
                    )
                    if ready_at >= wait_deadline:
                        raise CrawlBudgetExceeded(
                            "Deferred: host pacing or cooldown exceeds the lookup time budget"
                        )
                    await self.sleep(remaining)
                state.last_started = self.clock()
                state.delay = delay
                async with client.stream(
                    "GET",
                    url,
                    follow_redirects=False,
                    headers={"User-Agent": user_agent},
                ) as response:
                    if response.status_code in {429, 503}:
                        cooldown = self.retry_after(response.headers.get("retry-after"))
                        state.cooldown_until = max(
                            state.cooldown_until, self.clock() + cooldown
                        )
                        raise CrawlRestricted(
                            f"Deferred: host returned HTTP {response.status_code}; cooldown recorded"
                        )
                    yield response
        finally:
            state.last_used = self.clock()

    def retry_after(self, value: str | None) -> float:
        """
        Honour seconds and HTTP dates; use a minute when no valid delay is given.
        """
        if value is not None:
            if value.strip().isdigit():
                try:
                    return float(value.strip())
                except OverflowError:
                    return float("inf")
            try:
                return max(
                    0, parsedate_to_datetime(value).timestamp() - self.wall_clock()
                )
            except (ValueError, TypeError, OverflowError):
                pass
        return 60

    def cache_ttl(self, response: httpx2.Response) -> float:
        """
        Cache for an hour by default, respecting shorter directives and a day cap.
        """
        control = response.headers.get("cache-control", "").lower()
        if "no-cache" in control or "no-store" in control:
            return 0
        match = re.search(r'(?:^|,)\s*max-age\s*=\s*"?(\d+)', control)
        return min(86400, int(match[1])) if match else 3600
