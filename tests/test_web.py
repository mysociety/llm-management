"""
Public web navigation, redirect bounds and search adapters.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx2
import pytest
from pydantic import HttpUrl

from llm_management.web.crawl import CrawlPolicy
from llm_management.web.pages import PageReader
from llm_management.web.transport import PublicWebTransport, public_address

URL = "https://council.example/foi"


def test_reader_redirects_relative_links_and_limits() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        if request.url.path == "/":
            return httpx2.Response(302, headers={"Location": "/info/foi"})
        return httpx2.Response(
            200,
            text=(
                '<script>ignore me</script><p>FOI email <a href="mailto:foi@council.example">Email</a></p>'
                '<a href="form">Request form</a>'
            ),
            headers={"Content-Type": "text/html"},
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            page = await PageReader(client, policy=CrawlPolicy(interval=0)).fetch(
                HttpUrl("https://council.example/")
            )
            assert str(page.url) == "https://council.example/info/foi"
            assert "ignore me" not in page.content
            assert "foi@council.example" in page.content
            assert HttpUrl("https://council.example/info/form") in page.links
            assert (
                await PageReader(
                    client, max_chars=5, policy=CrawlPolicy(interval=0)
                ).fetch(HttpUrl(URL))
            ).truncated
            assert (
                await PageReader(
                    client, max_bytes=5, policy=CrawlPolicy(interval=0)
                ).fetch(HttpUrl(URL))
            ).error

    asyncio.run(run())


@pytest.mark.parametrize(
    "ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"]
)
def test_private_dns_results_blocked(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    async def run() -> None:
        monkeypatch.setattr(
            asyncio.get_running_loop(),
            "getaddrinfo",
            AsyncMock(return_value=[(2, 1, 6, "", (ip, 443))]),
        )
        with pytest.raises(ValueError):
            await public_address("council.example", 443)

    asyncio.run(run())


def test_transport_pins_ip_and_checks_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_management.web import transport as web_transport

    seen: list[httpx2.Request] = []

    async def resolve(host: str, port: int) -> str:
        if host == "internal.example":
            raise ValueError("Private address")
        return "93.184.216.34"

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        seen.append(request)
        return httpx2.Response(302, headers={"Location": "https://internal.example/"})

    monkeypatch.setattr(web_transport, "public_address", resolve)

    async def run() -> None:
        transport = PublicWebTransport()
        await transport.transport.aclose()
        transport.transport = httpx2.MockTransport(respond)
        async with httpx2.AsyncClient(transport=transport) as client:
            page = await PageReader(client, policy=CrawlPolicy(interval=0)).fetch(
                HttpUrl(URL)
            )
            assert page.error
        assert len(seen) == 1
        assert seen[0].url.host == "93.184.216.34"
        assert seen[0].headers["host"] == "council.example"
        assert seen[0].extensions["sni_hostname"] == "council.example"

    asyncio.run(run())


def test_tavily_adapter_reports_usage_and_skips_invalid_results() -> None:
    import json

    from llm_management.deployments.api_config import APISearchService
    from llm_management.web.search import build_search

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["Authorization"] == "Bearer search-secret"
        assert json.loads(request.content)["include_usage"] is True
        return httpx2.Response(
            200,
            json={
                "usage": {"credits": 1},
                "results": [
                    {"title": "Council", "url": URL, "content": "Public guidance"},
                    {"url": "invalid"},
                ],
            },
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            provider = build_search(
                APISearchService(provider="tavily", credential="tavily"),
                api_key="search-secret",
                client=client,
            )
            assert provider.name == "tavily"
            assert "search-secret" not in repr(provider)
            results = await provider.search("Council FOI")
            assert len(results.results) == 1
            assert results.credits == 1

    asyncio.run(run())


def test_redirect_bodies_are_closed_without_reading() -> None:
    closed: list[bool] = []

    class RedirectBody(httpx2.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            raise AssertionError("Redirect body must not be buffered")
            yield b""  # Make this an async iterator without returning a body.

        async def aclose(self) -> None:
            closed.append(True)

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        if request.url.path == "/start":
            return httpx2.Response(
                302, headers={"Location": "/end"}, stream=RedirectBody()
            )
        return httpx2.Response(200, text="OK", headers={"Content-Type": "text/plain"})

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            page = await PageReader(
                client, max_bytes=5, policy=CrawlPolicy(interval=0)
            ).fetch(HttpUrl("https://council.example/start"))
            assert page.error is None
            assert page.content == "OK"
            assert page.url == HttpUrl("https://council.example/end")
        assert closed == [True]

    asyncio.run(run())


def test_download_limit_still_applies_after_redirect() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        if request.url.path == "/start":
            return httpx2.Response(302, headers={"Location": "/end"})
        return httpx2.Response(
            200, text="oversized final page", headers={"Content-Type": "text/plain"}
        )

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond)
        ) as client:
            page = await PageReader(
                client, max_bytes=5, policy=CrawlPolicy(interval=0)
            ).fetch(HttpUrl("https://council.example/start"))
            assert page.error
            assert page.url == HttpUrl("https://council.example/end")

    asyncio.run(run())


def test_redirect_chain_has_a_fixed_limit() -> None:
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            return httpx2.Response(404)
        requests.append(request)
        return httpx2.Response(302, headers={"Location": "/loop"})

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(respond), max_redirects=100
        ) as client:
            page = await PageReader(
                client, max_redirects=2, policy=CrawlPolicy(interval=0)
            ).fetch(HttpUrl(URL))
            assert page.error
        assert len(requests) == 3

    asyncio.run(run())


@pytest.mark.parametrize("max_chars", [100, 100_000])
def test_link_heavy_pages_bound_navigation_output(max_chars: int) -> None:
    from llm_management.web.pages import MAX_PAGE_LINKS

    html = "".join(f'<a href="/page/{index}">Link</a>' for index in range(1000))

    async def run() -> None:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda request: (
                    httpx2.Response(404)
                    if request.url.path == "/robots.txt"
                    else httpx2.Response(
                        200, text=html, headers={"Content-Type": "text/html"}
                    )
                )
            )
        ) as client:
            page = await PageReader(
                client, max_chars=max_chars, policy=CrawlPolicy(interval=0)
            ).fetch(HttpUrl(URL))
        assert page.error is None
        assert len(page.content) <= max_chars
        assert 0 < len(page.links) <= MAX_PAGE_LINKS
        assert sum(len(str(link)) for link in page.links) <= max_chars
        assert page.links[0] == HttpUrl("https://council.example/page/0")
        assert page.truncated

    asyncio.run(run())


def test_navigation_skips_oversized_links_and_deduplicates() -> None:
    from llm_management.web.pages import ReadableHTML

    parser = ReadableHTML(HttpUrl(URL), max_link_chars=60)
    parser.feed(
        f'<a href="/{"x" * 100}">Too long</a>'
        '<a href="/contact">Contact</a><a href="/contact">Contact again</a>'
        '<a href="/form">Form</a><a href="mailto:foi%40council.example">Email</a>'
    )
    assert parser.links == [
        HttpUrl("https://council.example/contact"),
        HttpUrl("https://council.example/form"),
    ]
    assert parser.links_truncated
    assert "foi@council.example" in "".join(parser.parts)
