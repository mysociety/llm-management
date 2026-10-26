"""
Bounded page reading and visible HTML text extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import unquote, urljoin

import httpx2
from pydantic import HttpUrl

from .crawl import USER_AGENT, CrawlPolicy, CrawlRestricted
from .schemas import WebPage

MAX_PAGE_LINKS = 50


class PageSource(Protocol):
    """
    Supply readable pages through live navigation or a recorded research corpus.
    """

    async def fetch(self, url: HttpUrl) -> WebPage: ...


class ReadableHTML(HTMLParser):
    """
    Extract visible text and links without adding runtime dependencies.
    """

    def __init__(self, url: HttpUrl, *, max_link_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.url = url
        self.parts: list[str] = []
        self.links: list[HttpUrl] = []
        self.hidden = 0
        self.max_link_chars = max_link_chars
        self.link_chars = 0
        self.links_truncated = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")
        href = dict(attrs).get("href") if tag == "a" else None
        if href:
            if href.lower().startswith("mailto:"):
                address = unquote(href[7:].split("?", 1)[0])
                self.parts.append(f" {address} ")
            else:
                try:
                    link = HttpUrl(urljoin(str(self.url), href))
                except ValueError:
                    return
                if link not in self.links:
                    if (
                        len(self.links) >= MAX_PAGE_LINKS
                        or self.link_chars + len(str(link)) > self.max_link_chars
                    ):
                        self.links_truncated = True
                    else:
                        self.links.append(link)
                        self.link_chars += len(str(link))
                self.parts.append(f" [{link}] ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self.hidden:
            self.hidden -= 1
        if not self.hidden and tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


@dataclass
class PageReader:
    """
    Read pages under a project identity using a shareable crawl policy.
    """

    client: httpx2.AsyncClient
    max_chars: int = 40_000
    max_bytes: int = 5_000_000
    max_redirects: int = 5
    policy: CrawlPolicy = field(default_factory=CrawlPolicy)
    deadline: float | None = None
    user_agent: str = USER_AGENT

    async def fetch(self, url: HttpUrl) -> WebPage:
        final_url = url
        try:
            redirects = 0
            while True:
                target = httpx2.URL(str(final_url))
                rules = await self.policy.check(
                    self.client, target, self.deadline, user_agent=self.user_agent
                )
                async with self.policy.request(
                    self.client,
                    target,
                    self.deadline,
                    rules=rules,
                    user_agent=self.user_agent,
                ) as response:
                    final_url = HttpUrl(str(response.url))
                    if response.has_redirect_location:
                        if redirects >= self.max_redirects:
                            raise ValueError("Page exceeds redirect limit")
                        assert response.next_request is not None
                        final_url = HttpUrl(str(response.next_request.url))
                        redirects += 1
                        # Leaving the context closes the intermediate response
                        # without buffering an unbounded redirect body.
                        continue
                    response.raise_for_status()
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > self.max_bytes:
                            raise ValueError("Page exceeds download limit")
                        content.extend(chunk)
                    content_type = response.headers.get("content-type", "").lower()
                    text = content.decode(
                        response.encoding or "utf-8", errors="replace"
                    )
                    break
            links: list[HttpUrl] = []
            links_truncated = False
            if "text/plain" in content_type:
                pass
            elif "html" in content_type or not content_type:
                parser = ReadableHTML(final_url, max_link_chars=self.max_chars)
                parser.feed(text)
                text, links = "".join(parser.parts), parser.links
                links_truncated = parser.links_truncated
            else:
                raise ValueError("Unsupported content type")
            return WebPage(
                requested_url=url,
                url=final_url,
                content=text[: self.max_chars],
                links=links,
                truncated=len(text) > self.max_chars or links_truncated,
            )
        except CrawlRestricted as exc:
            return WebPage(requested_url=url, url=final_url, error=str(exc))
        except (httpx2.HTTPError, OSError, ValueError) as exc:
            # Never return raw upstream errors, which may contain credentials.
            return WebPage(
                requested_url=url,
                url=final_url,
                error=f"Could not read page ({type(exc).__name__})",
            )
