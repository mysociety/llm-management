"""
Replay captured public research without any live search or page fetching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, HttpUrl

from ...web.schemas import SearchHit, SearchResults, WebPage
from .capture import ResearchCapture


class ReplayTrace(BaseModel):
    """
    Keep replay tool requests and missing URLs available for model comparisons.
    """

    queries: list[str] = Field(default_factory=list)
    page_urls: list[HttpUrl] = Field(default_factory=list)
    missing_urls: list[HttpUrl] = Field(default_factory=list)


@dataclass
class ReplayResearch:
    """
    Supply the same frozen discovery results for differently phrased queries.

    Missing pages produce an explicit fixture error. There is no network fallback,
    and historical capture credits are never counted as evaluation charges.
    """

    capture: ResearchCapture
    trace: ReplayTrace = field(default_factory=ReplayTrace)

    @property
    def name(self) -> str:
        return "replay"

    async def search(self, query: str) -> SearchResults:
        self.trace.queries.append(query)
        seen: set[HttpUrl] = set()
        hits: list[SearchHit] = []
        for search in self.capture.searches:
            for hit in search.results:
                if hit.url not in seen:
                    seen.add(hit.url)
                    hits.append(hit.model_copy(deep=True))
        return SearchResults(query=query, results=hits, credits=0)

    async def fetch(self, url: HttpUrl) -> WebPage:
        self.trace.page_urls.append(url)
        for recorded in self.capture.pages:
            if url in (recorded.requested_url, recorded.url):
                page = recorded.model_copy(deep=True)
                page.requested_url = url
                return page
        self.trace.missing_urls.append(url)
        return WebPage(
            requested_url=url,
            url=url,
            error="Page is not available in this research fixture",
        )
