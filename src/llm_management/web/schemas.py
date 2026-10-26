"""
Typed search results, page content and navigation records.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class SearchHit(BaseModel):
    """
    A search result containing a title, URL and content snippet.
    """

    title: str = ""
    url: HttpUrl
    content: str = ""


class SearchResults(BaseModel):
    """
    Search results, provider usage and any operational error.
    """

    query: str
    results: list[SearchHit] = Field(default_factory=list)
    error: str | None = None
    credits: int | None = Field(default=None, ge=0)


class PageVisit(BaseModel):
    """
    Record redirects, failures, and truncation for later review.
    """

    requested_url: HttpUrl
    url: HttpUrl
    error: str | None = None
    truncated: bool = False


class WebPage(PageVisit):
    """
    Extracted page content and links alongside the fetch record.
    """

    content: str = ""
    links: list[HttpUrl] = Field(default_factory=list)
