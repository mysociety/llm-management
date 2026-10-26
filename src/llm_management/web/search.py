"""
Replaceable search API adapters, independent of direct page crawling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx2
from pydantic import ValidationError

from ..deployments.api_config import APISearchService
from .schemas import SearchHit, SearchResults


class SearchProvider(Protocol):
    @property
    def name(self) -> str:
        """
        Identify the provider whose search usage is reported.
        """
        ...

    async def search(self, query: str) -> SearchResults: ...


@dataclass
class TavilySearch:
    client: httpx2.AsyncClient
    api_key: str = field(repr=False)

    @property
    def name(self) -> str:
        """
        Identify the search provider without inspecting credentials or URLs.
        """
        return "tavily"

    async def search(self, query: str) -> SearchResults:
        response = await self.client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
                "include_usage": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        results: list[SearchHit] = []
        for item in payload.get("results", [])[:5]:
            try:
                results.append(SearchHit.model_validate(item))
            except ValidationError:
                continue
        return SearchResults(
            query=query,
            results=results,
            credits=(payload.get("usage") or {}).get("credits"),
        )


def build_search(
    service: APISearchService, *, api_key: str, client: httpx2.AsyncClient
) -> SearchProvider:
    """
    Construct the configured search adapter without provider details in features.
    """
    if service.provider == "tavily":
        return TavilySearch(client, api_key)
    raise ValueError(f"Unsupported search provider: {service.provider}")
