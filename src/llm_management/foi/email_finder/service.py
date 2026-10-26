"""
Async FOI research using explicitly permitted external services.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx2
from openai import APIError
from pydantic_ai.exceptions import AgentRunError
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from ...deployments.api_config import EmailFinderConfig
from ...deployments.catalog import DeploymentCatalog
from ...inference.errors import ClassifierBusy
from ...inference.external import (
    ExternalFailure,
    ExternalUnavailable,
    build_model,
    credential,
)
from ...settings import Settings
from ...web.crawl import CrawlPolicy
from ...web.pages import PageReader, PageSource
from ...web.schemas import PageVisit, SearchResults
from ...web.search import SearchProvider, build_search
from ...web.transport import PublicWebTransport
from .agent import ResearchContext, build_agent
from .schemas import (
    FOIResult,
    FOIUsage,
    ModelUsage,
    PublicAuthority,
    SearchUsage,
)

USER_AGENT = "mySociety-FOIFinder/0.1 (+https://mysociety.org)"


class FinderInputError(ValueError):
    """
    Public authority details are insufficient for the configured research mode.
    """


@dataclass
class FOIFinder:
    """
    Cold application owner; clients and research state belong to each lookup.
    """

    settings: Settings
    catalog: DeploymentCatalog
    active: int = 0
    crawl_policy: CrawlPolicy = field(default_factory=CrawlPolicy)

    async def find(self, authority: PublicAuthority) -> FOIResult:
        if self.active >= 4:
            raise ClassifierBusy("FOI finder is busy")
        self.active += 1
        try:
            config = self.catalog.foi_email_finder
            if config is None:
                raise ExternalUnavailable("FOI email finder is not configured")
            deadline = asyncio.get_running_loop().time() + config.timeout_seconds
            async with asyncio.timeout_at(deadline):
                return await self._configured_find(authority, config, deadline)
        finally:
            self.active -= 1

    async def _configured_find(
        self, authority: PublicAuthority, config: EmailFinderConfig, deadline: float
    ) -> FOIResult:
        deployment = self.catalog.api.get(config.deployment)
        model_service = self.catalog.api.model_service(deployment.service)
        model_key = credential(model_service.credential, self.settings)
        # API clients never share credentials, cookies or pools with web navigation.
        async with (
            httpx2.AsyncClient(
                timeout=config.http_timeout_seconds, trust_env=False
            ) as api,
            httpx2.AsyncClient(
                transport=PublicWebTransport(),
                trust_env=False,
                timeout=config.http_timeout_seconds,
            ) as web,
        ):
            model = build_model(
                deployment, model_service, api_key=model_key, client=api
            )
            search: SearchProvider | None = None
            if config.search:
                search_deployment = self.catalog.api.get_search(config.search)
                search_service = self.catalog.api.search_service(
                    search_deployment.service
                )
                search = build_search(
                    search_service,
                    api_key=credential(search_service.credential, self.settings),
                    client=api,
                )
            result = await find_foi_email(
                authority,
                model=model,
                reader=PageReader(
                    web,
                    max_chars=config.max_page_chars,
                    max_bytes=config.max_download_bytes,
                    policy=self.crawl_policy,
                    deadline=deadline,
                    user_agent=USER_AGENT,
                ),
                search_provider=search,
                request_limit=config.request_limit,
                tool_limit=config.tool_limit,
            )
            result.usage.model.service = deployment.service
            return result


async def find_foi_email(
    authority: PublicAuthority,
    *,
    model: Model,
    reader: PageSource,
    search_provider: SearchProvider | None = None,
    request_limit: int = 20,
    tool_limit: int = 30,
) -> FOIResult:
    """
    Inject a model, search and reader for direct batch use or offline tests.

    The caller owns supplied dependencies. This public-information pipeline never
    accepts request correspondence and does not change other pipelines' policy.
    """
    if search_provider is None and authority.website is None:
        raise FinderInputError("Supply a website when external search is disabled")
    context = ResearchContext(reader, search_provider, tool_limit)
    try:
        result = await build_agent(model).run(
            f"Find the FOI submission email for this authority:\n{authority.model_dump_json()}\n"
            f"Budget: {request_limit} model requests and {tool_limit} tool calls.",
            deps=context,
            usage_limits=UsageLimits(
                request_limit=request_limit, tool_calls_limit=tool_limit
            ),
        )
    except (AgentRunError, APIError, httpx2.HTTPError) as exc:
        raise ExternalFailure("FOI research failed or exceeded its budget") from exc
    return FOIResult(
        authority=authority,
        finding=result.output,
        checked_at=datetime.now(timezone.utc),
        searches=context.searches,
        usage=summarize_usage(model, result.usage, context.searches, search_provider),
        pages=[
            PageVisit(
                requested_url=page.requested_url,
                url=page.url,
                error=page.error,
                truncated=page.truncated,
            )
            for page in context.pages
        ],
    )


def summarize_usage(
    model: Model,
    usage: RunUsage,
    searches: list[SearchResults],
    search_provider: SearchProvider | None,
) -> FOIUsage:
    """
    Report model totals and preserve unknown credits for any unmeasured search.
    """
    search_usage = SearchUsage(provider=None, requests=0, credits=0)
    if search_provider is not None:
        credits = [search.credits for search in searches]
        search_usage = SearchUsage(
            provider=search_provider.name,
            requests=len(searches),
            credits=(
                sum(value for value in credits if value is not None)
                if all(value is not None for value in credits)
                else None
            ),
        )
    return FOIUsage(
        model=ModelUsage(
            provider=model.system,
            model=model.model_name,
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            cached_input_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.details.get("reasoning_tokens"),
        ),
        search=search_usage,
    )
