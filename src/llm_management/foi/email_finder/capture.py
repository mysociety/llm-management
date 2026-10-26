"""
Explicit live research capture, separate from replay-only model evaluations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx2
import typer
from pydantic import BaseModel, HttpUrl

from ...deployments.catalog import DeploymentCatalog
from ...inference.external import build_model, credential
from ...settings import Settings
from ...web.crawl import CrawlPolicy
from ...web.pages import PageReader, PageSource
from ...web.schemas import SearchResults, WebPage
from ...web.search import SearchProvider, build_search
from ...web.transport import PublicWebTransport
from .schemas import FOIResult, PublicAuthority
from .service import USER_AGENT, find_foi_email


class ResearchCapture(BaseModel):
    """
    Preserve the actual tool material and result from one dated live capture.
    """

    authority: PublicAuthority
    captured_at: datetime
    model_deployment: str
    searches: list[SearchResults]
    pages: list[WebPage]
    result: FOIResult | None = None
    error: str | None = None


@dataclass
class RecordingPages:
    """
    Retain page contents, redirects and errors without changing the live reader.
    """

    source: PageSource
    pages: list[WebPage] = field(default_factory=list)

    async def fetch(self, url: HttpUrl) -> WebPage:
        page = await self.source.fetch(url)
        self.pages.append(page.model_copy(deep=True))
        return page


@dataclass
class RecordingSearch:
    """
    Record search results and failures while retaining actual provider usage.
    """

    source: SearchProvider
    searches: list[SearchResults] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.source.name

    async def search(self, query: str) -> SearchResults:
        try:
            result = await self.source.search(query)
        except Exception as exc:
            result = SearchResults(
                query=query, error=f"Search failed ({type(exc).__name__})"
            )
        self.searches.append(result.model_copy(deep=True))
        return result


async def capture_authority(
    authority: PublicAuthority,
    *,
    settings: Settings,
    catalog: DeploymentCatalog,
    model_slug: str | None = None,
    policy: CrawlPolicy | None = None,
) -> ResearchCapture:
    """
    Make an explicitly requested live capture using the configured research budgets.

    Only this operation uses live search and page navigation. Tests replay the
    saved material and never invoke this function to fill missing fixture pages.
    """
    config = catalog.foi_email_finder
    if config is None:
        raise ValueError("FOI email finder is not configured")
    deployment = catalog.api.get(model_slug or config.deployment)
    service = catalog.api.model_service(deployment.service)
    api_key = credential(service.credential, settings)
    deadline = asyncio.get_running_loop().time() + config.timeout_seconds
    capture = ResearchCapture(
        authority=authority,
        captured_at=datetime.now(timezone.utc),
        model_deployment=deployment.slug,
        searches=[],
        pages=[],
    )
    async with (
        httpx2.AsyncClient(timeout=config.http_timeout_seconds, trust_env=False) as api,
        httpx2.AsyncClient(
            transport=PublicWebTransport(),
            timeout=config.http_timeout_seconds,
            trust_env=False,
        ) as web,
    ):
        reader = RecordingPages(
            PageReader(
                web,
                max_chars=config.max_page_chars,
                max_bytes=config.max_download_bytes,
                policy=policy if policy is not None else CrawlPolicy(),
                deadline=deadline,
                user_agent=USER_AGENT,
            )
        )
        search: RecordingSearch | None = None
        if config.search:
            search_deployment = catalog.api.get_search(config.search)
            search_service = catalog.api.search_service(search_deployment.service)
            search = RecordingSearch(
                build_search(
                    search_service,
                    api_key=credential(search_service.credential, settings),
                    client=api,
                )
            )
        try:
            async with asyncio.timeout_at(deadline):
                capture.result = await find_foi_email(
                    authority,
                    model=build_model(deployment, service, api_key=api_key, client=api),
                    reader=reader,
                    search_provider=search,
                    request_limit=config.request_limit,
                    tool_limit=config.tool_limit,
                )
                capture.result.usage.model.service = deployment.service
        except Exception as exc:
            capture.error = f"Capture did not complete ({type(exc).__name__})"
        finally:
            capture.pages = reader.pages
            capture.searches = search.searches if search is not None else []
    return capture


app = typer.Typer()


@app.command()
def capture(
    name: str = typer.Option(..., "--name"),
    output: Path = typer.Option(
        ..., "--output", help="New capture JSON file; existing files are protected."
    ),
    website: str | None = typer.Option(None, "--website"),
    context: str = typer.Option("", "--context"),
    previous_email: str | None = typer.Option(None, "--previous-email"),
    model_slug: str | None = typer.Option(None, "--model"),
) -> None:
    """
    Capture one authority through paid APIs and courteous live page navigation.
    """
    if output.exists():
        raise typer.BadParameter(
            "Choose a new output file to preserve the existing capture"
        )
    result = asyncio.run(
        capture_authority(
            PublicAuthority(
                name=name,
                previous_email=previous_email,
                website=HttpUrl(website) if website else None,
                context=context,
            ),
            settings=Settings(),
            catalog=DeploymentCatalog.load(),
            model_slug=model_slug,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as target:
        target.write(result.model_dump_json(indent=2) + "\n")
    typer.echo(
        f"Saved {len(result.searches)} searches and {len(result.pages)} pages to {output}"
    )
    if result.error:
        typer.echo(result.error)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
