"""
Opt-in paid model comparisons against frozen authority research, with no scraping.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

import httpx2
import pytest
from pydantic import HttpUrl

from llm_management.deployments.catalog import DeploymentCatalog
from llm_management.foi.email_finder.replay import ReplayResearch
from llm_management.foi.email_finder.service import find_foi_email
from llm_management.inference.external import build_model, credential
from llm_management.settings import settings
from tests.foi_email_evaluation import (
    EvaluationCase,
    EvaluationReport,
    ModelOnlyTransport,
    assert_finding_matches,
)

pytestmark = [pytest.mark.external, pytest.mark.model_eval]
FIXTURES = Path(__file__).parent / "fixtures" / "foi_email"
CASE_IDS = sorted(path.name for path in FIXTURES.iterdir() if path.is_dir())


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_model_against_recorded_authority(
    case_id: str,
    foi_model_slug: str | None,
    pytestconfig: pytest.Config,
    record_property: Callable[[str, object], None],
) -> None:
    case_directory = FIXTURES / case_id
    if not (case_directory / "expected.json").exists():
        pytest.skip(
            "A reviewed capture and expectation are required; tests never capture live pages"
        )
    case = EvaluationCase.load(case_directory)
    catalog = DeploymentCatalog.load()
    config = catalog.foi_email_finder
    assert config is not None
    deployment = catalog.api.get(foi_model_slug or config.deployment)
    service = catalog.api.model_service(deployment.service)
    replay = ReplayResearch(case.capture)
    report = EvaluationReport(
        case_id=case_id,
        model_slug=deployment.slug,
        deployment=deployment.model_copy(deep=True),
        started_at=datetime.now(timezone.utc),
        capture_sha256=case.expected.capture_sha256,
        trace=replay.trace,
    )

    async def run() -> None:
        transport = ModelOnlyTransport(
            service.base_url or HttpUrl("https://api.openai.com/v1/")
        )
        async with httpx2.AsyncClient(
            transport=transport, trust_env=False, timeout=config.http_timeout_seconds
        ) as client:
            model = build_model(
                deployment,
                service,
                api_key=credential(service.credential, settings),
                client=client,
            )
            async with asyncio.timeout(config.timeout_seconds):
                report.result = await find_foi_email(
                    case.capture.authority,
                    model=model,
                    reader=replay,
                    search_provider=replay,
                    request_limit=config.request_limit,
                    tool_limit=config.tool_limit,
                )
                report.result.usage.model.service = deployment.service

    start = monotonic()
    try:
        asyncio.run(run())
        assert report.result is not None
        assert_finding_matches(report.result, case.expected)
    except Exception as exc:
        report.error = f"Evaluation failed ({type(exc).__name__})"
        raise
    finally:
        report.elapsed_seconds = round(monotonic() - start, 3)
        output = pytestconfig.getoption("foi_eval_output")
        assert isinstance(output, str)
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        safe_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", deployment.slug)
        artifact = directory / f"{case_id}-{safe_slug}-{uuid4().hex}.json"
        artifact.write_text(report.model_dump_json(indent=2) + "\n")
        record_property("case_id", case_id)
        record_property("model_slug", deployment.slug)
        record_property("latency_seconds", report.elapsed_seconds)
        record_property("report_path", str(artifact))
        record_property(
            "actual_output", report.result.model_dump_json() if report.result else ""
        )
