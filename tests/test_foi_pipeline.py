"""Direct pipeline tests: no FastAPI app or TestClient needed."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from llm_management.deployments import get_catalog
from llm_management.deployments.types import RunningDeployment
from llm_management.foi import backends, pipeline, topic_classification
from llm_management.foi.backends import ClassificationRows, DeploymentAccess
from llm_management.foi.schemas import TopicOutput
from llm_management.sanitization import presidio


def test_pipeline_can_run_without_http_and_skip_empty_classification(
    monkeypatch, foi_resources
):
    classify = AsyncMock(
        return_value=ClassificationRows([[1, 0, 0, 0]], "test", "pinned")
    )
    monkeypatch.setattr(backends, "classify_question_units", classify)
    deployments = DeploymentAccess(
        Mock(side_effect=AssertionError("No deployment needed")), AsyncMock(), Mock()
    )
    result = asyncio.run(
        pipeline.process_information_request(
            "Thank you.", deployments=deployments, resources=foi_resources
        )
    )
    assert result.classification is None
    assert result.extraction_status == "no_questions_found"
    assert classify.call_args.kwargs["backend"] == "cpu"
    assert classify.call_args.kwargs["resources"] is foi_resources
    deployments.ensure_running.assert_not_called()


def test_waiting_topic_model_does_not_block_other_extractions(
    monkeypatch, foi_resources
):
    calls = 0

    async def classify_units(texts, **kwargs):
        nonlocal calls
        calls += 1
        return ClassificationRows(
            [[0, 0, 0, 1] if calls == 1 else [1, 0, 0, 0]], "test", "pinned"
        )

    monkeypatch.setattr(backends, "classify_question_units", classify_units)
    monkeypatch.setattr(
        topic_classification,
        "prepare_topic_request",
        lambda *args, **kwargs: presidio.sanitize_payload({"messages": []}),
    )
    cfg = SimpleNamespace(
        model=get_catalog().get(get_catalog().require_foi().topic_deployment).model
    )
    deployments = DeploymentAccess(
        lambda slug: cfg,
        AsyncMock(
            return_value=RunningDeployment(
                config=cfg,
                state=SimpleNamespace(deployment_url="https://test/v1", api_key="test"),
            )
        ),
        Mock(),
    )

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def classify_topics(**kwargs):
            entered.set()
            await release.wait()
            return TopicOutput(
                questions=[{"question_id": "q1", "regime": "FOI", "topic": "Reports"}],
                request_topics=["Reports"],
            )

        monkeypatch.setattr(topic_classification, "classify_topics", classify_topics)
        waiting = asyncio.create_task(
            pipeline.process_information_request(
                "Please provide reports.",
                deployments=deployments,
                resources=foi_resources,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            other = await asyncio.wait_for(
                pipeline.extract_questions(
                    "Thank you.", deployments=deployments, resources=foi_resources
                ),
                2,
            )
            assert other.extraction_status == "no_questions_found"
        finally:
            release.set()
            result = await waiting
        assert result.questions[0].text == "Please provide reports."
        assert result.promoted_continuation_index == 0
        assert result.classification.questions[0].regime == "FOI"
        deployments.touch.assert_called_once()

    asyncio.run(run())
