import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from llm_management.foi import granite as foi_topic
from llm_management.foi.model_spec import GRANITE_MERGED
from llm_management.foi.question_slice import build_extraction_result, segment_request


def extraction(labels=(3,)):
    units = segment_request(
        "\n\n".join(f"Please provide report {i}." for i in range(len(labels)))
    )
    return build_extraction_result(
        units,
        [[float(i == label) for i in range(4)] for label in labels],
        backend="cpu",
        model="modernbert",
        revision="pinned",
    )


def topic_data():
    return {
        "questions": [{"question_id": "q1", "regime": "FOI", "topic": "Reports"}],
        "request_topics": ["Reports"],
    }


@pytest.mark.parametrize("count,budget", [(1, 256), (3, 640), (10, 1984)])
def test_budget(count, budget):
    assert foi_topic.completion_budget(count) == budget


@pytest.mark.parametrize("count", [0, 11])
def test_excess_budget_rejected(count):
    with pytest.raises(ValueError):
        foi_topic.completion_budget(count)


def test_prompt_uses_training_payload_and_exact_chat_tokenization(monkeypatch):
    calls = []

    def tokenize(messages, **kwargs):
        calls.append((messages, kwargs))
        return [1] * 2048

    monkeypatch.setattr(
        foi_topic,
        "granite_tokenizer",
        lambda: SimpleNamespace(apply_chat_template=tokenize),
    )
    payload = foi_topic.prepare_topic_request(
        "Original request", extraction().questions
    )
    user = json.loads(payload["messages"][1]["content"])
    assert list(user) == ["questions", "request_text"]
    assert user["questions"] == [
        {"question_id": "q1", "text": "Please provide report 0."}
    ]
    assert calls[0][1] == dict(
        tokenize=True, add_generation_prompt=True, truncation=False
    )
    assert payload["response_format"]["json_schema"]["strict"] is True
    monkeypatch.setattr(
        foi_topic,
        "granite_tokenizer",
        lambda: SimpleNamespace(apply_chat_template=lambda *a, **kw: [1] * 2049),
    )
    with pytest.raises(ValueError, match="not truncated"):
        foi_topic.prepare_topic_request("Original request", extraction().questions)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "length",
        "refusal",
        "malformed",
        "missing",
        "count",
        "ids",
        "duplicate_topics",
        "blank",
    ],
)
def test_remote_output_validation(monkeypatch, case):
    data = topic_data()
    reason = "stop"
    if case == "length":
        reason = "length"
    if case == "refusal":
        reason = "content_filter"
    if case == "ids":
        data["questions"][0]["question_id"] = "q2"
    if case == "duplicate_topics":
        data["request_topics"] = ["Reports", " reports "]
    if case == "blank":
        data["questions"][0]["topic"] = "  "
    content = "not json" if case == "malformed" else json.dumps(data)
    response = {"choices": [{"finish_reason": reason, "message": {"content": content}}]}
    if case == "missing":
        response = {}
    real_client = httpx.AsyncClient

    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["model"] == GRANITE_MERGED
        return httpx.Response(200, json=response)

    monkeypatch.setattr(
        foi_topic.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )

    async def call():
        return await foi_topic.classify_topics(
            payload={},
            model=GRANITE_MERGED,
            deployment_url="https://test/v1/",
            api_key="test",
            question_ids=["q1", "q2"] if case == "count" else ["q1"],
        )

    if case == "valid":
        assert asyncio.run(call()).model_dump() == data
    else:
        with pytest.raises(foi_topic.TopicOutputError):
            asyncio.run(call())
