import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from llm_management.foi import question_extractor as foi_topic
from llm_management.deployments import get_catalog
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
        "question_extractor_tokenizer",
        lambda *args: SimpleNamespace(apply_chat_template=tokenize),
    )
    payload = foi_topic.prepare_topic_request(
        "Original request", extraction().questions
    )
    payload = payload.value
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
        "question_extractor_tokenizer",
        lambda *args: SimpleNamespace(apply_chat_template=lambda *a, **kw: [1] * 2049),
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
        "no_finish_reason",
        "http_error",
        "timeout",
        "explicit_refusal",
    ],
)
def test_remote_output_validation(monkeypatch, case):
    data = topic_data()
    reason = "stop"
    if case == "length":
        reason = "length"
    if case == "refusal":
        reason = "content_filter"
    if case == "no_finish_reason":
        reason = None
    if case == "ids":
        data["questions"][0]["question_id"] = "q2"
    if case == "duplicate_topics":
        data["request_topics"] = ["Reports", " reports "]
    if case == "blank":
        data["questions"][0]["topic"] = "  "
    content = "not json" if case == "malformed" else json.dumps(data)
    response = {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1,
        "model": get_catalog().get(get_catalog().require_foi().topic_deployment).model,
        "choices": [
            {
                "index": 0,
                "finish_reason": reason,
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if case == "missing":
        response = {}
    if case == "explicit_refusal":
        response["choices"][0]["message"]["refusal"] = "Cannot classify"
    monkeypatch.setattr(
        foi_topic,
        "question_extractor_tokenizer",
        lambda *args: SimpleNamespace(apply_chat_template=lambda *a, **kw: [1]),
    )
    payload = foi_topic.prepare_topic_request(
        "Email alice@example.org", extraction().questions
    )
    real_client = httpx.AsyncClient
    calls = []
    clients = []

    def handle(request):
        calls.append(request)
        if case == "http_error":
            return httpx.Response(500, json={"error": {"message": "Unavailable"}})
        if case == "timeout":
            raise httpx.ReadTimeout("Timed out", request=request)
        return httpx.Response(200, json=response)

    class MockClient(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)
            clients.append(self)

    monkeypatch.setattr(foi_topic.httpx, "AsyncClient", MockClient)

    async def call():
        return await foi_topic.classify_topics(
            payload=payload,
            model=get_catalog().get(get_catalog().require_foi().topic_deployment).model,
            deployment_url="https://test/v1/",
            api_key="test",
            question_ids=["q1", "q2"] if case == "count" else ["q1"],
        )

    if case == "valid":
        assert asyncio.run(call()).model_dump() == data
    else:
        with pytest.raises(foi_topic.TopicOutputError):
            asyncio.run(call())
    assert len(calls) == 1
    assert len(clients) == 1
    assert clients[0].is_closed
    request = calls[0]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test"
    body = json.loads(request.content)
    assert (
        body["model"]
        == get_catalog().get(get_catalog().require_foi().topic_deployment).model
    )
    assert body["messages"] == payload.value["messages"]
    assert body["response_format"] == payload.value["response_format"]
    assert body["temperature"] == payload.value["temperature"]
    assert body["max_tokens"] == payload.value["max_tokens"]
    assert "max_completion_tokens" not in body
    assert "tools" not in body
    assert "alice@example.org" not in request.content.decode()


def test_token_budget_checks_final_sanitized_messages(monkeypatch):
    seen = []

    def tokenize(messages, **kwargs):
        seen.extend(messages)
        return [1]

    monkeypatch.setattr(
        foi_topic,
        "question_extractor_tokenizer",
        lambda *args: SimpleNamespace(apply_chat_template=tokenize),
    )
    payload = foi_topic.prepare_topic_request(
        "Email alice@example.org", extraction().questions
    )
    assert "alice@example.org" not in seen[1]["content"]
    assert "<EMAIL_ADDRESS_1>" in seen[1]["content"]
    assert seen == payload.value["messages"]
