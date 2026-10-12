"""
Status transport, exact training prompts and lifecycle behavior without cloud calls.
"""

import asyncio
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest

from llm_management import server
from llm_management.deployments.types import DeploymentState, RunningDeployment
from llm_management.foi import response_model
from llm_management.foi.response_schemas import ExtractionInput
from llm_management.inference.resources import LocalResource


@pytest.fixture
def body():
    return {
        "request": {
            "questions": [
                {"question_id": "q1", "text": "Please provide Alice Smith's report."}
            ],
            "extraction_status": "questions_found",
            "additional_text": [],
        },
        "sources": [
            {
                "id": "email1",
                "kind": "email",
                "role": "current",
                "sender": "authority",
                "filename": None,
                "availability": "visible",
                "text": "Dear Alice Smith, the total is 42.",
            }
        ],
    }


@pytest.fixture
def output():
    return {
        "outcomes": [
            {
                "scope": {"kind": "question", "question_id": "q1"},
                "act": "supplied",
                "answer_content_visibility": "visible",
                "content_source_ids": ["email1"],
                "references": [],
            }
        ],
        "events": [],
        "process_references": [],
    }


@pytest.fixture
def serving(monkeypatch, output):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    calls = []
    clients = []
    tokenization = []
    control = {"tokens": 20, "finish_reason": "stop", "content": output}

    def tokenize(messages, **kwargs):
        tokenization.append((messages, kwargs))
        return [1] * control["tokens"]

    guard = LocalResource(
        "foi_status_tokenizer",
        lambda: SimpleNamespace(apply_chat_template=tokenize),
        lambda: None,
    )
    monkeypatch.setattr(
        server.app.state.application.foi, "response_tokenizer", lambda: guard
    )
    cfg = server.app.state.service.catalog.get("foi_status")
    ensure = AsyncMock(
        return_value=RunningDeployment(
            config=cfg,
            state=DeploymentState(
                slug=cfg.slug,
                replicas=1,
                deployment_url="https://status.test/v1/",
                api_key="upstream-secret",
            ),
        )
    )
    monkeypatch.setattr(server.app.state.service, "ensure_running", ensure)
    touch = []
    monkeypatch.setattr(server.app.state.service.cache, "touch", touch.append)
    real_client = httpx2.AsyncClient

    def handle(request):
        calls.append(request)
        if control.get("error"):
            raise control["error"]
        if control.get("http_status"):
            return httpx2.Response(
                control["http_status"], json={"error": {"message": "failed"}}
            )
        content = control["content"]
        reply = {
            "id": "status-completion",
            "object": "chat.completion",
            "created": 1,
            "model": cfg.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": control["finish_reason"],
                    "message": {
                        "role": "assistant",
                        "content": content
                        if isinstance(content, str)
                        else json.dumps(content),
                        "refusal": control.get("refusal"),
                    },
                }
            ],
        }
        if control.get("no_choices"):
            reply["choices"] = []
        return httpx2.Response(200, json=reply)

    class MockClient(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx2.MockTransport(handle), **kwargs)
            clients.append(self)

    monkeypatch.setattr(response_model.httpx2, "AsyncClient", MockClient)
    return SimpleNamespace(
        control=control,
        calls=calls,
        clients=clients,
        tokenize=tokenization,
        ensure=ensure,
        touch=touch,
    )


def test_exact_prompt_and_sanitized_transport(serving, body, output):
    from starlette.testclient import TestClient

    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == {**output, "process_states": None}
    serving.ensure.assert_awaited_once_with("foi_status")
    assert serving.touch == ["foi_status"]
    request = serving.calls[0]
    payload = json.loads(request.content)
    system = payload["messages"][0]["content"]
    assert (
        hashlib.sha256(system.encode()).hexdigest()
        == "9ddffc18c8fe3d5d88fb3d3b5a07387d7bdfe363cb3e7fd01f0a40263e86a7f5"
    )
    canonical = json.loads(payload["messages"][1]["content"])
    assert canonical["request_text"] is None
    assert "Alice Smith" not in payload["messages"][1]["content"]
    assert canonical["request"]["questions"][0]["question_id"] == "q1"
    assert canonical["sources"][0]["id"] == "email1"
    assert payload["messages"][1]["content"] == json.dumps(
        canonical, ensure_ascii=False, sort_keys=True
    )
    assert (
        "process_states"
        not in payload["response_format"]["json_schema"]["schema"]["properties"]
    )
    assert payload["max_tokens"] == 2048
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert request.headers["Authorization"] == "Bearer upstream-secret"
    assert request.url.path == "/v1/chat/completions"
    assert len(serving.calls) == 1
    assert serving.clients[0].is_closed
    messages, options = serving.tokenize[0]
    assert messages == payload["messages"]
    assert options == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "truncation": False,
    }


@pytest.mark.parametrize("tokens,expected", [(6144, 200), (6145, 422)])
def test_context_budget_before_startup(serving, body, tokens, expected):
    from starlette.testclient import TestClient

    serving.control["tokens"] = tokens
    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == expected, response.text
    if expected == 422:
        serving.ensure.assert_not_called()
        assert not serving.calls
        assert not serving.touch


@pytest.mark.parametrize(
    "failure",
    [
        "length",
        "refusal",
        "empty",
        "malformed",
        "extra_states",
        "question",
        "source",
        "http",
        "timeout",
        "choices",
    ],
)
def test_failed_status_generation_is_502_and_records_activity(
    serving, body, output, failure
):
    from starlette.testclient import TestClient

    data = deepcopy(output)
    if failure == "length":
        serving.control["finish_reason"] = "length"
    elif failure == "refusal":
        serving.control["refusal"] = "Cannot comply"
    elif failure == "empty":
        serving.control["content"] = ""
    elif failure == "malformed":
        serving.control["content"] = "not JSON"
    elif failure == "extra_states":
        data["process_states"] = []
    elif failure == "question":
        data["outcomes"][0]["scope"]["question_id"] = "q99"
    elif failure == "source":
        data["outcomes"][0]["content_source_ids"] = ["missing"]
    elif failure == "http":
        serving.control["http_status"] = 500
    elif failure == "timeout":
        serving.control["error"] = httpx2.ReadTimeout("timeout")
    elif failure == "choices":
        serving.control["no_choices"] = True
    if failure in {"extra_states", "question", "source"}:
        serving.control["content"] = data
    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 502, response.text
    assert serving.touch == ["foi_status"]
    assert len(serving.calls) == 1
    assert serving.clients[0].is_closed


def test_status_transport_rejects_unattested_messages():
    with pytest.raises(TypeError, match="sanitization policy"):
        asyncio.run(
            response_model.classify_status(
                response_model.StatusMessages("system", "raw input"),
                model="test",
                deployment_url="https://unused/",
                api_key="unused",
                output_tokens=2048,
            )
        )


def test_tokenizer_failure_does_not_start_deployment(serving, body, monkeypatch):
    from starlette.testclient import TestClient

    def failed():
        raise RuntimeError("private details")

    monkeypatch.setattr(server.app.state.application.foi, "response_tokenizer", failed)
    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 503
    assert "private details" not in response.text
    serving.ensure.assert_not_called()


def test_canonical_input_is_a_valid_v4_record(body):
    body["request_text"] = None
    assert ExtractionInput.model_validate(body).model_dump(mode="json") == body


def test_changed_instruction_is_rejected_before_startup(serving, body, monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setattr(response_model, "SYSTEM_PROMPT_SHA256", "0" * 64)
    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 503
    serving.ensure.assert_not_called()
    assert not serving.calls


def test_startup_failure_records_activity(serving, body):
    from starlette.testclient import TestClient

    from llm_management.deployments.errors import LLMManagementError

    serving.ensure.side_effect = LLMManagementError("private provider details")
    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 503
    assert "private provider details" not in response.text
    assert serving.touch == ["foi_status"]
    assert not serving.calls


def test_missing_inference_credentials_are_503(serving, body):
    """
    Reject an unavailable connection before constructing a transport client.
    """
    from starlette.testclient import TestClient

    serving.ensure.return_value.state.api_key = ""
    with TestClient(server.app) as client:
        response = client.post("/agents/foi_response_analysis", json=body)
    assert response.status_code == 503
    assert serving.calls == []
    assert serving.touch == ["foi_status"]
