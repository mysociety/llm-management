from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.foi import backends
from llm_management.foi.question_slice import segment_request
from llm_management.inference import (
    ClassifierBusy,
)


from llm_management.foi import granite as foi_topic, pipeline as foi_pipeline
from llm_management.foi.model_spec import GRANITE_MERGED
from llm_management.foi.question_slice import build_extraction_result


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(server.settings, "cpu_inference_preload", False)
    monkeypatch.setattr(server.cache, "all_active", lambda: [])
    # The app lifespan must never manage real deployments in these tests.
    with TestClient(server.app, raise_server_exceptions=True) as test_client:
        yield test_client


@pytest.fixture
def classifier(monkeypatch):
    model = SimpleNamespace(
        model_name="question-slice-test",
        revision="pinned",
        batch_size=8,
        classify=lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
        validate_inputs=lambda texts: None,
    )
    monkeypatch.setattr(backends, "question_classifier", lambda: model)
    monkeypatch.setattr(
        backends,
        "question_classification_head",
        lambda: SimpleNamespace(load=lambda: None, classify=lambda vectors: vectors),
    )
    return model


def test_no_questions_skips_all_remote_work(client, classifier, monkeypatch):
    ensure = AsyncMock(side_effect=AssertionError("CPU must not provision a GPU"))
    monkeypatch.setattr(server, "ensure_running", ensure)
    response = client.post(
        "/agents/foi_structure/extract", json={"request": "Thank you."}
    )
    assert response.status_code == 200
    assert response.json()["extraction_status"] == "no_questions_found"
    assert response.json()["questions"] == []
    ensure.assert_not_called()


def test_orphan_keeps_indices_and_does_not_invent_start(client, classifier):
    classifier.classify = lambda texts: [
        [0.01, 0.01, 0.08, 0.9],
        [0.01, 0.01, 0.97, 0.01],
    ]
    response = client.post(
        "/agents/foi_structure/extract",
        json={"request": "Please provide the totals.\n\nPlease provide the reports."},
    )
    assert response.status_code == 200
    output = response.json()
    assert output["extraction_status"] == "uncertain"
    assert output["orphan_continuation_indices"] == [0]
    assert output["questions"][0]["unit_indices"] == [1]
    assert output["unit_predictions"][0]["confidence"] == 0.9


def test_gpu_disabled_does_not_provision(client, classifier, monkeypatch):
    monkeypatch.setattr(server.settings, "question_slice_gpu_enabled", False)
    ensure = AsyncMock()
    monkeypatch.setattr(server, "ensure_running", ensure)
    response = client.post(
        "/agents/foi_structure/extract?backend=exoscale",
        json={"request": "Please provide the report."},
    )
    assert response.status_code == 503
    ensure.assert_not_called()


def test_backend_switch_shares_reconstruction(client, classifier, monkeypatch):
    rows = [[0.0, 0.0, 1.0, 0.0]]
    classifier.classify = lambda texts: rows
    monkeypatch.setattr(server.settings, "question_slice_gpu_enabled", True)
    cfg = SimpleNamespace(model=classifier.model_name)
    state = SimpleNamespace(deployment_url="https://example.test/v1", api_key="secret")
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: cfg)
    monkeypatch.setattr(server, "ensure_running", AsyncMock(return_value=(cfg, state)))
    remote = AsyncMock(return_value=rows)
    monkeypatch.setattr(backends, "classify_remote", remote)
    results = []
    for backend in ["cpu", "exoscale"]:
        response = client.post(
            "/agents/foi_structure/extract",
            params={"backend": backend},
            json={"request": "Please provide the report."},
        )
        assert response.status_code == 200
        results.append(response.json())
    assert results[0]["questions"] == results[1]["questions"]
    assert results[0]["unit_predictions"] == results[1]["unit_predictions"]
    assert results[0]["extraction_revision"] == "pinned"
    assert results[1]["extraction_revision"] is None
    assert remote.call_args.kwargs["texts"] == [
        "[PREVIOUS] [NONE]\n[CURRENT] Please provide the report.\n[NEXT] [NONE]"
    ]


def test_overlength_rejected_before_gpu_start(client, classifier, monkeypatch):
    monkeypatch.setattr(server.settings, "question_slice_gpu_enabled", True)

    def reject(texts):
        raise ValueError("window exceeds 768 tokens; not truncated")

    classifier.validate_inputs = reject
    ensure = AsyncMock()
    monkeypatch.setattr(server, "ensure_running", ensure)
    response = client.post(
        "/agents/foi_structure/extract?backend=exoscale",
        json={"request": "Long request."},
    )
    assert response.status_code == 422
    ensure.assert_not_called()


@pytest.mark.parametrize(
    "rows",
    [[[0.0, 0.0, 1.0]], [[float("nan"), 0.0, 1.0, 0.0]], [[0.5, 0.5, 0.5, 0.5]], []],
)
def test_bad_probabilities_are_upstream_errors(client, classifier, rows):
    classifier.classify = lambda texts: rows
    response = client.post(
        "/agents/foi_structure/extract",
        json={"request": "Please provide the report."},
    )
    assert response.status_code == 502


def test_busy_cpu_returns_retryable_response(client, classifier):
    def busy(texts):
        raise ClassifierBusy("busy")

    classifier.classify = busy
    response = client.post(
        "/agents/foi_structure/extract",
        json={"request": "Please provide the report."},
    )
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"


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


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(server.settings, "cpu_inference_preload", False)
    monkeypatch.setattr(server.cache, "all_active", lambda: [])
    extractor = AsyncMock(return_value=extraction())
    monkeypatch.setattr(foi_pipeline, "extract_questions", extractor)
    monkeypatch.setattr(
        foi_topic, "prepare_topic_request", lambda *args: {"max_tokens": 256}
    )
    cfg = SimpleNamespace(model=GRANITE_MERGED)
    monkeypatch.setattr(server, "get_deployment_config", lambda slug: cfg)
    ensure = AsyncMock(
        return_value=(
            cfg,
            SimpleNamespace(deployment_url="https://test/v1", api_key="test"),
        )
    )
    classify = AsyncMock(
        return_value=foi_topic.TopicOutput.model_validate(topic_data())
    )
    monkeypatch.setattr(server, "ensure_running", ensure)
    monkeypatch.setattr(foi_topic, "classify_topics", classify)
    with TestClient(server.app) as client:
        yield SimpleNamespace(
            client=client,
            extractor=extractor,
            ensure=ensure,
            classify=classify,
            cfg=cfg,
        )


def test_new_route_orchestrates_and_keeps_source_diagnostics(pipeline):
    response = pipeline.client.post(
        "/agents/foi_structure?backend=exoscale",
        json={"request": "Please provide report 0."},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["classification"] == topic_data()
    assert result["questions"][0]["text"] == "Please provide report 0."
    assert result["promoted_continuation_index"] == 0
    assert result["unit_predictions"][0]["label"] == "QUESTION_CONTINUATION"
    assert result["classification_model"] == GRANITE_MERGED
    assert result["extraction_model"] == "modernbert"
    assert result["extraction_revision"] == "pinned"
    assert result["extraction_backend"] == "cpu"  # The mocked extractor's result.
    assert not {
        "keywords",
        "model",
        "revision",
        "backend",
        "granite_model",
        "granite_base_model",
        "granite_adapter",
    }.intersection(result)
    assert pipeline.extractor.call_args.kwargs["backend"] == "exoscale"
    assert pipeline.classify.call_args.kwargs["question_ids"] == ["q1"]


def test_no_questions_never_starts_granite(pipeline):
    pipeline.extractor.return_value = extraction((0,))
    response = pipeline.client.post(
        "/agents/foi_structure", json={"request": "Thank you."}
    )
    assert response.status_code == 200
    assert response.json()["classification"] is None
    assert response.json()["classification_model"] is None
    pipeline.ensure.assert_not_called()
    pipeline.classify.assert_not_called()


def test_partial_uncertainty_is_retained(pipeline):
    pipeline.extractor.return_value = extraction((3, 2))
    result = pipeline.client.post(
        "/agents/foi_structure", json={"request": "Request"}
    ).json()
    assert result["extraction_status"] == "uncertain"
    assert result["orphan_continuation_indices"] == [0]
    assert result["classification"] == topic_data()


def test_base_model_cannot_be_substituted(pipeline):
    pipeline.cfg.model = "ibm-granite/granite-4.0-1b"
    response = pipeline.client.post(
        "/agents/foi_structure", json={"request": "Request"}
    )
    assert response.status_code == 503
    pipeline.ensure.assert_not_called()


def test_prompt_rejection_precedes_provisioning(pipeline, monkeypatch):
    def reject(*args):
        raise ValueError("Input was not truncated")

    monkeypatch.setattr(foi_topic, "prepare_topic_request", reject)
    response = pipeline.client.post(
        "/agents/foi_structure", json={"request": "Request"}
    )
    assert response.status_code == 422
    pipeline.ensure.assert_not_called()


def test_bad_upstream_returns_502(pipeline):
    pipeline.classify.side_effect = foi_topic.TopicOutputError("invalid")
    assert (
        pipeline.client.post(
            "/agents/foi_structure", json={"request": "Request"}
        ).status_code
        == 502
    )


def test_old_flow_removed_from_api():
    paths = server.app.openapi()["paths"]
    assert "/agents/foi_structure/legacy" not in paths
    assert "/agents/foi_structure/legacy/extract" not in paths
    assert "/agents/foi_structure/metadata" not in paths
    assert "/agents/foi_structure/v2/extract" not in paths
    assert "/agents/foi_structure/extract" in paths
    assert "/agents/foi_structure" in paths
