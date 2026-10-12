"""SAR routing and fail-closed behavior, without private model downloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.deployments import get_catalog
from llm_management.inference.errors import ClassifierBusy, ClassifierUnavailable
from llm_management.inference.logistic_v1 import (
    normalize_text_v1,
)
from llm_management.sar import pipeline as sar
from llm_management.sar import resources as sar_resources
from llm_management.settings import Settings


@pytest.fixture
def chain(monkeypatch):
    logistic = Mock()
    logistic.predict.return_value = SimpleNamespace(result=True, positive_score=0.01)
    deberta = Mock()
    deberta.classify.return_value = [[0.5, 0.5]]
    monkeypatch.setattr(sar, "logistic_detector", lambda: logistic)
    monkeypatch.setattr(sar, "deberta_detector", lambda: deberta)
    monkeypatch.setattr(
        server.app.state.application.sar, "logistic_provider", lambda: logistic
    )
    monkeypatch.setattr(
        server.app.state.application.sar, "encoder_provider", lambda: deberta
    )
    return logistic, deberta


def test_logistic_negative_skips_deberta(chain):
    logistic, deberta = chain
    logistic.predict.return_value = SimpleNamespace(result=False, positive_score=0.001)
    result = sar._detect_sar("Original correspondence")
    assert not result.is_sar and result.status == "complete"
    logistic.predict.assert_called_once_with("Original correspondence")
    deberta.classify.assert_not_called()


@pytest.mark.parametrize(
    "positive,expected", [(0.4999, False), (0.5, True), (0.9, True)]
)
def test_positive_routes_normalized_whole_text(chain, positive, expected):
    logistic, deberta = chain
    raw = "<NAME>  ＭＹ\nStraße Records"
    deberta.classify.return_value = [[1 - positive, positive]]
    result = sar._detect_sar(raw)
    assert result.is_sar is expected and result.status == "complete"
    logistic.predict.assert_called_once_with(raw)
    deberta.classify.assert_called_once_with(["my strasse records"])


@pytest.mark.parametrize(
    "error,reason",
    [
        (ClassifierBusy(), "deberta_busy"),
        (ClassifierUnavailable(), "deberta_unavailable"),
        (ValueError("513 tokens"), "deberta_input_unsupported"),
        (RuntimeError(), "inference_failed"),
    ],
)
def test_deberta_failure_keeps_flag(chain, error, reason):
    chain[1].classify.side_effect = error
    result = sar._detect_sar("personal records")
    assert result.is_sar and result.status == "unclear" and result.reason == reason
    assert result.logistic_score == 0.01 and result.deberta_score is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0.2]],
        [[0.1, float("nan")]],
        [[0.1, float("inf")]],
        [[0.1, 0.1]],
        [[-1, 2]],
    ],
)
def test_invalid_deberta_scores_keep_flag(chain, rows):
    chain[1].classify.return_value = rows
    result = sar._detect_sar("records")
    assert result.is_sar and result.reason == "invalid_score"


def test_logistic_failure_keeps_flag(chain):
    chain[0].predict.side_effect = RuntimeError()
    result = sar._detect_sar("records")
    assert result.is_sar and result.status == "unclear"
    chain[1].classify.assert_not_called()


def test_token_aliases(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "new")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "old")
    assert Settings(_env_file=None).huggingface_token == "new"
    monkeypatch.delenv("HF_TOKEN")
    assert Settings(_env_file=None).huggingface_token == "old"


def test_resource_registration_and_warmup_group():
    from llm_management.deployments import DeploymentCatalog

    config = DeploymentCatalog.load()
    assert {
        get_catalog().require_sar().logistic,
        get_catalog().require_sar().classifier,
    } <= server.app.state.service.resources.names()
    assert config.get_group("sar_pipeline_cpu").deployments == [
        get_catalog().require_sar().logistic,
        get_catalog().require_sar().classifier,
    ]


def test_endpoint_conservative_response(chain, monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(server.app.state.service.cache, "all_active", lambda: [])
    chain[1].classify.side_effect = ClassifierBusy()
    with TestClient(server.app) as client:
        response = client.post("/agents/sar_detection", json={"request": "my records"})
    assert response.status_code == 200
    assert response.json()["is_sar"] is True
    assert response.json()["reason"] == "deberta_busy"


@pytest.mark.parametrize("token_count,accepted", [(512, True), (513, False)])
def test_deberta_token_budget_without_truncation(monkeypatch, token_count, accepted):
    model = sar.deberta_detector()
    tokenizer = Mock(return_value={"input_ids": [[1] * token_count]})
    monkeypatch.setattr(model, "_tokenizer", tokenizer)
    if accepted:
        model.validate_inputs(["prepared text"])
    else:
        with pytest.raises(ValueError, match="Input was not truncated"):
            model.validate_inputs(["prepared text"])
    tokenizer.assert_called_once_with(
        ["prepared text"], padding=False, truncation=False
    )


def test_logistic_load_once_and_idle_reload(monkeypatch):
    import huggingface_hub

    artifact = SimpleNamespace(
        true_label="sar",
        false_label="not-sar",
        probability_threshold=sar_resources.LOGISTIC_THRESHOLD,
        inclusive=True,
        logistic=SimpleNamespace(features=SimpleNamespace(unicode_version="16.0.0")),
    )
    loader = Mock(return_value=artifact)
    download = Mock(return_value="/unused/model.json")
    monkeypatch.setattr(sar_resources, "load_model_v1", loader)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    # Use an isolated controller without replacing the application's registry entry.
    detector = sar_resources.LogisticDetector()
    detector.resource.warmup()
    detector.resource.warmup()
    loader.assert_called_once()
    assert detector.resource.status().ready
    assert (
        download.call_args.kwargs["revision"]
        == get_catalog().model["sar_logistic"].revision
    )
    assert detector.resource.release()
    assert not detector.resource.status().ready
    detector.resource.warmup()
    assert loader.call_count == 2
    detector.resource.release()


def test_unexpected_logistic_cutoff_is_rejected(monkeypatch):
    import huggingface_hub

    artifact = SimpleNamespace(
        true_label="sar",
        false_label="not-sar",
        probability_threshold=0.5,
        inclusive=True,
        logistic=SimpleNamespace(features=SimpleNamespace(unicode_version="16.0.0")),
    )
    monkeypatch.setattr(sar_resources, "load_model_v1", lambda path: artifact)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: "/unused")
    detector = sar_resources.LogisticDetector()
    with pytest.raises(ClassifierUnavailable, match="artifact policy"):
        detector.resource.warmup()
    assert detector._model is None
    assert not detector.resource.status().ready


def test_endpoint_requires_authentication(monkeypatch):
    monkeypatch.setattr(server.settings, "auth_tokens", {"test": "secret"})
    monkeypatch.setattr(server.app.state.service.cache, "all_active", lambda: [])
    with TestClient(server.app) as client:
        assert (
            client.post(
                "/agents/sar_detection", json={"request": "records"}
            ).status_code
            == 401
        )


def test_tokenizer_loads_authenticated_snapshot_locally(monkeypatch):
    import huggingface_hub
    import transformers

    model = sar_resources.SARSequenceClassifier(
        model=get_catalog().model["sar_deberta"].repo,
        revision=get_catalog().model["sar_deberta"].revision,
        labels=("not-sar", "sar"),
        token="private-test-token",
    )
    download = Mock(return_value="/local/pinned/snapshot")
    tokenizer = Mock(return_value={"input_ids": [[1, 42, 2]]})
    loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", loader)
    assert model._tokenize(["records"])["input_ids"] == [[1, 42, 2]]
    assert download.call_args.kwargs["token"] == "private-test-token"
    assert (
        download.call_args.kwargs["revision"]
        == get_catalog().model["sar_deberta"].revision
    )
    loader.assert_called_once_with(
        "/local/pinned/snapshot",
        local_files_only=True,
        fix_mistral_regex=False,
        extra_special_tokens={},
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<NAME> ＭＹ\nStraße Records", "my strasse records"),
        ("Jose\u0301, Zoe\u0308", "josé, zoë"),
        (
            "İstanbul I ı İ i Straße STRASSE ﬃ K",
            "i\u0307stanbul i ı i\u0307 i strasse strasse ffi k",
        ),
        (
            "Records\u00a0for\u2003me\u202fplease\n\tcase notes",
            "records for me please case notes",
        ),
        ("My records 😀 👩🏽‍💻 🇬🇧", "my records 😀 👩🏽‍💻 🇬🇧"),
    ],
)
def test_normalization_on_correspondence_characters(raw, expected):
    assert normalize_text_v1(raw) == expected


@pytest.mark.parametrize("positive,score", [(False, 0.001), (True, 0.01)])
def test_fast_returns_logistic_decision_without_deberta(chain, positive, score):
    logistic, deberta = chain
    logistic.predict.return_value = SimpleNamespace(
        result=positive, positive_score=score
    )
    result = sar._detect_sar("Original correspondence", fast=True)
    assert result.is_sar is positive and result.status == "complete"
    assert result.stage == "logistic" and result.logistic_score == score
    assert result.deberta_score is None and result.reason is None
    logistic.predict.assert_called_once_with("Original correspondence")
    deberta.classify.assert_not_called()


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 2])
def test_fast_invalid_logistic_score_keeps_unclear_flag(chain, score):
    chain[0].predict.return_value = SimpleNamespace(result=False, positive_score=score)
    result = sar._detect_sar("records", fast=True)
    assert result.is_sar and result.status == "unclear"
    assert result.stage == "logistic" and result.reason == "invalid_score"
    assert result.logistic_score is None and result.deberta_score is None
    chain[1].classify.assert_not_called()


def test_fast_logistic_loading_failure_keeps_unclear_flag(chain):
    chain[0].predict.side_effect = ClassifierUnavailable()
    result = sar._detect_sar("records", fast=True)
    assert result.is_sar and result.status == "unclear"
    assert result.stage == "logistic" and result.reason == "logistic_unavailable"
    chain[1].classify.assert_not_called()


@pytest.mark.parametrize("fast", [None, False, True])
def test_endpoint_fast_parameter_and_stage(chain, monkeypatch, fast):
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(server.app.state.service.cache, "all_active", lambda: [])
    payload = {"request": "my records"}
    if fast is not None:
        payload["fast"] = fast
    # Disagreement demonstrates that only fast mode skips confirmation.
    chain[1].classify.return_value = [[0.9, 0.1]]
    with TestClient(server.app) as client:
        response = client.post("/agents/sar_detection", json=payload)
    assert response.status_code == 200
    assert response.json()["is_sar"] is bool(fast)
    assert response.json()["stage"] == ("logistic" if fast else "deberta")
    if fast:
        chain[1].classify.assert_not_called()
        assert response.json()["deberta_score"] is None
    else:
        chain[1].classify.assert_called_once()
