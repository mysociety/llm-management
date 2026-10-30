"""Automatic detection routing, uncertainty, lazy loading and admission without downloads."""

import asyncio
import math
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from llm_management import server
from llm_management.automatic import pipeline as inference
from llm_management.automatic import resources as automatic_resources
from llm_management.deployments import get_catalog
from llm_management.inference import logistic_v1
from llm_management.inference.errors import (
    ClassifierBusy,
    ClassifierInputTooLong,
    ClassifierUnavailable,
)
from llm_management.inference.logistic_v1 import (
    BinaryTextLogisticModelV1,
    predict_binary,
)
from llm_management.inference.resources import ResourceRegistry


def policy(cutoff=automatic_resources.GATE_CUTOFF, name="gate"):
    return BinaryTextLogisticModelV1.model_validate(
        {
            "schema_version": "1",
            "name": name,
            "kind": "logistic",
            "true_label": "automatic",
            "false_label": "authored",
            "probability_threshold": cutoff,
            "inclusive": True,
            "logistic": {
                "schema_version": "1",
                "model_kind": "multilabel_logistic_regression",
                "input_kind": "prepared_text",
                "features": {
                    "unicode_version": "16.0.0",
                    "minimum_ngram": 1,
                    "maximum_ngram": 1,
                    "vocabulary": ["receipt"],
                },
                "classifiers": [
                    {"label": "automatic", "coefficients": [2.0], "intercept": -1.0}
                ],
                "provenance": {
                    "training_sha256": "0" * 64,
                    "sklearn_version": "test",
                    "training_documents": 1,
                    "config": {},
                },
            },
        }
    )


@pytest.fixture
def chain(monkeypatch):
    predictor = Mock(return_value=SimpleNamespace(result=True, positive_score=0.6))
    encoder = Mock()
    encoder.classify.return_value = [[0.5, 0.5]]
    monkeypatch.setattr(logistic_v1, "predict_binary", predictor)
    return (
        inference.AutoDetector(
            lambda body: logistic_v1.predict_binary(body, policy()), encoder
        ),
        predictor,
        encoder,
    )


def test_negative_skips_encoder(chain):
    detector, predictor, encoder = chain
    predictor.return_value = SimpleNamespace(result=False, positive_score=0.4)
    result = detector.classify("An authored reply")
    assert not result.is_automatic and not result.unresolved
    assert result.stage == "logistic" and result.encoder_score is None
    encoder.classify.assert_not_called()


@pytest.mark.parametrize("score,expected", [(0.4999, False), (0.5, True), (0.9, True)])
def test_confirmation_boundary_and_normalization(chain, score, expected):
    detector, predictor, encoder = chain
    encoder.classify.return_value = [[1 - score, score]]
    raw = "<NAME>  ＭＹ\nStraße Receipt"
    result = detector.classify(raw)
    assert result.is_automatic is expected and not result.unresolved
    assert result.stage == "deberta" and result.encoder_score == score
    predictor.assert_called_once_with(raw, policy())
    encoder.classify.assert_called_once_with(["my strasse receipt"])


@pytest.mark.parametrize("positive", [False, True])
def test_fast_skips_encoder_for_both_decisions(chain, positive):
    detector, predictor, encoder = chain
    detector.mode = "logistic"
    predictor.return_value = SimpleNamespace(result=positive, positive_score=0.6)
    result = detector.classify("receipt")
    assert result.is_automatic is positive and result.stage == "logistic"
    encoder.classify.assert_not_called()


def test_generic_predictor_uses_inclusive_policy_on_older_unicode():
    model = policy()
    score = predict_binary("receipt", model).positive_score
    boundary = model.model_copy(update={"probability_threshold": score})
    assert predict_binary("receipt", boundary).result
    assert not predict_binary(
        "receipt",
        boundary.model_copy(
            update={
                "probability_threshold": math.nextafter(score, 1.0),
            }
        ),
    ).result


@pytest.mark.parametrize("body", ["", " \n\t", "<NAME> <EMAIL>"])
def test_empty_is_unresolved(chain, body):
    detector, predictor, encoder = chain
    result = detector.classify(body)
    assert not result.is_automatic and result.unresolved
    assert result.stage == "none" and result.reason == "empty_body"
    predictor.assert_not_called()
    encoder.classify.assert_not_called()


@pytest.mark.parametrize(
    "error,reason",
    [
        (ClassifierBusy(), "encoder_busy"),
        (ClassifierUnavailable(), "encoder_unavailable"),
        (ClassifierInputTooLong(), "overlong_body"),
        (ValueError(), "encoder_failure"),
        (RuntimeError(), "encoder_failure"),
    ],
)
def test_encoder_failure_is_unresolved(chain, error, reason):
    detector, _, encoder = chain
    encoder.classify.side_effect = error
    result = detector.classify("receipt")
    assert not result.is_automatic and result.unresolved and result.reason == reason
    assert result.stage == "deberta" and result.logistic_score == 0.6


@pytest.mark.parametrize(
    "rows", [[], [[0.5]], [[0.1, 0.1]], [[0.0, float("nan")]], [[-1, 2]]]
)
def test_invalid_scores_are_unresolved(chain, rows):
    chain[2].classify.return_value = rows
    result = chain[0].classify("receipt")
    assert (
        not result.is_automatic
        and result.unresolved
        and result.reason == "invalid_score"
    )


def test_logistic_failure(chain):
    chain[1].side_effect = RuntimeError()
    result = chain[0].classify("receipt")
    assert (
        not result.is_automatic
        and result.unresolved
        and result.reason == "logistic_failure"
    )
    chain[2].classify.assert_not_called()


@pytest.mark.parametrize("count,accepted", [(512, True), (513, False)])
def test_encoder_never_truncates(monkeypatch, count, accepted):
    encoder = server.app.state.application.automatic.encoder_provider()
    tokenizer = Mock(return_value={"input_ids": [[1] * count]})
    monkeypatch.setattr(encoder, "_tokenizer", tokenizer)
    if accepted:
        encoder.validate_inputs(["receipt"])
    else:
        with pytest.raises(ClassifierInputTooLong):
            encoder.validate_inputs(["receipt"])
    tokenizer.assert_called_once_with(["receipt"], padding=False, truncation=False)


def test_loading_once_and_shared_weights(monkeypatch):
    import huggingface_hub

    gate, standalone = policy(), policy(name="standalone")
    loader = Mock(side_effect=[gate, standalone])
    download = Mock(return_value="/unused/policy.json")
    monkeypatch.setattr(automatic_resources, "load_model_v1", loader)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    config = get_catalog().local.get(
        get_catalog().automatic.logistic, automatic_resources.AutoLogisticDeployment
    )
    detector = automatic_resources.LogisticDetector(
        config=config, catalog=get_catalog(), registry=ResourceRegistry()
    )
    detector.resource.warmup()
    detector.resource.warmup()
    assert loader.call_count == 2
    assert detector.cascade.logistic is detector.standalone.logistic
    assert [c.args[1] for c in download.call_args_list] == [
        "model.json",
        "standalone-model.json",
    ]
    assert all(
        c.kwargs["revision"] == "9cff7b1018a61bd919fb049e3aca005952a101c7"
        for c in download.call_args_list
    )
    assert all(
        c.kwargs["expected_sha256"] == automatic_resources.POLICY_SHA256
        for c in loader.call_args_list
    )


def test_endpoint_contract_and_auth(chain, monkeypatch):
    detector, _, encoder = chain
    owner = Mock()
    owner.predict.side_effect = lambda body, fast=False: logistic_v1.predict_binary(
        body, policy()
    )
    monkeypatch.setattr(
        server.app.state.application.automatic, "logistic_provider", lambda: owner
    )
    monkeypatch.setattr(
        server.app.state.application.automatic, "encoder_provider", lambda: encoder
    )
    monkeypatch.setattr(server.settings, "auth_tokens", {})
    monkeypatch.setattr(
        server.app.state.application.deployments, "active_deployments", lambda: []
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/endpoint/foi/automatic_response_detection",
            json={"body": "receipt", "fast": True},
        )
        assert response.status_code == 200
        assert response.json() == {
            "is_automatic": True,
            "stage": "logistic",
            "unresolved": False,
            "reason": None,
            "logistic_score": 0.6,
            "encoder_score": None,
        }
        encoder.classify.assert_not_called()
        assert (
            client.post(
                "/endpoint/foi/automatic_response_detection", json={"body": ["receipt"]}
            ).status_code
            == 422
        )
        monkeypatch.setattr(server.settings, "auth_tokens", {"test": "secret"})
        assert (
            client.post(
                "/endpoint/foi/automatic_response_detection", json={"body": "receipt"}
            ).status_code
            == 401
        )


def test_bounded_admission_survives_cancellation(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def classify(body, fast):
        entered.set()
        assert release.wait(5)
        finished.set()
        return inference.AutomaticResult(is_automatic=False, stage="logistic")

    capacity = asyncio.Semaphore(1)
    monkeypatch.setattr(server.app.state.application.automatic, "capacity", capacity)
    monkeypatch.setattr(
        server.app.state.application.automatic, "classify_gate", classify
    )

    async def scenario():
        task = asyncio.create_task(
            server.app.state.application.automatic.detect("receipt")
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            waiting = asyncio.create_task(
                server.app.state.application.automatic.detect("another")
            )
            await asyncio.sleep(0.01)
            assert not waiting.done()
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        assert not (await waiting).unresolved
        # Yield to let the worker release its lease before the next admission.
        await asyncio.sleep(0.01)
        assert not (
            await server.app.state.application.automatic.detect("last")
        ).unresolved

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "update",
    [
        {"probability_threshold": 0.5},
        {"inclusive": False},
        {"true_label": "sar"},
    ],
)
def test_unexpected_gate_policy_fails_loading(monkeypatch, update):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: "/unused")
    monkeypatch.setattr(
        automatic_resources,
        "load_model_v1",
        Mock(side_effect=[policy().model_copy(update=update), policy()]),
    )
    config = get_catalog().local.get(
        get_catalog().automatic.logistic, automatic_resources.AutoLogisticDeployment
    )
    detector = automatic_resources.LogisticDetector(
        config=config, catalog=get_catalog(), registry=ResourceRegistry()
    )
    with pytest.raises(ClassifierUnavailable):
        detector.load()
    assert detector.cascade is None and detector.standalone is None


def test_generic_loader_rejects_checksum_mismatch(tmp_path):
    from llm_management.inference.logistic_v1 import load_model_v1

    artifact = tmp_path / "model.json"
    artifact.write_text(policy().model_dump_json())
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_v1(artifact, expected_sha256="0" * 64)


def test_api_startup_does_not_load_automatic_models(monkeypatch):
    logistic, encoder = Mock(), Mock()
    monkeypatch.setattr(
        server.app.state.application.automatic, "logistic_provider", logistic
    )
    monkeypatch.setattr(
        server.app.state.application.automatic, "encoder_provider", encoder
    )
    monkeypatch.setattr(
        server.app.state.application.deployments, "active_deployments", lambda: []
    )
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
    logistic.assert_not_called()
    encoder.assert_not_called()


def test_lazy_modes_share_policy_and_reload_after_idle_expiry(monkeypatch):
    import huggingface_hub

    registry = ResourceRegistry()
    download = Mock(return_value="/unused/policy.json")
    loader = Mock(
        side_effect=[policy(), policy(cutoff=0.9), policy(), policy(cutoff=0.9)]
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(automatic_resources, "load_model_v1", loader)
    config = get_catalog().local.get(
        get_catalog().automatic.logistic, automatic_resources.AutoLogisticDeployment
    )
    logistic = automatic_resources.LogisticDetector(
        config=config, catalog=get_catalog(), registry=registry
    )
    monkeypatch.setattr(
        server.app.state.application.automatic, "logistic_provider", lambda: logistic
    )
    encoder = Mock()
    encoder.classify.return_value = [[0.1, 0.9]]
    monkeypatch.setattr(
        server.app.state.application.automatic, "encoder_provider", lambda: encoder
    )
    fast = server.app.state.application.automatic.detector(True)
    cascade = server.app.state.application.automatic.detector(False)
    loader.assert_not_called()
    assert not fast.classify("receipt").is_automatic
    assert cascade.classify("receipt").is_automatic
    assert logistic.cascade.logistic is logistic.standalone.logistic
    assert loader.call_count == 2
    registry.expire(0)
    assert logistic.cascade is None and logistic.standalone is None
    assert not logistic.resource.status().ready
    # Cached mode controllers must not hold stale model references.
    assert not fast.classify("receipt").is_automatic
    assert loader.call_count == 4
    assert cascade.classify("receipt").is_automatic
    assert loader.call_count == 4


def test_fast_and_logistic_negative_do_not_load_encoder(monkeypatch):
    logistic = Mock()
    logistic.predict.return_value = SimpleNamespace(result=False, positive_score=0.2)
    encoder = Mock()
    monkeypatch.setattr(
        server.app.state.application.automatic, "logistic_provider", lambda: logistic
    )
    factory = Mock(return_value=encoder)
    monkeypatch.setattr(
        server.app.state.application.automatic, "encoder_provider", factory
    )
    assert not server.app.state.application.automatic.classify(
        "authored", True
    ).is_automatic
    factory.assert_not_called()
    assert not server.app.state.application.automatic.classify(
        "authored", False
    ).is_automatic
    encoder.classify.assert_not_called()
    encoder.resource.warmup.assert_not_called()
    assert logistic.predict.call_count == 2


def test_lazy_logistic_loading_failure_is_unresolved(monkeypatch):
    logistic = Mock()
    logistic.predict.side_effect = ClassifierUnavailable()
    monkeypatch.setattr(
        server.app.state.application.automatic, "logistic_provider", lambda: logistic
    )
    result = asyncio.run(
        server.app.state.application.automatic.detect("receipt", fast=True)
    )
    assert not result.is_automatic and result.unresolved
    assert result.stage == "logistic" and result.reason == "logistic_failure"


def test_app_warmup_inference_and_expiry_share_one_logistic_owner(monkeypatch):
    import huggingface_hub

    application = server.app.state.application
    service = application.deployments
    config = service.catalog.automatic
    download = Mock(return_value="/unused/policy.json")
    loader = Mock(side_effect=[policy(), policy(), policy(), policy()])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(automatic_resources, "load_model_v1", loader)
    encoder = Mock(
        side_effect=AssertionError("Fast mode must not construct an encoder")
    )
    monkeypatch.setattr(application.automatic, "encoder_provider", encoder)
    monkeypatch.setattr(server.settings, "auth_tokens", {})

    with TestClient(server.app) as client:
        assert client.post(f"/deployments/{config.logistic}/warm-up").status_code == 200
        assert loader.call_count == 2
        owner = application.automatic.logistic_provider()
        assert owner.resource is service.resources.get(config.logistic)
        result = client.post(
            "/endpoint/foi/automatic_response_detection",
            json={"body": "receipt", "fast": True},
        )
        assert result.json()["is_automatic"]
        assert loader.call_count == 2
        service.resources.expire(0)
        assert owner.cascade is None
        result = client.post(
            "/endpoint/foi/automatic_response_detection",
            json={"body": "receipt", "fast": True},
        )
        assert result.json()["is_automatic"]
        assert loader.call_count == 4
        assert application.automatic.logistic_provider() is owner
        encoder.assert_not_called()


@pytest.mark.parametrize("fail_encoder", [False, True])
def test_warmup_group_reports_required_models(monkeypatch, fail_encoder):
    from llm_management.inference.resources import LocalResource

    registry = ResourceRegistry()
    pipeline = get_catalog().automatic
    logistic_load, encoder_load = Mock(), Mock()
    if fail_encoder:
        encoder_load.side_effect = ClassifierUnavailable("Missing model")
    registry.register(LocalResource(pipeline.logistic, logistic_load, Mock()))
    registry.register(LocalResource(pipeline.classifier, encoder_load, Mock()))
    monkeypatch.setattr(server.app.state.application.deployments, "resources", registry)
    result = asyncio.run(
        server.app.state.application.deployments.ensure_group(
            "group_automatic_pipeline_cpu"
        )
    )
    assert result.success is (not fail_encoder)
    assert [r.slug for r in result.deployments] == [
        pipeline.logistic,
        pipeline.classifier,
    ]
    assert result.deployments[0].success
    assert result.deployments[1].success is (not fail_encoder)
    logistic_load.assert_called_once()
    encoder_load.assert_called_once()
