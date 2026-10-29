"""
Offline scoring contracts, data integrity, resource ownership and enrichment.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import numpy as np
import pytest

from llm_management.bootstrap import assemble_application
from llm_management.deployments.catalog import (
    DeploymentCatalog,
)
from llm_management.foi import category_classification as categories
from llm_management.foi.category_index import CategoryIndex
from llm_management.foi.category_schemas import QuestionCategory
from llm_management.foi.schemas import TopicOutput
from llm_management.inference.errors import ClassifierOutputError, ClassifierUnavailable
from llm_management.inference.resources import ResourceRegistry


def vector(cosine: float) -> list[float]:
    return [cosine, float(np.sqrt(1 - cosine**2)), *([0.0] * 382)]


def scoring_index(centroids: list[float], descriptions: list[float]) -> CategoryIndex:
    """
    Build a complete validated taxonomy with unused categories scoring below candidates.
    """
    count = len(centroids)
    centroids = centroids + [-1.0] * (181 - count)
    descriptions = descriptions + [-1.0] * (181 - count)
    return CategoryIndex.model_validate(
        {
            "format_version": 2,
            "model_name": "test-index",
            "embedding_model": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "dimensions": 384,
                "pooling": "attention-mask mean",
                "normalization": "L2",
                "max_wordpieces": 256,
            },
            "category_rule": {
                "score": "0.5 * cosine(topic, centroid_embedding) + 0.5 * cosine(topic, description_embedding)",
                "selection": "highest score among 181 categories",
            },
            "no_match": {
                "score": "maximum cosine similarity to any category centroid",
                "comparison": "less_than",
                "threshold": 0.5,
                "output_category_slug": None,
            },
            "categories": [
                {
                    "label_id": i,
                    "slug": f"category_{i}",
                    "centroid_embedding": vector(centroid),
                    "description_embedding": vector(description),
                }
                for i, (centroid, description) in enumerate(
                    zip(centroids, descriptions, strict=True)
                )
            ],
        }
    )


@pytest.mark.parametrize(
    "maximum,rejected", [(0.4999, True), (0.5, False), (0.5001, False)]
)
def test_abstention_uses_strict_centroid_boundary(maximum, rejected):
    result = categories.predict_vectors(scoring_index([maximum], [1]), [vector(1)])[0]
    assert result == (
        None if rejected else QuestionCategory(category_slug="category_0")
    )


def test_blended_winner_differs_from_centroid_winner_and_uses_global_gate():
    index = scoring_index([0.8, 0.4], [-1, 1])
    result = categories.predict_vectors(index, [vector(1)])[0]
    assert result.category_slug == "category_1"


@pytest.mark.parametrize(
    "centroids,descriptions,winner",
    [([0.9, 0.6], [0.4, 0.8], "category_1"), ([0.9, 0.4], [0.4, 0.8], "category_0")],
)
def test_ranking_weights_centroids_and_descriptions_equally(
    centroids, descriptions, winner
):
    result = categories.predict_vectors(
        scoring_index(centroids, descriptions), [vector(1)]
    )[0]
    assert result == QuestionCategory(category_slug=winner)


def test_rejected_result_is_null_and_ties_use_label_order():
    result = categories.predict_vectors(scoring_index([0.3, 0.4], [1, 1]), [vector(1)])[
        0
    ]
    assert result is None
    tied = categories.predict_vectors(scoring_index([0.8, 0.8], [1, 1]), [vector(1)])[0]
    assert tied.category_slug == "category_0"


@pytest.mark.parametrize(
    "values",
    [
        [[0.0] * 384],
        [[float("nan")] * 384],
        [[float("inf")] * 384],
        [[1.0] * 384],
        [[1.0] * 383],
        [1.0] * 384,
    ],
)
def test_invalid_encoder_vectors_rejected(values):
    with pytest.raises(ClassifierOutputError):
        categories.predict_vectors(scoring_index([1], [1]), values)


@pytest.fixture
def matcher():
    application = assemble_application()
    return application.foi.category_matcher()


@pytest.fixture
def valid_index():
    """
    Return wire data so integrity tests can deliberately corrupt individual fields.
    """
    return scoring_index([1.0] * 181, [1.0] * 181).model_dump()


class WrittenIndex(NamedTuple):
    """
    Paths and trusted checksum for a temporary artifact pair.
    """

    index_path: Path
    manifest_path: Path
    checksum: str


def write_index(tmp_path, index) -> WrittenIndex:
    """
    Write index wire data and the manifest fields used by inference.
    """
    raw = json.dumps(index).encode()
    digest = hashlib.sha256(raw).hexdigest()
    index_path, manifest_path = tmp_path / "index.json", tmp_path / "manifest.json"
    index_path.write_bytes(raw)
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "model_name": "test-index",
                "index_sha256": digest,
            }
        )
    )
    return WrittenIndex(
        index_path=index_path, manifest_path=manifest_path, checksum=digest
    )


def test_index_checksum_is_anchored_in_config_not_just_manifest(
    tmp_path, valid_index, matcher
):
    paths = write_index(tmp_path, valid_index)
    assert categories.load_index(
        paths.index_path,
        paths.manifest_path,
        checksum=paths.checksum,
        encoder=matcher.encoder,
    ) == CategoryIndex.model_validate(valid_index)
    with pytest.raises(ValueError, match="checksum"):
        categories.load_index(
            paths.index_path,
            paths.manifest_path,
            checksum="0" * 64,
            encoder=matcher.encoder,
        )
    paths.index_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        categories.load_index(
            paths.index_path,
            paths.manifest_path,
            checksum=paths.checksum,
            encoder=matcher.encoder,
        )


def test_unused_metadata_is_ignored_but_still_covered_by_checksum(
    tmp_path, valid_index, matcher
):
    expected = CategoryIndex.model_validate(valid_index)
    valid_index.update(groups="unused", source={"research": None}, description=123)
    valid_index["categories"][0].update(
        name=None, group_slugs="unused", description_text=123
    )
    valid_index["embedding_model"]["research_note"] = {"unvalidated": True}
    artifact = write_index(tmp_path, valid_index)
    manifest = json.loads(artifact.manifest_path.read_text())
    manifest.update(evaluation="unused", source_index_sha256=None)
    artifact.manifest_path.write_text(json.dumps(manifest))
    assert (
        categories.load_index(
            artifact.index_path,
            artifact.manifest_path,
            checksum=artifact.checksum,
            encoder=matcher.encoder,
        )
        == expected
    )
    valid_index["source"] = {"research": "changed"}
    artifact.index_path.write_text(json.dumps(valid_index))
    with pytest.raises(ValueError, match="checksum"):
        categories.load_index(
            artifact.index_path,
            artifact.manifest_path,
            checksum=artifact.checksum,
            encoder=matcher.encoder,
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "order",
        "duplicate_slug",
        "dimensions",
        "nonfinite",
        "unnormalized",
        "encoder",
        "threshold",
        "ranking",
        "category_count",
        "blank_slug",
        "numeric_coercion",
    ],
)
def test_index_contract_corruption_rejected(tmp_path, valid_index, matcher, corruption):
    index = copy.deepcopy(valid_index)
    row = index["categories"][0]
    if corruption == "order":
        row["label_id"] = 1
    elif corruption == "duplicate_slug":
        row["slug"] = "category_1"
    elif corruption == "dimensions":
        row["centroid_embedding"] = [1]
    elif corruption == "nonfinite":
        row["description_embedding"][0] = float("nan")
    elif corruption == "unnormalized":
        row["description_embedding"] = [0] * 384
    elif corruption == "encoder":
        index["embedding_model"]["revision"] = "0" * 40
    elif corruption == "threshold":
        index["no_match"]["threshold"] = 0.4
    elif corruption == "category_count":
        index["categories"].pop()
    elif corruption == "blank_slug":
        row["slug"] = " "
    elif corruption == "numeric_coercion":
        row["centroid_embedding"][0] = "1.0"
    else:
        index["category_rule"]["score"] = "centroid only"
    index_path, manifest_path, digest = write_index(tmp_path, index)
    with pytest.raises(ValueError):
        categories.load_index(
            index_path, manifest_path, checksum=digest, encoder=matcher.encoder
        )


@pytest.mark.parametrize("corruption", ["checksum", "identity", "schema"])
def test_manifest_must_match_trusted_index(tmp_path, valid_index, matcher, corruption):
    artifact = write_index(tmp_path, valid_index)
    manifest = json.loads(artifact.manifest_path.read_text())
    if corruption == "checksum":
        manifest["index_sha256"] = "0" * 64
    elif corruption == "identity":
        manifest["model_name"] = "different-index"
    else:
        manifest["index_sha256"] = 123
    artifact.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        categories.load_index(
            artifact.index_path,
            artifact.manifest_path,
            checksum=artifact.checksum,
            encoder=matcher.encoder,
        )


def test_downloads_only_pinned_data_and_safetensors_and_caches_load(
    tmp_path, valid_index, matcher, monkeypatch
):
    import huggingface_hub
    import transformers

    artifact = write_index(tmp_path, valid_index)
    matcher.checksum = artifact.checksum
    snapshot = Mock(return_value=str(tmp_path))
    tokenizer = Mock(return_value=object())
    model = Mock(
        return_value=SimpleNamespace(
            eval=lambda: SimpleNamespace(config=SimpleNamespace(hidden_size=384))
        )
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", model)
    matcher.resource.warmup()
    matcher.load()
    assert snapshot.call_count == 2
    index_call, encoder_call = snapshot.call_args_list
    assert index_call.kwargs["repo_id"] == matcher.checkpoint.repo
    assert index_call.kwargs["revision"] == matcher.checkpoint.revision
    assert index_call.kwargs["allow_patterns"] == ["index.json", "manifest.json"]
    allowed = encoder_call.kwargs["allow_patterns"]
    assert "model.safetensors" in allowed
    assert not any(name.endswith((".py", ".bin")) or "*" in name for name in allowed)
    assert encoder_call.kwargs["repo_id"] == matcher.encoder.repo
    assert encoder_call.kwargs["revision"] == matcher.encoder.revision
    for call in (tokenizer, model):
        assert call.call_args.kwargs["trust_remote_code"] is False
        assert call.call_args.kwargs["local_files_only"] is True
    assert model.call_args.kwargs["use_safetensors"] is True
    assert matcher.resource.status().ready
    matcher.resource.stop()
    assert matcher.loaded is None


def test_corrupt_index_stops_before_encoder_download(
    tmp_path, valid_index, matcher, monkeypatch
):
    import huggingface_hub

    write_index(tmp_path, valid_index)
    snapshot = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    with pytest.raises(ClassifierUnavailable):
        matcher.load()
    snapshot.assert_called_once()
    assert snapshot.call_args.kwargs["repo_id"] == matcher.checkpoint.repo
    assert matcher.loaded is None
    assert not matcher.resource.status().ready


def test_encoder_masks_padding_normalizes_and_batches(matcher):
    import torch

    index = scoring_index([1], [1])
    batches = []

    def tokenizer(texts, **kwargs):
        batches.append((texts, kwargs))
        return {"attention_mask": torch.tensor([[1, 0]] * len(texts))}

    def model(attention_mask):
        hidden = torch.zeros((len(attention_mask), 2, 384))
        hidden[:, 0, 0] = 2
        hidden[:, 1, 1] = 999
        return SimpleNamespace(last_hidden_state=hidden)

    matcher.loaded = categories.LoadedCategoryMatcher(
        category_index=index,
        tokenizer=tokenizer,
        model=model,
        matrices=categories.category_matrices(index),
    )
    results = matcher.predict([f"topic {i}" for i in range(17)])
    assert len(results) == 17
    assert all(
        result == QuestionCategory(category_slug="category_0") for result in results
    )
    assert [len(batch[0]) for batch in batches] == [16, 1]
    assert batches[0][1] == {
        "padding": True,
        "truncation": True,
        "max_length": 256,
        "return_tensors": "pt",
    }
    with pytest.raises(ValueError):
        matcher.predict([" "])


def topic_output(regimes=("FOI", "EIR")):
    return TopicOutput(
        questions=[
            {"question_id": f"q{i + 1}", "regime": regime, "topic": f"topic {i}"}
            for i, regime in enumerate(regimes)
        ],
    )


def matched(slug="air_quality"):
    return QuestionCategory(category_slug=slug)


def test_question_topics_are_batched_and_assignments_belong_to_questions(matcher):
    matcher.predict = Mock(side_effect=lambda texts: [matched() for _ in texts])
    topics = topic_output()
    result = asyncio.run(categories.classify_categories(topics, lambda: matcher))
    matcher.predict.assert_called_once_with(["topic 0", "topic 1"])
    assert result.status == "completed"
    assert [q.category.model_dump() for q in topics.questions] == [
        {"category_type": "foi_v1", "category_slug": "air_quality"}
    ] * 2
    assert result.provenance == matcher.provenance


def test_duplicate_topics_fill_each_question_without_repeating_inference(matcher):
    topics = topic_output()
    original_questions = list(topics.questions)
    assert all(q.category is None for q in topics.questions)
    topics.questions[1].topic = topics.questions[0].topic
    matcher.predict = Mock(return_value=[matched()])
    asyncio.run(categories.classify_categories(topics, lambda: matcher))
    matcher.predict.assert_called_once_with(["topic 0"])
    assert all(q.category.category_type == "foi_v1" for q in topics.questions)
    assert all(q.category.category_slug == "air_quality" for q in topics.questions)
    assert all(
        before is after
        for before, after in zip(original_questions, topics.questions, strict=True)
    )


def test_incomplete_batch_never_partially_assigns_categories(matcher):
    topics = topic_output()
    matcher.predict = Mock(return_value=[matched()])
    result = asyncio.run(categories.classify_categories(topics, lambda: matcher))
    assert result.status == "unavailable"
    assert all(q.category is None for q in topics.questions)


def test_sar_is_skipped_and_mixed_requests_classify_only_eligible_questions(matcher):
    provider = Mock(return_value=matcher)
    result = asyncio.run(
        categories.classify_categories(topic_output(("SAR",)), provider)
    )
    assert result.status == "not_applicable"
    provider.assert_not_called()
    matcher.predict = Mock(return_value=[matched()])
    topics = topic_output(("SAR", "EIR"))
    mixed = asyncio.run(categories.classify_categories(topics, provider))
    matcher.predict.assert_called_once_with(["topic 1"])
    assert mixed.status == "completed"
    assert topics.questions[0].category is None
    assert topics.questions[1].category.category_slug == "air_quality"


def test_no_match_is_completed_and_operational_failure_is_unavailable(matcher):
    matcher.predict = Mock(side_effect=lambda texts: [None for _ in texts])
    topics = topic_output()
    result = asyncio.run(categories.classify_categories(topics, lambda: matcher))
    assert result.status == "completed"
    assert all(q.category is None for q in topics.questions)
    matcher.predict.side_effect = RuntimeError("private failure")
    topics = topic_output()
    result = asyncio.run(categories.classify_categories(topics, lambda: matcher))
    assert result.status == "unavailable"
    assert all(q.category is None for q in topics.questions)
    assert "private failure" not in result.model_dump_json()


def test_slow_encoder_does_not_block_event_loop(matcher):
    entered, release = threading.Event(), threading.Event()

    def predict(texts):
        entered.set()
        assert release.wait(5)
        return [matched() for _ in texts]

    matcher.predict = predict

    async def run():
        pending = asyncio.create_task(
            categories.classify_categories(topic_output(), lambda: matcher)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            await asyncio.sleep(0)
            assert not pending.done()
        finally:
            release.set()
        assert (await pending).status == "completed"

    asyncio.run(run())


def test_cold_matchers_are_owned_by_their_applications():
    first_registry, second_registry = ResourceRegistry(), ResourceRegistry()
    first = assemble_application(registry=first_registry)
    second = assemble_application(registry=second_registry)
    assert first_registry.all() == second_registry.all() == []
    first_matcher, second_matcher = (
        first.foi.category_matcher(),
        second.foi.category_matcher(),
    )
    assert first_matcher is not second_matcher
    assert first_matcher.resource is first_registry.get("foi_categories_cpu")
    assert first_matcher.resource is not second_matcher.resource
    data = DeploymentCatalog.load().model_dump()
    data["foi"]["category_matcher"] = "question_slice_cpu"
    with pytest.raises(ValueError, match="CategoryMatcherDeployment"):
        DeploymentCatalog.model_validate(data)
    data["foi"]["category_matcher"] = None
    assert (
        assemble_application(
            DeploymentCatalog.model_validate(data)
        ).foi.category_matcher
        is None
    )
    entry = next(
        row
        for row in data["local"]["deployment"]
        if row["loader"] == "foi_category_matcher"
    )
    entry["encoder_ref"] = "missing"
    with pytest.raises(ValueError, match="encoder reference"):
        DeploymentCatalog.model_validate(data)
