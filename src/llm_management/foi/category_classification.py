"""
Match FOI/EIR topic descriptions to categories using a pinned MiniLM encoder.

Read the inference path from classify_categories(): it selects eligible topics,
resolves the application's CategoryMatcher through an injected provider, and
calls predict() in a worker thread. The matcher holds a resource lease while it
loads artifacts if needed, encodes topics and applies predict_vectors(). Results
are attached to questions only after the complete batch succeeds.

Only data and safetensors are retrieved from pinned Hub revisions. No upstream
Python is downloaded or executed, and Transformers remote code is disabled.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from ..deployments.catalog import DeploymentCatalog, ModelCheckpoint
from ..deployments.local_config import CategoryMatcherDeployment
from ..inference.errors import ClassifierOutputError, ClassifierUnavailable
from ..inference.resources import LocalResource, ResourceRegistry
from ..settings import settings
from .category_index import CategoryIndex, CategoryManifest
from .category_schemas import (
    CategoryMetadata,
    CategoryProvenance,
    QuestionCategory,
)
from .schemas import TopicOutput

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class CategoryMatrices(NamedTuple):
    """
    Float32 category vectors retained in label order for batch scoring.
    """

    centroids: NDArray[np.float32]
    descriptions: NDArray[np.float32]


class LoadedCategoryMatcher(NamedTuple):
    """
    Encoder and validated index state protected by the matcher's resource lease.
    """

    category_index: CategoryIndex
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    matrices: CategoryMatrices


class CategoryAssignments(NamedTuple):
    """
    Complete topic assignments and provenance returned from the worker thread.
    """

    assignments: dict[str, QuestionCategory | None]
    provenance: CategoryProvenance


logger = logging.getLogger(__name__)


def category_matrices(index: CategoryIndex) -> CategoryMatrices:
    """
    Build float32 matrices once for a validated index.
    """
    import numpy as np

    return CategoryMatrices(
        centroids=np.asarray(
            [row.centroid_embedding for row in index.categories], dtype=np.float32
        ),
        descriptions=np.asarray(
            [row.description_embedding for row in index.categories], dtype=np.float32
        ),
    )


def load_index(
    index_path: Path,
    manifest_path: Path,
    *,
    checksum: str,
    encoder: ModelCheckpoint,
) -> CategoryIndex:
    """
    Verify the artifact against trusted configuration and parse its typed contract.
    """
    raw = index_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("Category index checksum mismatch")
    manifest = CategoryManifest.model_validate_json(manifest_path.read_bytes())
    if manifest.index_sha256 != checksum:
        raise ValueError("Category index checksum mismatch")
    index = CategoryIndex.model_validate_json(raw)
    if index.model_name != manifest.model_name:
        raise ValueError("Category index and manifest identities differ")
    if (
        index.embedding_model.name != encoder.repo
        or index.embedding_model.revision != encoder.revision
    ):
        raise ValueError("Category encoder differs from configured checkpoint")
    return index


def predict_vectors(
    index: CategoryIndex,
    vectors: ArrayLike,
    *,
    matrices: CategoryMatrices | None = None,
) -> list[QuestionCategory | None]:
    """
    Score one normalized embedding per topic against the published category vectors.

    Rank categories by the mean of centroid and description similarity. Decide
    whether to abstain separately, using the best centroid similarity across all
    categories. A topic below that threshold returns None. NumPy's first-maximum
    rule breaks ranking ties in the index's validated label order.
    """
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float32)
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 384
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    ):
        raise ClassifierOutputError("Invalid normalized category input embeddings")
    categories = index.categories
    if matrices is None:
        matrices = category_matrices(index)
    centroid_scores = vectors @ matrices.centroids.T
    description_scores = vectors @ matrices.descriptions.T
    combined_scores = (centroid_scores + description_scores) / 2
    winning_labels = combined_scores.argmax(axis=1)
    best_centroid_similarity = centroid_scores.max(axis=1)
    return [
        None
        if best_centroid_similarity[topic_index] < index.no_match.threshold
        else QuestionCategory(category_slug=categories[int(label)].slug)
        for topic_index, label in enumerate(winning_labels)
    ]


def load_matcher_artifacts(
    *,
    index_checkpoint: ModelCheckpoint,
    encoder_checkpoint: ModelCheckpoint,
    index_sha256: str,
) -> LoadedCategoryMatcher:
    """
    Build complete model state from verified, pinned index and encoder artifacts.

    This is the expensive part of loading: it may download artifacts and imports
    the ML libraries. Verify the index checksum, manifest and encoder identity
    before downloading encoder files. Construct the tokenizer and model from
    local files with remote code disabled and safetensors required for weights.

    Return one complete state object without changing the owner's cache or guard.
    CategoryMatcher.load() holds the lease, publishes this state only on success,
    and translates loading errors into ClassifierUnavailable.
    """
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    index_directory = Path(
        snapshot_download(
            repo_id=index_checkpoint.repo,
            revision=index_checkpoint.revision,
            allow_patterns=["index.json", "manifest.json"],
            token=settings.huggingface_token or None,
            cache_dir=settings.classifier_cache_dir,
        )
    )
    index = load_index(
        index_directory / "index.json",
        index_directory / "manifest.json",
        checksum=index_sha256,
        encoder=encoder_checkpoint,
    )
    encoder_directory = snapshot_download(
        repo_id=encoder_checkpoint.repo,
        revision=encoder_checkpoint.revision,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "added_tokens.json",
        ],
        token=settings.huggingface_token or None,
        cache_dir=settings.classifier_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_directory, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        encoder_directory,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).eval()
    if model.config.hidden_size != 384:
        raise ValueError("Category encoder must produce 384 dimensions")
    matrices = category_matrices(index)
    torch.set_num_threads(settings.cpu_inference_threads)
    return LoadedCategoryMatcher(
        category_index=index,
        tokenizer=tokenizer,
        model=model,
        matrices=matrices,
    )


class CategoryMatcher:
    """
    Own one application's category model state and its lifecycle guard.

    Construction retains pinned configuration and registers a LocalResource;
    loaded remains None until warm-up or predict() calls load(). A successful
    load stores one LoadedCategoryMatcher containing the model, tokenizer, index
    and scoring matrices. unload() drops that state, keeping this controller
    available to load it again on a later request.

    resource is the lifecycle guard. predict() holds its lease for both loading
    and scoring, so idle cleanup cannot clear loaded while inference uses it.
    """

    def __init__(
        self,
        config: CategoryMatcherDeployment,
        *,
        catalog: DeploymentCatalog,
        registry: ResourceRegistry,
    ) -> None:
        """
        Retain index/encoder identities and register callbacks without loading either.

        checkpoint identifies the category index; encoder identifies the embedding
        checkpoint. They are configuration records, not loaded model objects.
        Loaded runtime objects are held together in loaded.
        """
        self.checkpoint = catalog.model[config.model_ref]
        self.encoder = catalog.model[config.encoder_ref]
        self.checksum = config.index_sha256
        self.loaded: LoadedCategoryMatcher | None = None
        self.provenance = CategoryProvenance(
            model=self.checkpoint.repo,
            revision=self.checkpoint.revision,
            index_sha256=self.checksum,
            encoder_model=self.encoder.repo,
            encoder_revision=self.encoder.revision,
        )
        self.resource: LocalResource[None] = registry.register(
            LocalResource(config.slug, self.load, self.unload)
        )

    def unload(self) -> None:
        """
        Drop loaded state after the guard has excluded concurrent inference.

        LocalResource invokes this callback under its lock during idle cleanup
        or an explicit stop. Checkpoint configuration and provenance are retained.
        """
        self.loaded = None

    def load(self) -> None:
        """
        Reuse cached state or load all artifacts while holding the resource lease.

        Warm-up enters here through the guard's load callback; predict() calls it
        under an outer lease that continues through scoring. Nested leases on the
        same thread are supported. Mark readiness only after the complete state
        is available; a failed initial load leaves loaded as None for a later retry.
        """
        with self.resource.use():
            if self.loaded is not None:
                return
            try:
                self.loaded = load_matcher_artifacts(
                    index_checkpoint=self.checkpoint,
                    encoder_checkpoint=self.encoder,
                    index_sha256=self.checksum,
                )
                self.resource.mark_ready()
            except Exception as exc:
                raise ClassifierUnavailable(
                    "FOI category matcher could not be loaded"
                ) from exc

    def predict(self, texts: list[str]) -> list[QuestionCategory | None]:
        """
        Return one category or abstention per topic, preserving input order.

        Hold the lease across loading, tokenization, embedding and scoring. Encode
        batches of 16 with the index's wordpiece limit, take attention-masked mean
        pooling, then normalize each embedding before applying predict_vectors().
        Topic selection and attaching these results to questions belong to
        classify_categories(), rather than this model controller.
        """
        if not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("Category matching requires nonempty topic descriptions")
        with self.resource.use():
            self.load()
            import numpy as np
            import torch
            import torch.nn.functional as functional

            assert self.loaded is not None
            loaded = self.loaded
            embedding_batches: list[NDArray[np.float32]] = []
            with torch.inference_mode():
                for start in range(0, len(texts), 16):
                    batch = loaded.tokenizer(
                        texts[start : start + 16],
                        padding=True,
                        truncation=True,
                        max_length=loaded.category_index.embedding_model.max_wordpieces,
                        return_tensors="pt",
                    )
                    token_embeddings = loaded.model(**batch).last_hidden_state.float()
                    attention_mask = batch["attention_mask"].unsqueeze(-1)
                    mean_embeddings = (token_embeddings * attention_mask).sum(
                        dim=1
                    ) / attention_mask.sum(dim=1).clamp(min=1)
                    embedding_batches.append(
                        functional.normalize(mean_embeddings, p=2, dim=1).cpu().numpy()
                    )
            return predict_vectors(
                loaded.category_index,
                np.concatenate(embedding_batches),
                matrices=loaded.matrices,
            )


def category_matcher(
    config: CategoryMatcherDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> CategoryMatcher:
    """
    Construct a cold, application-owned matcher without network access.
    """
    return CategoryMatcher(config, catalog=catalog, registry=registry)


async def classify_categories(
    topics: TopicOutput, provider: Callable[[], CategoryMatcher]
) -> CategoryMetadata:
    """
    Enrich non-SAR questions with category matches after a complete successful batch.

    Clear existing categories first and skip the provider entirely if no eligible
    questions remain. Predict once for each distinct topic in a worker thread;
    the provider returns this application's retained CategoryMatcher.

    An inference/setup failure returns unavailable metadata and leaves categories
    null. Apply completed assignments on the event loop, so a failing or cancelled
    worker cannot partially modify the caller's questions. A completed prediction
    may still be None when its topic does not meet the match threshold.
    """
    for question in topics.questions:
        question.category = None
    eligible = [question for question in topics.questions if question.regime != "SAR"]
    if not eligible:
        return CategoryMetadata(status="not_applicable")
    texts = list(dict.fromkeys(question.topic for question in eligible))

    def predict_topics() -> CategoryAssignments:
        """
        Resolve the cold matcher and return a complete batch without mutating questions.
        """
        matcher = provider()
        assignments = dict(zip(texts, matcher.predict(texts), strict=True))
        return CategoryAssignments(
            assignments=assignments, provenance=matcher.provenance
        )

    try:
        completed = await asyncio.to_thread(predict_topics)
    except Exception as exc:
        logger.warning("FOI category matching unavailable: %s", type(exc).__name__)
        return CategoryMetadata(status="unavailable")
    # Apply only a complete result, back on the event loop. Failed or cancelled
    # worker tasks cannot leave partial category assignments on caller-owned objects.
    for question in eligible:
        question.category = completed.assignments[question.topic]
    return CategoryMetadata(status="completed", provenance=completed.provenance)
