"""
Repository-owned MiniLM encoder and frozen FOI/EIR category scoring.

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

from ..deployments.catalog import (
    CategoryMatcherDeployment,
    DeploymentCatalog,
    ModelCheckpoint,
)
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
    Apply the published float32 scoring, abstention and label-order tie rules.
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
    scores = (centroid_scores + vectors @ matrices.descriptions.T) / 2
    winners = scores.argmax(axis=1)
    maximum = centroid_scores.max(axis=1)
    return [
        None
        if maximum[i] < index.no_match.threshold
        else QuestionCategory(category_slug=categories[int(winner)].slug)
        for i, winner in enumerate(winners)
    ]


class CategoryMatcher:
    """
    Cache the encoder, tokenizer and matrices under one process-local lease.
    """

    def __init__(
        self,
        config: CategoryMatcherDeployment,
        *,
        catalog: DeploymentCatalog,
        registry: ResourceRegistry,
    ) -> None:
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
        self.resource = registry.register(
            LocalResource(config.slug, self.load, self.unload)
        )

    def unload(self) -> None:
        """
        Clear encoder and index references; callers hold the resource lease.
        """
        self.loaded = None

    def load(self) -> None:
        """
        Retrieve only explicit data/model files, then load with remote code disabled.
        """
        with self.resource.use():
            if self.loaded is not None:
                return
            try:
                import torch
                from huggingface_hub import snapshot_download
                from transformers import AutoModel, AutoTokenizer

                artifact_dir = Path(
                    snapshot_download(
                        repo_id=self.checkpoint.repo,
                        revision=self.checkpoint.revision,
                        allow_patterns=["index.json", "manifest.json"],
                        token=settings.huggingface_token or None,
                        cache_dir=settings.classifier_cache_dir,
                    )
                )
                index = load_index(
                    artifact_dir / "index.json",
                    artifact_dir / "manifest.json",
                    checksum=self.checksum,
                    encoder=self.encoder,
                )
                snapshot = snapshot_download(
                    repo_id=self.encoder.repo,
                    revision=self.encoder.revision,
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
                    snapshot, local_files_only=True, trust_remote_code=False
                )
                model = AutoModel.from_pretrained(
                    snapshot,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                ).eval()
                if model.config.hidden_size != 384:
                    raise ValueError("Category encoder must produce 384 dimensions")
                matrices = category_matrices(index)
                torch.set_num_threads(settings.cpu_inference_threads)
                self.loaded = LoadedCategoryMatcher(
                    category_index=index,
                    tokenizer=tokenizer,
                    model=model,
                    matrices=matrices,
                )
                self.resource.mark_ready()
            except Exception as exc:
                raise ClassifierUnavailable(
                    "FOI category matcher could not be loaded"
                ) from exc

    def predict(self, texts: list[str]) -> list[QuestionCategory | None]:
        """
        Encode descriptions in batches, with the pinned pooling and truncation.
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
            vectors: list[NDArray[np.float32]] = []
            with torch.inference_mode():
                for start in range(0, len(texts), 16):
                    batch = loaded.tokenizer(
                        texts[start : start + 16],
                        padding=True,
                        truncation=True,
                        max_length=loaded.category_index.embedding_model.max_wordpieces,
                        return_tensors="pt",
                    )
                    hidden = loaded.model(**batch).last_hidden_state.float()
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    vectors.append(
                        functional.normalize(pooled, p=2, dim=1).cpu().numpy()
                    )
            return predict_vectors(
                loaded.category_index, np.concatenate(vectors), matrices=loaded.matrices
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
    Fill question categories after successful inference; retain nulls otherwise.
    """
    for question in topics.questions:
        question.category = None
    eligible = [question for question in topics.questions if question.regime != "SAR"]
    if not eligible:
        return CategoryMetadata(status="not_applicable")
    texts = list(dict.fromkeys(question.topic for question in eligible))

    def run() -> CategoryAssignments:
        matcher = provider()
        assignments = dict(zip(texts, matcher.predict(texts), strict=True))
        return CategoryAssignments(
            assignments=assignments, provenance=matcher.provenance
        )

    try:
        completed = await asyncio.to_thread(run)
    except Exception as exc:
        logger.warning("FOI category matching unavailable: %s", type(exc).__name__)
        return CategoryMetadata(status="unavailable")
    # Apply only a complete result, back on the event loop. Failed or cancelled
    # worker tasks cannot leave partial category assignments on caller-owned objects.
    for question in eligible:
        question.category = completed.assignments[question.topic]
    return CategoryMetadata(status="completed", provenance=completed.provenance)
