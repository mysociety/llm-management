"""
Inference fields consumed from the pinned FOI category index and manifest.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

NonblankText = Annotated[str, StringConstraints(pattern=r"\S")]
SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
EmbeddingVector = Annotated[list[FiniteFloat], Field(min_length=384, max_length=384)]


class IndexModel(BaseModel):
    """
    Validate inference fields strictly while ignoring unused research metadata.
    """

    model_config = ConfigDict(extra="ignore", strict=True)


class EmbeddingSpecification(IndexModel):
    """
    Encoder identity and the pooling contract used to build category vectors.
    """

    name: NonblankText
    revision: Revision
    dimensions: Literal[384]
    pooling: Literal["attention-mask mean"]
    normalization: Literal["L2"]
    max_wordpieces: Literal[256]


class CategoryRule(IndexModel):
    """
    Supported ranking rule for the fixed taxonomy.
    """

    score: Literal[
        "0.5 * cosine(topic, centroid_embedding) + 0.5 * cosine(topic, description_embedding)"
    ]
    selection: Literal["highest score among 181 categories"]


class NoMatchRule(IndexModel):
    """
    Reject topics using the global centroid maximum, independently of ranking.
    """

    score: Literal["maximum cosine similarity to any category centroid"]
    comparison: Literal["less_than"]
    threshold: float = Field(ge=0.5, le=0.5, allow_inf_nan=False)
    output_category_slug: None


class IndexCategory(IndexModel):
    """
    Stable category identity and normalized scoring vectors.
    """

    label_id: int = Field(ge=0, le=180)
    slug: NonblankText
    centroid_embedding: EmbeddingVector
    description_embedding: EmbeddingVector

    @field_validator("centroid_embedding", "description_embedding")
    @classmethod
    def validate_normalization(cls, values: list[float]) -> list[float]:
        """
        Check normalization in float32, matching the scoring implementation.
        """
        import numpy as np

        vector = np.asarray(values, dtype=np.float32)
        if not np.isfinite(vector).all() or not np.allclose(
            np.linalg.norm(vector), 1, atol=1e-3
        ):
            raise ValueError("Category vectors must be finite and L2 normalized")
        return values


class CategoryIndex(IndexModel):
    """
    Inference contract with fixed label order and unique category identities.
    """

    format_version: Literal[2]
    model_name: NonblankText
    embedding_model: EmbeddingSpecification
    category_rule: CategoryRule
    no_match: NoMatchRule
    categories: list[IndexCategory] = Field(min_length=181, max_length=181)

    @model_validator(mode="after")
    def validate_taxonomy(self) -> Self:
        """
        Require fixed label order and unique category slugs.
        """
        if [category.label_id for category in self.categories] != list(range(181)):
            raise ValueError("Category IDs must be ordered from 0 to 180")
        if len({category.slug for category in self.categories}) != len(self.categories):
            raise ValueError("Category slugs must be unique")
        return self


class CategoryManifest(IndexModel):
    """
    Index identity and integrity fields checked by the artifact loader.
    """

    format_version: Literal[2]
    model_name: NonblankText
    index_sha256: SHA256
