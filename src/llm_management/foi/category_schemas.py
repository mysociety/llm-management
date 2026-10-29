"""
Category matching results, separate from the generative model's contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CategoryModel(BaseModel):
    """
    Reject unexpected fields in category results and metadata.
    """

    model_config = ConfigDict(extra="forbid")


class QuestionCategory(CategoryModel):
    """
    Taxonomy identifier and assigned category returned on a question.
    """

    category_type: Literal["foi_v1"] = "foi_v1"
    category_slug: str = Field(min_length=1)


class CategoryProvenance(CategoryModel):
    """
    Pinned model identities and index integrity recorded after successful inference.
    """

    model: str
    revision: str
    index_sha256: str
    encoder_model: str
    encoder_revision: str


class CategoryMetadata(CategoryModel):
    """
    Shared operational status and provenance; assignments live on questions.
    """

    status: Literal[
        "completed", "unavailable", "not_configured", "not_applicable", "no_questions"
    ] = "not_configured"
    provenance: CategoryProvenance | None = None
