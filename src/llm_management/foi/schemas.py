"""Shared FOI contracts, independent of inference implementations.

QuestionSliceResult predictions and probabilities describe raw ModernBERT output.
Questions/status/residual text describe reconstruction after any promotion
recorded in promoted_continuation_index. Source membership and orphan indices
are not part of the public response.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExtractionBackend = Literal["cpu", "exoscale"]


class QuestionSliceRequest(BaseModel):
    """
    Raw correspondence accepted by request extraction and classification routes.
    """

    body: str = Field(min_length=1, max_length=100_000)


class UnitLabel(StrEnum):
    """
    String labels in model-output order.
    """

    IGNORE = "IGNORE"
    ADDITIONAL = "ADDITIONAL"
    QUESTION_START = "QUESTION_START"
    QUESTION_CONTINUATION = "QUESTION_CONTINUATION"


class StrictModel(BaseModel):
    """
    Forbid unexpected persisted or generated fields.
    """

    model_config = ConfigDict(extra="forbid")


class SemanticUnit(StrictModel):
    """
    One coherent unit classified by ModernBERT.
    """

    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    kind: Literal["list_item", "sentence"]


class UnitPrediction(StrictModel):
    """
    One auditable ModernBERT decision.
    """

    unit: SemanticUnit
    label: UnitLabel
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[UnitLabel, float]


class ExtractedQuestion(StrictModel):
    """
    One question reconstructed from classified semantic units.
    """

    question_id: str = Field(pattern=r"^q[1-9][0-9]*$")
    text: str = Field(min_length=1)


# Keep the established wire schema name; Python callers use the role-specific alias.
ReconstructedQuestion = ExtractedQuestion


class QuestionSliceResult(StrictModel):
    questions: list[ReconstructedQuestion]
    extraction_status: Literal["questions_found", "no_questions_found", "uncertain"]
    additional_text: list[str]
    ignored_text: list[str]
    unit_predictions: list[UnitPrediction]
    extraction_backend: ExtractionBackend
    extraction_model: str
    extraction_revision: str | None
    # Predictions and confidence always describe ModernBERT.
    # Reconstruction may promote one continuation; raw predictions stay intact.
    promoted_continuation_index: int | None = None


UNIT_LABELS = tuple(UnitLabel)


class TopicQuestion(StrictModel):
    question_id: str = Field(pattern=r"^q[1-9][0-9]*$")
    regime: Literal["FOI", "EIR", "SAR"]
    topic: str = Field(min_length=2, max_length=160)


class TopicOutput(StrictModel):
    questions: list[TopicQuestion] = Field(min_length=1)
    request_topics: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if [q.question_id for q in self.questions] != [
            f"q{i}" for i in range(1, len(self.questions) + 1)
        ]:
            raise ValueError("questions must use ordered IDs q1, q2, ...")
        topics = [" ".join(t.lower().split()) for t in self.request_topics]
        if any(not t for t in topics) or len(set(topics)) != len(topics):
            raise ValueError("request_topics must be nonblank and unique")
        if any(not q.topic.strip() for q in self.questions):
            raise ValueError("question topics must not be blank")
        return self


class InformationRequestResult(QuestionSliceResult):
    request_text: str
    classification: TopicOutput | None
    classification_model: str | None


class ClassifiedQuestion(TopicQuestion):
    """An extracted question with its access regime and short topic."""

    text: str = Field(min_length=1)


class ExtractionMetadata(StrictModel):
    extraction_backend: ExtractionBackend
    extraction_model: str
    extraction_revision: str | None
    classification_model: str | None
    ignored_text: list[str]
    promoted_continuation_index: int | None = None


class ExtractQuestionsResponse(StrictModel):
    """Public combined extraction and classification result."""

    request_text: str
    questions: list[ClassifiedQuestion]
    extraction_status: Literal["questions_found", "no_questions_found", "uncertain"]
    additional_text: list[str]
    request_topics: list[str]
    meta: ExtractionMetadata

    @classmethod
    def from_result(cls, result: InformationRequestResult) -> Self:
        classifications = (
            {q.question_id: q for q in result.classification.questions}
            if result.classification
            else {}
        )
        if set(classifications) != {q.question_id for q in result.questions}:
            raise ValueError("Classification IDs must match extracted questions")
        return cls(
            request_text=result.request_text,
            questions=[
                ClassifiedQuestion(
                    **classifications[q.question_id].model_dump(), text=q.text
                )
                for q in result.questions
            ],
            extraction_status=result.extraction_status,
            additional_text=result.additional_text,
            request_topics=result.classification.request_topics
            if result.classification
            else [],
            meta=ExtractionMetadata.model_validate(
                result.model_dump(include=set(ExtractionMetadata.model_fields))
            ),
        )
