"""Shared FOI contracts, independent of inference implementations.

QuestionSliceResult predictions, probabilities and orphan indices describe raw
ModernBERT output. Questions/status/residual text describe reconstruction after
any promotion recorded in promoted_continuation_index. These can legitimately
report questions_found alongside original orphan indices.
"""

from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ExtractionBackend = Literal["cpu", "exoscale"]


class UnitLabel(StrEnum):
    """String labels in model-output order."""

    IGNORE = "IGNORE"
    ADDITIONAL = "ADDITIONAL"
    QUESTION_START = "QUESTION_START"
    QUESTION_CONTINUATION = "QUESTION_CONTINUATION"


class StrictModel(BaseModel):
    """Forbid unexpected persisted or generated fields."""

    model_config = ConfigDict(extra="forbid")


class SemanticUnit(StrictModel):
    """One coherent unit classified by ModernBERT."""

    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    kind: Literal["list_item", "sentence"]


class UnitPrediction(StrictModel):
    """One auditable ModernBERT decision."""

    unit: SemanticUnit
    label: UnitLabel
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[UnitLabel, float]


class ExtractedQuestion(StrictModel):
    """One question reconstructed from classified semantic units."""

    question_id: str = Field(pattern=r"^q[1-9][0-9]*$")
    text: str = Field(min_length=1)
    unit_indices: list[int] = Field(min_length=1)


class QuestionSliceResult(StrictModel):
    questions: list[ExtractedQuestion]
    extraction_status: Literal["questions_found", "no_questions_found", "uncertain"]
    orphan_continuation_indices: list[int]
    additional_text: list[str]
    ignored_text: list[str]
    unit_predictions: list[UnitPrediction]
    extraction_backend: ExtractionBackend
    extraction_model: str
    extraction_revision: str | None
    # Predictions, confidence and orphan indices always describe ModernBERT.
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
    def validate_alignment(self):
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
