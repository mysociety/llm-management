"""
Typed dataset examples shared by live validation tests and inference-free mocks.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .response_schemas import ExtractionOutput, ResponseAnalysisInput
from .schemas import (
    ExtractionBackend,
    ExtractQuestionsResponse,
    InformationRequestResult,
    QuestionSliceRequest,
    QuestionSliceResult,
    ReconstructedQuestion,
    TopicOutput,
)


class ExtractionExpectation(BaseModel):
    """
    Gold extraction fields, excluding model diagnostics and topic classification.
    """

    model_config = ConfigDict(extra="forbid")

    questions: list[ReconstructedQuestion]
    extraction_status: Literal["questions_found", "no_questions_found", "uncertain"]
    additional_text: list[str]

    @classmethod
    def from_result(
        cls, result: QuestionSliceResult | ExtractQuestionsResponse
    ) -> Self:
        """
        Select the public fields for which the dataset provides gold labels.
        """
        return cls(
            questions=[
                ReconstructedQuestion(question_id=q.question_id, text=q.text)
                for q in result.questions
            ],
            extraction_status=result.extraction_status,
            additional_text=result.additional_text,
        )


class AcceptedExtraction(BaseModel):
    """
    A reviewed equivalent reconstruction, kept separate from dataset gold labels.
    """

    model_config = ConfigDict(extra="forbid")

    output: ExtractionExpectation
    reason: str = Field(min_length=1)
    provenance: str = Field(min_length=1)


class AcceptedClassification(BaseModel):
    """
    Reviewed topic wording with unchanged question IDs, order and access regimes.
    """

    model_config = ConfigDict(extra="forbid")

    output: TopicOutput
    reason: str = Field(min_length=1)
    provenance: str = Field(min_length=1)


class RequestExample(BaseModel):
    """
    Original request and gold reconstruction with copied dataset provenance.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    provenance: str
    input: QuestionSliceRequest
    expected: ExtractionExpectation
    classification: TopicOutput
    accepted_extractions: list[AcceptedExtraction] = Field(default_factory=list)
    accepted_classifications: list[AcceptedClassification] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        """
        Require topic question IDs to match the gold reconstruction in order.
        """
        if [q.question_id for q in self.classification.questions] != [
            q.question_id for q in self.expected.questions
        ]:
            raise ValueError(
                "Example classification does not match extracted questions"
            )
        for alternative in self.accepted_extractions:
            if (
                [q.question_id for q in alternative.output.questions]
                != [q.question_id for q in self.expected.questions]
                or alternative.output.extraction_status
                != self.expected.extraction_status
            ):
                raise ValueError("Accepted extraction changes question IDs or status")
        for alternative in self.accepted_classifications:
            if [q.question_id for q in alternative.output.questions] != [
                q.question_id for q in self.classification.questions
            ] or [q.regime for q in alternative.output.questions] != [
                q.regime for q in self.classification.questions
            ]:
                raise ValueError(
                    "Accepted classification changes question IDs or regimes"
                )
        return self

    def extraction_result(self, backend: ExtractionBackend) -> QuestionSliceResult:
        """
        Wrap gold extraction fields with explicit, inference-free mock diagnostics.
        """
        return QuestionSliceResult(
            questions=[q.model_copy(deep=True) for q in self.expected.questions],
            extraction_status=self.expected.extraction_status,
            additional_text=list(self.expected.additional_text),
            ignored_text=[],
            unit_predictions=[],
            extraction_backend=backend,
            extraction_model="mock:validation-examples",
            extraction_revision=None,
        )

    def information_result(
        self, backend: ExtractionBackend
    ) -> InformationRequestResult:
        """
        Return the complete example, including gold topics and example request text.
        """
        extraction = self.extraction_result(backend)
        return InformationRequestResult(
            questions=extraction.questions,
            extraction_status=extraction.extraction_status,
            additional_text=extraction.additional_text,
            ignored_text=extraction.ignored_text,
            unit_predictions=extraction.unit_predictions,
            extraction_backend=extraction.extraction_backend,
            extraction_model=extraction.extraction_model,
            extraction_revision=extraction.extraction_revision,
            promoted_continuation_index=extraction.promoted_continuation_index,
            request_text=self.input.body,
            classification=self.classification.model_copy(deep=True),
            classification_model="mock:validation-examples",
        )


class ResponseExample(BaseModel):
    """
    Response-analysis pair retaining its independent gold request context.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    provenance: str
    input: ResponseAnalysisInput
    expected: ExtractionOutput

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """
        Require gold question and source references to resolve within this pair.
        """
        self.expected.validate_against(self.input.model_input())
        return self


class ValidationExamples(BaseModel):
    """
    Packaged request and response examples from the supplied validation exports.
    """

    model_config = ConfigDict(extra="forbid")

    requests: list[RequestExample]
    responses: list[ResponseExample]


@cache
def validation_examples() -> ValidationExamples:
    """
    Load bundled payloads without contacting a model, tokenizer or dataset service.

    Treat the cached examples as read-only; copy responses before adapting them.
    Dataset revisions and selection details are documented in the validation guide.
    """
    data = files("llm_management.foi").joinpath("example_data/validation.json")
    return ValidationExamples.model_validate_json(data.read_text(encoding="utf-8"))
