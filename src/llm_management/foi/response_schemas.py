"""FOI response extraction v4 contract, matching the supplied training example."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import InformationRequestResult, QuestionSliceResult


NonEmptyText = Annotated[str, Field(min_length=1)]
QuestionId = Annotated[str, Field(pattern=r"^q[1-9][0-9]*$")]
SnakeCaseIdentifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")]
Visibility = Literal[
    "visible", "partially_visible", "unobservable", "unknown", "not_applicable"
]
ReferenceRole = Literal[
    "basis_for_withholding",
    "basis_for_release",
    "under_consideration",
    "explicitly_not_applied",
    "unclear",
]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResponseSource(APIModel):
    """Model-visible source; ingestion failure reasons remain outside this API.

    unavailable means content cannot be observed; unknown means visibility itself
    is unresolved. Convert old ingestion diagnostics explicitly, not via coercion.
    """

    id: NonEmptyText
    kind: Literal["email", "attachment"]
    role: Literal["current", "prior", "quoted"]
    sender: Literal["authority", "requester", "other", "unknown"] | None
    filename: NonEmptyText | None
    availability: Literal["visible", "unavailable", "unknown"]
    text: str | None

    @model_validator(mode="after")
    def observed_text_only(self) -> Self:
        if self.availability == "visible" and self.text is None:
            raise ValueError("Visible source requires text")
        if self.availability != "visible" and self.text is not None:
            raise ValueError("Non-visible source requires null text")
        return self


class ExtractedQuestion(APIModel):
    """One question reconstructed upstream, including its qualifications."""

    question_id: QuestionId
    text: NonEmptyText


class ResponseRequestContext(APIModel):
    """Public question result, distinct from training.question_slice diagnostics.

    Status is supplied upstream, never guessed from the response. Uncertain can
    contain tentative questions or none. Additional text retains request context.
    """

    questions: list[ExtractedQuestion]
    extraction_status: Literal["questions_found", "no_questions_found", "uncertain"]
    additional_text: list[str]

    @model_validator(mode="after")
    def consistent_questions(self) -> Self:
        ids = [question.question_id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate question ID")
        if self.extraction_status == "questions_found" and not ids:
            raise ValueError("questions_found requires at least one question")
        if self.extraction_status == "no_questions_found" and ids:
            raise ValueError("no_questions_found requires an empty question list")
        return self


class WholeRequestScope(APIModel):
    kind: Literal["whole_request"]


class QuestionScope(APIModel):
    kind: Literal["question"]
    question_id: QuestionId


class NarrowerScope(APIModel):
    """Response introduces a smaller scope not already addressable in the request."""

    kind: Literal["narrower"]
    question_id: QuestionId
    description: NonEmptyText


class UnresolvedScope(APIModel):
    kind: Literal["unresolved"]


Scope = Annotated[
    WholeRequestScope | QuestionScope | NarrowerScope | UnresolvedScope,
    Field(discriminator="kind"),
]


class ExtractionInput(APIModel):
    request: ResponseRequestContext
    request_text: NonEmptyText | None = Field(
        default=None,
        description="Optional original request for context; IDs come only from request.questions.",
    )
    sources: list[ResponseSource] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("Duplicate response source ID")
        return self


class LegalReference(APIModel):
    """Explicit citation; instrument is a snake_case identifier when known."""

    instrument: SnakeCaseIdentifier | None = Field(
        description="E.g. foia_2000 when identified; null if unresolved."
    )
    provision: NonEmptyText = Field(
        description="Explicit section/regulation, e.g. '43'."
    )
    subsection: NonEmptyText | None = Field(
        description="Explicit subdivision, e.g. '(2)'; null when unspecified."
    )
    role: ReferenceRole


class InformationOutcome(APIModel):
    scope: Scope
    act: Literal[
        "supplied",
        "withheld",
        "not_held",
        "pending",
        "unavailable_unspecified",
        "unclear",
    ]
    answer_content_visibility: Visibility
    content_source_ids: list[NonEmptyText]
    references: list[LegalReference]

    @model_validator(mode="after")
    def visibility_consistency(self) -> Self:
        if len(set(self.content_source_ids)) != len(self.content_source_ids):
            raise ValueError("Duplicate content source ID")
        if self.act == "supplied":
            if self.answer_content_visibility == "not_applicable":
                raise ValueError("Supply visibility cannot be not_applicable")
        elif (
            self.answer_content_visibility != "not_applicable"
            or self.content_source_ids
        ):
            raise ValueError(
                "Non-supply needs not_applicable and empty content sources"
            )
        return self


SimpleEventKind = Literal[
    "processing",
    "delay_notice",
    "clarification_request",
    "fee_notice",
    "rejection_as_invalid",
    "transfer",
    "referral",
    "alternative_access",
    "internal_review_request",
    "internal_review_acknowledgement",
    "review_rights",
    "other",
    "unclear",
]


class SimpleProceduralEvent(APIModel):
    """An observed act, coexisting with information outcomes.

    processing: ongoing handling without a promise of future information/response.
    pending (an outcome): explicitly promised future information/substantive response.
    Do not add processing merely because an outcome is pending. delay_notice:
    explicit delay, which may accompany pending. transfer means authority forwards;
    referral directs the requester elsewhere. review_rights is routine advice, not
    a review decision. internal_review_request can be a current requester message.
    """

    scope: Scope
    kind: SimpleEventKind
    references: list[LegalReference]


class AcknowledgementEvent(APIModel):
    scope: Scope
    kind: Literal["acknowledgement"]
    automatic: bool | None = Field(
        default=None,
        description="True/false only when known; null means origin unclear.",
    )
    references: list[LegalReference]


class ReviewDecisionEvent(APIModel):
    scope: Scope
    kind: Literal["internal_review_decision"]
    review_result: Literal["upheld", "revised", "partly_revised", "unclear"]
    references: list[LegalReference]


ProceduralEvent = Annotated[
    SimpleProceduralEvent | AcknowledgementEvent | ReviewDecisionEvent,
    Field(discriminator="kind"),
]


class ProcessState(APIModel):
    """Communicated state for a scope; do not manufacture finality from silence.

    open requires outstanding action. closed is stated final disposition, not proof
    of adequate answers. conflict concerns the same scope, not different subparts.
    next_actor concerns a required next action, not an optional invitation to reply.
    process_states=null means not assessed; [] means assessed with none identified.
    unclear inside a state means assessed but ambiguous, not a requirement to guess.
    """

    scope: Scope
    completion: Literal["open", "closed", "unclear", "conflict"]
    next_actor: Literal[
        "authority", "requester", "other_authority", "none_stated", "unclear"
    ]


class ProcessReferenceGroup(APIModel):
    """Process-level reference, or reference whose outcome/event owner is unresolved."""

    scope: Scope
    references: list[LegalReference]


class ExtractionOutput(APIModel):
    outcomes: list[InformationOutcome]
    events: list[ProceduralEvent]
    process_states: list[ProcessState] | None = Field(
        default=None,
        description="Optional enrichment: null/omitted = not assessed; [] = assessed and none identified.",
    )
    process_references: list[ProcessReferenceGroup]

    def validate_against(self, observed: ExtractionInput) -> Self:
        questions = {question.question_id for question in observed.request.questions}
        for item in [
            *self.outcomes,
            *self.events,
            *(self.process_states or []),
            *self.process_references,
        ]:
            if isinstance(item.scope, (QuestionScope, NarrowerScope)):
                if item.scope.question_id not in questions:
                    raise ValueError("Unknown question ID")
        sources = {source.id: source for source in observed.sources}
        for outcome in self.outcomes:
            if not set(outcome.content_source_ids) <= sources.keys():
                raise ValueError("Unknown response source ID")
            selected = [sources[source_id] for source_id in outcome.content_source_ids]
            if outcome.answer_content_visibility in ("visible", "partially_visible"):
                if not any(
                    source.availability == "visible" and source.text
                    for source in selected
                ):
                    raise ValueError("Visible supply requires observable destination")
            if outcome.answer_content_visibility == "unobservable":
                if not selected or any(
                    source.availability == "visible" for source in selected
                ):
                    raise ValueError(
                        "Unobservable supply requires unavailable destinations"
                    )
        return self


class ExtractionTrainingPair(APIModel):
    """Only the model-facing pair; validity grants no training approval."""

    input: ExtractionInput
    output: ExtractionOutput

    @model_validator(mode="after")
    def references_resolve(self) -> Self:
        self.output.validate_against(self.input)
        return self


class ResponseAnalysisInput(ExtractionInput):
    """Accept either request-step result or the compact v4 request context."""

    request: InformationRequestResult | QuestionSliceResult | ResponseRequestContext

    @model_validator(mode="after")
    def consistent_request_context(self) -> Self:
        # Apply the v4 invariants even to the existing request-step schemas.
        self.model_input()
        if (
            isinstance(self.request, InformationRequestResult)
            and self.request_text is not None
            and self.request_text != self.request.request_text
        ):
            raise ValueError("request_text differs from the request analysis output")
        return self

    def model_input(self) -> ExtractionInput:
        """Project public diagnostics out of the training-compatible model input."""
        request = ResponseRequestContext.model_validate(
            self.request.model_dump(
                mode="json",
                include={"questions", "extraction_status", "additional_text"},
            )
        )
        request_text = self.request_text
        if request_text is None and isinstance(self.request, InformationRequestResult):
            request_text = self.request.request_text
        return ExtractionInput(
            request=request, request_text=request_text, sources=self.sources
        )
