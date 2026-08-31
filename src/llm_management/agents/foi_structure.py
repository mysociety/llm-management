import re
import string
import unicodedata
from typing import Literal

import pysbd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel

InformationRequestType = Literal["FOI", "EIR", "SAR", "OTHER_PERSONAL", "OTHER"]
LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|\([a-z0-9]+\)|[-*\u2022])\s+", re.IGNORECASE)
QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)
_sentence_segmenter = pysbd.Segmenter(language="en", clean=False)


class Unit(BaseModel):
    """An addressable, exact-text unit from the source request."""

    idx: int
    text: str
    kind: Literal["list_item", "line", "sentence"]


def segment(text: str, long_line_chars: int = 200) -> list[Unit]:
    """Split a request line-first, sentence-splitting only long prose lines."""
    units: list[Unit] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        kind: Literal["list_item", "line", "sentence"] = (
            "list_item" if LIST_MARKER.match(line) else "line"
        )
        if len(line) > long_line_chars and kind == "line":
            for sentence in _sentence_segmenter.segment(line):
                if sentence := sentence.strip():
                    units.append(Unit(idx=len(units), text=sentence, kind="sentence"))
        else:
            units.append(Unit(idx=len(units), text=line, kind=kind))
    return units


def normalize_text(text: str) -> str:
    """Normalize Unicode punctuation and collapse whitespace for comparisons."""
    text = unicodedata.normalize("NFKC", text).translate(QUOTES)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.lower().split())


class QuestionSpan(BaseModel):
    units: list[int] = Field(
        ..., min_length=1, description="Ordered unit indices comprising this ask"
    )
    ir_type: InformationRequestType


class Extraction(BaseModel):
    """Model-facing span extraction schema."""

    questions: list[QuestionSpan]
    additional_info_units: list[int] = Field(default_factory=list)


class ExtractionResult(Extraction):
    """Standalone extraction response, including the indexed source units."""

    source_units: list[Unit]


class Question(BaseModel):
    text: str
    ir_type: InformationRequestType


class RequestMetadata(BaseModel):
    short_description: str = Field(
        ..., description="A one-sentence description of the request's main topic"
    )
    keywords: list[str] = Field(
        ..., min_length=5, max_length=5, description="Exactly five concise keywords"
    )


class FOIRequest(BaseModel):
    questions: list[Question]
    additional_info: str | None = None
    short_description: str
    keywords: list[str] = Field(..., min_length=5, max_length=5)


EXTRACTION_SYSTEM_PROMPT = """
Extract information-request asks by selecting indices from the supplied source units.

Rules:
- Return unit indices only. Never copy, rewrite, summarize, or paraphrase an ask.
- Every list_item is requested information: include every list_item in questions.
- Return each numbered or bulleted list_item as its own separate question.
- If any list_item exists, every question must contain exactly one list_item.
- A list-item question may include prose units after it only when they clarify that
  ask. Never combine prose before a list item into that question; preceding request
  context belongs in additional_info_units.
- Keep indices unique, ascending, and in source order.
- Put relevant prose context that is not itself an ask in additional_info_units.
- Never put a list_item, greeting, addressee, sign-off, or thanks in
  additional_info_units.
- Do not select greetings, sign-offs, thanks, or generic correspondence boilerplate.

Classify each ask by the information it directly requests:
- EIR: pollution, emissions, air, water, waste, land, energy, environmental
  monitoring, or environmental impacts. An air-pollution monitoring report is EIR.
- FOI: other recorded information from a UK public authority, including meeting
  minutes and internal publication guidance.
- SAR: the requester's own personal data under subject-access rights.
- OTHER_PERSONAL: another personal-information request outside SAR.
- OTHER: none of these. Do not guess a Scottish regime without evidence.
"""

METADATA_SYSTEM_PROMPT = """
Create metadata for a validated information request. Return a short, factual,
one-sentence description of the overall topic and exactly five concise keywords.
Base the metadata on the supplied request and extracted asks. Do not add facts.
"""


def _format_units(units: list[Unit]) -> str:
    return "\n".join(f"[{unit.idx}] ({unit.kind}) {unit.text}" for unit in units)


def _validate_indices(
    extraction: Extraction, unit_count: int, *, source_units: list[Unit] | None = None
) -> list[str]:
    errors: list[str] = []
    question_units: set[int] = set()
    previous_first = -1
    for number, question in enumerate(extraction.questions, 1):
        if question.units != sorted(set(question.units)):
            errors.append(f"Question {number} indices must be unique and ascending")
        invalid = [idx for idx in question.units if idx < 0 or idx >= unit_count]
        if invalid:
            errors.append(f"Question {number} has invalid indices: {invalid}")
        if overlap := question_units.intersection(question.units):
            errors.append(f"Question {number} reuses indices: {sorted(overlap)}")
        if question.units and question.units[0] < previous_first:
            errors.append("Questions must be in source order")
        if question.units:
            previous_first = question.units[0]
        question_units.update(question.units)
    if extraction.additional_info_units != sorted(
        set(extraction.additional_info_units)
    ):
        errors.append("Additional-information indices must be unique and ascending")
    invalid = [
        idx for idx in extraction.additional_info_units if idx < 0 or idx >= unit_count
    ]
    if invalid:
        errors.append(f"Additional information has invalid indices: {invalid}")
    if overlap := question_units.intersection(extraction.additional_info_units):
        errors.append(
            f"Question and additional-information indices overlap: {sorted(overlap)}"
        )
    if source_units is not None:
        list_item_indices = {
            unit.idx for unit in source_units if unit.kind == "list_item"
        }
        missing_list_items = list_item_indices - question_units
        if missing_list_items:
            errors.append(
                "Every list item must be included in a question; missing indices: "
                f"{sorted(missing_list_items)}"
            )
        for number, question in enumerate(extraction.questions, 1):
            selected_list_items = list_item_indices.intersection(question.units)
            if list_item_indices and not selected_list_items:
                errors.append(f"Question {number} is not anchored to a list item")
            elif len(selected_list_items) > 1:
                errors.append(
                    f"Question {number} combines separate list items: "
                    f"{sorted(selected_list_items)}"
                )
            elif selected_list_items:
                list_item_idx = next(iter(selected_list_items))
                preceding_units = [idx for idx in question.units if idx < list_item_idx]
                if preceding_units:
                    errors.append(
                        f"Question {number} includes prose before its list item: "
                        f"{preceding_units}"
                    )
    return errors


def reconstruct_units(
    units: list[Unit], indices: list[int], *, remove_list_markers: bool = False
) -> str:
    """Reconstruct selected source units without model-generated wording."""
    parts: list[str] = []
    for idx in indices:
        text = units[idx].text
        parts.append(
            LIST_MARKER.sub("", text, count=1) if remove_list_markers else text
        )
    return "\n".join(parts)


def resolve_extraction(
    extraction: Extraction, units: list[Unit]
) -> tuple[list[Question], str | None]:
    if errors := _validate_indices(extraction, len(units), source_units=units):
        raise ValueError("; ".join(errors))
    questions = [
        Question(
            text=reconstruct_units(units, item.units, remove_list_markers=True),
            ir_type=item.ir_type,
        )
        for item in extraction.questions
    ]
    additional_info = (
        reconstruct_units(units, extraction.additional_info_units)
        if extraction.additional_info_units
        else None
    )
    return questions, additional_info


async def extract_request_spans(
    *, model: OpenAIChatModel, request_text: str
) -> ExtractionResult:
    """Run the constrained span-selection agent over an information request."""
    units = segment(request_text)
    agent = Agent(
        model,
        output_type=NativeOutput(Extraction, strict=True),
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        retries=2,
        model_settings={"temperature": 0.1, "max_tokens": 1024},
    )

    @agent.output_validator
    async def validate_extraction(output: Extraction) -> Extraction:
        if errors := _validate_indices(output, len(units), source_units=units):
            raise ModelRetry("Invalid source-unit selection:\n" + "\n".join(errors))
        return output

    result = await agent.run(
        "Select the asks from these source units:\n\n" + _format_units(units)
    )
    return ExtractionResult(**result.output.model_dump(), source_units=units)


async def generate_request_metadata(
    *,
    model: OpenAIChatModel,
    request_text: str,
    questions: list[Question],
    additional_info: str | None = None,
) -> RequestMetadata:
    """Generate descriptive metadata independently of span extraction."""
    agent = Agent(
        model,
        output_type=NativeOutput(RequestMetadata, strict=True),
        system_prompt=METADATA_SYSTEM_PROMPT,
        retries=2,
        model_settings={"temperature": 0.1, "max_tokens": 256},
    )
    question_text = "\n".join(
        f"- [{question.ir_type}] {question.text}" for question in questions
    )
    prompt = f"Original request:\n{request_text}\n\nValidated asks:\n{question_text}"
    if additional_info:
        prompt += f"\n\nAdditional information:\n{additional_info}"
    result = await agent.run(prompt)
    return result.output


async def extract_structure_from_request(
    *, model: OpenAIChatModel, request_text: str
) -> FOIRequest:
    """Run extraction and metadata agents and combine their API response."""
    extraction = await extract_request_spans(model=model, request_text=request_text)
    questions, additional_info = resolve_extraction(extraction, extraction.source_units)
    metadata = await generate_request_metadata(
        model=model,
        request_text=request_text,
        questions=questions,
        additional_info=additional_info,
    )
    return FOIRequest(
        questions=questions,
        additional_info=additional_info,
        short_description=metadata.short_description,
        keywords=metadata.keywords,
    )
