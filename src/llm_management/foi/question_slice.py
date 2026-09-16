"""QuestionSlice preprocessing-v2 and raw-label extraction, shared by all backends."""

from __future__ import annotations

import math
import re
from typing import NamedTuple

import pysbd

from ..errors import ClassifierOutputError
from .schemas import (
    UNIT_LABELS,
    UnitLabel,
    SemanticUnit,
    UnitPrediction,
    ExtractedQuestion,
    QuestionSliceResult,
    ExtractionBackend,
)


LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|\([0-9A-Za-z]+\)|[A-Za-z][.)]|[-*•])\s+")

STRUCTURAL_LINE = re.compile(
    r"^(?:dear\b|hello\b|hi\b|to whom it may concern\b|"
    r"yours (?:faithfully|sincerely)\b|kind regards\b|regards\b|"
    r"thanks\b|thank you\b)",
    re.IGNORECASE,
)

CONNECTOR_END = re.compile(
    r"\b(?:a|an|the|and|or|of|to|for|with|including|on|in|at|by|from)$",
    re.IGNORECASE,
)

SENTENCE_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


class SourceLine(NamedTuple):
    """One non-blank physical line."""

    text: str
    is_list_item: bool


def should_join_lines(left: str, right: str) -> bool:
    """Apply the audited preprocessing-v2 hard-wrap rules."""

    left_text = left.strip()
    right_text = right.strip()
    if not left_text or not right_text:
        return False
    if STRUCTURAL_LINE.match(left_text) or STRUCTURAL_LINE.match(right_text):
        return False
    if LIST_MARKER.match(right_text) or re.search(r"[.!?;:]$", left_text):
        return False
    if right_text[:1].islower() or CONNECTOR_END.search(left_text):
        return True
    return not left_text.endswith(",")


def physical_lines(text: str) -> list[SourceLine | None]:
    """Return stripped lines while preserving explicit paragraph breaks."""

    lines: list[SourceLine | None] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        lines.append(
            SourceLine(stripped, bool(LIST_MARKER.match(raw_line)))
            if stripped
            else None
        )
    return lines


def segment_request(text: str) -> list[SemanticUnit]:
    """Create coherent semantic units using preprocessing-v2."""

    chunks: list[tuple[str, bool]] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            chunks.append((" ".join(pending), False))
            pending.clear()

    for line in physical_lines(text):
        if line is None:
            flush()
            continue
        if line.is_list_item:
            flush()
            chunks.append((line.text, True))
            continue
        if pending and not should_join_lines(pending[-1], line.text):
            flush()
        pending.append(line.text)
    flush()

    units: list[SemanticUnit] = []
    for chunk, is_list_item in chunks:
        if is_list_item:
            units.append(SemanticUnit(index=len(units), text=chunk, kind="list_item"))
            continue
        for sentence in SENTENCE_SEGMENTER.segment(chunk):
            sentence = sentence.strip()
            if sentence:
                units.append(
                    SemanticUnit(index=len(units), text=sentence, kind="sentence")
                )
    if not units:
        raise ValueError("request text contains no semantic units")
    return units


def contextual_inputs(units: list[SemanticUnit]) -> list[str]:
    """Encode neighboring units exactly as used during ModernBERT training."""

    return [
        "\n".join(
            (
                "[PREVIOUS] " + (units[index - 1].text if index else "[NONE]"),
                "[CURRENT] " + unit.text,
                "[NEXT] "
                + (units[index + 1].text if index + 1 < len(units) else "[NONE]"),
            )
        )
        for index, unit in enumerate(units)
    ]


class QuestionExtraction(NamedTuple):
    """
    Reconstructed asks and residual text, including unresolved question evidence.
    """

    questions: list[ExtractedQuestion]
    additional: list[str]
    ignored: list[str]
    orphan_continuation_indices: list[int]


def reconstruct_questions(
    predictions: list[UnitPrediction],
) -> QuestionExtraction:
    """Build ordered questions and retain all non-question text."""

    groups: list[list[SemanticUnit]] = []
    additional: list[str] = []
    ignored: list[str] = []
    orphan_continuation_indices: list[int] = []
    for prediction in predictions:
        if prediction.label == UnitLabel.QUESTION_START:
            groups.append([prediction.unit])
        elif prediction.label == UnitLabel.QUESTION_CONTINUATION and groups:
            groups[-1].append(prediction.unit)
        elif prediction.label == UnitLabel.IGNORE:
            ignored.append(prediction.unit.text)
        else:
            additional.append(prediction.unit.text)
            if prediction.label == UnitLabel.QUESTION_CONTINUATION:
                orphan_continuation_indices.append(prediction.unit.index)
    questions = [
        ExtractedQuestion(
            question_id=f"q{index}",
            text=" ".join(unit.text for unit in group),
            unit_indices=[unit.index for unit in group],
        )
        for index, group in enumerate(groups, 1)
    ]
    return QuestionExtraction(
        questions, additional, ignored, orphan_continuation_indices
    )


def predictions_from_probabilities(
    units: list[SemanticUnit], rows: list[list[float]]
) -> list[UnitPrediction]:
    """Validate normalized model rows and preserve the raw argmax decisions."""
    if len(rows) != len(units):
        raise ClassifierOutputError(
            "Classifier output count does not match source units"
        )
    predictions = []
    for unit, row in zip(units, rows, strict=True):
        if (
            len(row) != len(UNIT_LABELS)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row)
            or not math.isclose(sum(row), 1, abs_tol=0.001)
        ):
            raise ClassifierOutputError(
                "Classifier must return four normalized label probabilities"
            )
        label_id = max(range(len(row)), key=row.__getitem__)
        predictions.append(
            UnitPrediction(
                unit=unit,
                label=UNIT_LABELS[label_id],
                confidence=row[label_id],
                probabilities=dict(zip(UNIT_LABELS, row, strict=True)),
            )
        )
    return predictions


def promote_initial_continuation_if_needed(
    predictions: list[UnitPrediction],
) -> tuple[list[UnitPrediction], int | None]:
    """Change reconstruction inputs only when no question starts exist anywhere."""
    if any(p.label == UnitLabel.QUESTION_START for p in predictions):
        return predictions, None
    for prediction in predictions:
        if prediction.label == UnitLabel.QUESTION_CONTINUATION:
            promoted_index = prediction.unit.index
            return [
                p.model_copy(update={"label": UnitLabel.QUESTION_START})
                if p.unit.index == promoted_index
                else p
                for p in predictions
            ], promoted_index
    return predictions, None


def build_extraction_result(
    units: list[SemanticUnit],
    rows: list[list[float]],
    *,
    backend: ExtractionBackend,
    model: str,
    revision: str | None,
) -> QuestionSliceResult:
    """Combine raw diagnostics with reconstruction and the explicit promotion policy."""
    predictions = predictions_from_probabilities(units, rows)
    original = reconstruct_questions(predictions)
    reconstruction_predictions, promoted_index = promote_initial_continuation_if_needed(
        predictions
    )
    extraction = (
        reconstruct_questions(reconstruction_predictions)
        if promoted_index is not None
        else original
    )
    if extraction.orphan_continuation_indices:
        status = "uncertain"
    elif extraction.questions:
        status = "questions_found"
    else:
        status = "no_questions_found"
    return QuestionSliceResult(
        questions=extraction.questions,
        extraction_status=status,
        orphan_continuation_indices=original.orphan_continuation_indices,
        promoted_continuation_index=promoted_index,
        additional_text=extraction.additional,
        ignored_text=extraction.ignored,
        unit_predictions=predictions,
        extraction_backend=backend,
        extraction_model=model,
        extraction_revision=revision,
    )
