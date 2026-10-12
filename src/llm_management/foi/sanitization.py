"""
Select related FOI text fields for one request-scoped sanitization mapping.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

from pydantic import BaseModel

from ..sanitization.contracts import Sanitized, TextSanitizer, _attest
from .response_schemas import ExtractionInput


class SanitizedRequest(NamedTuple):
    """
    Request text, units, and attested model inputs with shared PII mapping.
    """

    request_text: str
    units: list[str]
    model_inputs: Sanitized[list[str]]


def sanitize_request(
    sanitizer: TextSanitizer,
    request_text: str,
    units: list[str],
    contexts: list[str],
) -> SanitizedRequest:
    clean = sanitizer.sanitize_strings([request_text, *units, *contexts])
    return SanitizedRequest(
        request_text=clean[0],
        units=clean[1 : 1 + len(units)],
        model_inputs=_attest(clean[1 + len(units) :]),
    )


class TextField(NamedTuple):
    """
    Text and its destination in the copied response-analysis input.
    """

    text: str
    replace: Callable[[str], None]


def sanitize_extraction(
    sanitizer: TextSanitizer, observed: ExtractionInput
) -> Sanitized[ExtractionInput]:
    """
    Sanitize the explicit text fields of a response model input together.
    """
    clean_input = observed.model_copy(deep=True)
    fields: list[TextField] = []

    def field(owner: BaseModel, key: str) -> None:
        text = getattr(owner, key)
        if text is not None:
            fields.append(TextField(text, partial(setattr, owner, key)))

    field(clean_input, "request_text")
    for question in clean_input.request.questions:
        field(question, "text")
    additional = clean_input.request.additional_text
    for index, text in enumerate(additional):
        fields.append(TextField(text, partial(additional.__setitem__, index)))
    for source in clean_input.sources:
        field(source, "text")
        field(source, "filename")
    clean = sanitizer.sanitize_strings([slot.text for slot in fields])
    for slot, text in zip(fields, clean, strict=True):
        slot.replace(text)
    return _attest(ExtractionInput.model_validate(clean_input.model_dump()))
