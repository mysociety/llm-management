"""Local Presidio preprocessing and immutable evidence for external inference.

The wrapper attests that a policy ran, not that detection is exhaustive. Construct
wrappers only here; never accept attestations supplied by an API client.
"""

import asyncio
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Protocol, TypeVar, cast

from pydantic import TypeAdapter

from .errors import ClassifierUnavailable
from .local_resources import LocalResource, ResourceRegistry, local_resources
from .settings import settings

if TYPE_CHECKING:
    from .foi.response_schemas import ExtractionInput

T = TypeVar("T")
POLICY_VERSION = "foi-pii-v1"
_PLACEHOLDER = re.compile(
    r"<(?:PERSON|EMAIL_ADDRESS|PHONE_NUMBER|CREDIT_CARD|IBAN_CODE|UK_NHS|UK_NINO|POSTAL_ADDRESS)_\d+>"
)
_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "UK_NHS",
    "UK_NINO",
    "POSTAL_ADDRESS",
]
_SEAL = object()


@dataclass(frozen=True, init=False)
class Sanitized(Generic[T]):
    _json: bytes
    _type: Any
    policy_version: str

    def __init__(self, value: T, *, _seal=None):
        if _seal is not _SEAL:
            raise TypeError("Sanitized values must be produced by a sanitizer")
        kind = type(value)
        object.__setattr__(self, "_json", TypeAdapter(kind).dump_json(value))
        object.__setattr__(self, "_type", kind)
        object.__setattr__(self, "policy_version", POLICY_VERSION)

    @property
    def value(self) -> T:
        # Each access gets a fresh view; mutating it cannot alter the attestation.
        return cast(T, TypeAdapter(self._type).validate_json(self._json))


class Sanitizer(Protocol[T]):
    async def sanitize(self, value: T) -> Sanitized[T]: ...


def require_sanitized(value: Sanitized[T]) -> T:
    if not isinstance(value, Sanitized) or value.policy_version != POLICY_VERSION:
        raise TypeError("External inference requires the current sanitization policy")
    return value.value


def _attest(value: T) -> Sanitized[T]:
    return Sanitized(value, _seal=_SEAL)


class SanitizedRequest(NamedTuple):
    """Request text, units, and attested model inputs with shared PII mapping."""

    request_text: str
    units: list[str]
    model_inputs: Sanitized[list[str]]


class PresidioSanitizer:
    def __init__(self, registry: ResourceRegistry = local_resources):
        self._analyzer = None
        self.resource = registry.register(
            LocalResource("presidio", self._load, self._unload)
        )

    def _load(self):
        if self._analyzer is not None:
            return self._analyzer
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import SpacyNlpEngine
            from presidio_analyzer.predefined_recognizers import PhoneRecognizer

            from thinc.api import require_cpu

            require_cpu()
            # Explicit local loading avoids Presidio's automatic model download.
            nlp = spacy.load(settings.presidio_spacy_model)
            engine = SpacyNlpEngine(
                models=[
                    {"lang_code": "en", "model_name": settings.presidio_spacy_model}
                ]
            )
            engine.nlp = {"en": nlp}
            analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])
            analyzer.registry.add_recognizer(PhoneRecognizer(supported_regions=["GB"]))
            # libphonenumber excludes reserved/test ranges. Mask obvious UK
            # phone-shaped markers too, including numbers used in sample FOI text.
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="PHONE_NUMBER",
                    patterns=[
                        Pattern(
                            "UK phone marker",
                            r"(?<!\w)(?:0(?:[ ().-]*\d){10}|\+44(?:[ ().-]*0)?(?:[ ().-]*\d){10})(?!\d)",
                            0.8,
                        )
                    ],
                )
            )
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="UK_NINO",
                    patterns=[
                        Pattern(
                            "UK NINO",
                            r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
                            0.8,
                        )
                    ],
                )
            )
            # Conservative numbered UK street-address pattern; retain cities and dates.
            analyzer.registry.add_recognizer(
                PatternRecognizer(
                    supported_entity="POSTAL_ADDRESS",
                    patterns=[
                        Pattern(
                            "street address",
                            r"(?i)\b\d{1,5}\s+(?:[a-z][a-z'-]*\s+){1,5}(?:street|road|avenue|lane|drive|close|crescent|terrace|way)\b(?:,?\s+[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})?",
                            0.8,
                        )
                    ],
                )
            )
            self._analyzer = analyzer
            return analyzer
        except Exception as exc:
            raise ClassifierUnavailable(
                "Presidio unavailable; install the configured local spaCy model"
            ) from exc

    def _unload(self):
        self._analyzer = None

    def sanitize_strings(self, texts: list[str]) -> list[str]:
        """One request-scoped mapping across fields; no raw mapping is retained."""
        with self.resource.use():
            analyzer = self.resource.warmup()
            mapping: dict[tuple[str, str], str] = {}
            counters: dict[str, int] = {}
            # Reserve existing placeholders to avoid assigning their IDs anew.
            for text in texts:
                for match in _PLACEHOLDER.finditer(text):
                    entity, number = match.group()[1:-1].rsplit("_", 1)
                    counters[entity] = max(counters.get(entity, 0), int(number))
            output = []
            try:
                for text in texts:
                    protected = [
                        (m.start(), m.end()) for m in _PLACEHOLDER.finditer(text)
                    ]
                    results = (
                        analyzer.analyze(
                            text=text,
                            language="en",
                            entities=_ENTITIES,
                            score_threshold=settings.presidio_score_threshold,
                        )
                        if text
                        else []
                    )
                    # Highest confidence / longest span wins conflicting detections.
                    selected = []
                    for item in sorted(
                        results,
                        key=lambda r: (
                            -r.score,
                            -(r.end - r.start),
                            r.start,
                            r.entity_type,
                        ),
                    ):
                        if any(
                            item.start < end and item.end > start
                            for start, end in protected
                        ):
                            continue
                        if any(
                            item.start < r.end and item.end > r.start for r in selected
                        ):
                            continue
                        selected.append(item)
                    parts, cursor = [], 0
                    for item in sorted(selected, key=lambda r: r.start):
                        key = (item.entity_type, text[item.start : item.end].casefold())
                        if key not in mapping:
                            counters[item.entity_type] = (
                                counters.get(item.entity_type, 0) + 1
                            )
                            mapping[key] = (
                                f"<{item.entity_type}_{counters[item.entity_type]}>"
                            )
                        parts.extend([text[cursor : item.start], mapping[key]])
                        cursor = item.end
                    parts.append(text[cursor:])
                    output.append("".join(parts))
                return output
            except Exception as exc:
                raise ClassifierUnavailable(
                    "Presidio sanitization failed; inference was blocked"
                ) from exc

    async def sanitize(self, value: str) -> Sanitized[str]:
        return await asyncio.to_thread(self.sanitize_text, value)

    def sanitize_text(self, text: str) -> Sanitized[str]:
        return _attest(self.sanitize_strings([text])[0])

    def sanitize_request(
        self, request_text: str, units: list[str], contexts: list[str]
    ) -> SanitizedRequest:
        clean = self.sanitize_strings([request_text, *units, *contexts])
        return SanitizedRequest(
            request_text=clean[0],
            units=clean[1 : 1 + len(units)],
            model_inputs=_attest(clean[1 + len(units) :]),
        )

    def sanitize_texts(self, texts: list[str]) -> Sanitized[list[str]]:
        return _attest(self.sanitize_strings(texts))

    def sanitize_payload(self, payload: dict) -> Sanitized[dict]:
        """Sanitize final chat messages, preserving configuration and schema."""
        from copy import deepcopy

        payload = deepcopy(payload)
        messages = payload["messages"]
        clean = self.sanitize_strings([message["content"] for message in messages])
        for message, content in zip(messages, clean, strict=True):
            message["content"] = content
        return _attest(payload)

    def sanitize_extraction(
        self, observed: "ExtractionInput"
    ) -> "Sanitized[ExtractionInput]":
        """Sanitize the explicit text fields of a response model input together."""
        from .foi.response_schemas import ExtractionInput

        data = observed.model_dump()
        fields = []

        def field(owner, key):
            if owner[key] is not None:
                fields.append((owner, key))

        field(data, "request_text")
        for question in data["request"]["questions"]:
            field(question, "text")
        additional = data["request"]["additional_text"]
        for index in range(len(additional)):
            fields.append((additional, index))
        for source in data["sources"]:
            field(source, "text")
            field(source, "filename")
        clean = self.sanitize_strings([owner[key] for owner, key in fields])
        for (owner, key), text in zip(fields, clean, strict=True):
            owner[key] = text
        return _attest(ExtractionInput.model_validate(data))


presidio = PresidioSanitizer()
