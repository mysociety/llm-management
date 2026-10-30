"""
Local Presidio analyzer lifecycle and deterministic placeholder replacement.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, NamedTuple

from ..deployments.catalog import DeploymentCatalog
from ..deployments.local_config import PresidioDeployment
from ..inference.errors import ClassifierUnavailable
from ..inference.resources import LocalResource, ResourceRegistry
from .contracts import Sanitized, _attest

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult


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


class ProtectedSpan(NamedTuple):
    """
    Character offsets of an existing placeholder that must be preserved.
    """

    start: int
    end: int


def select_spans(
    results: list[RecognizerResult], protected: list[ProtectedSpan]
) -> list[RecognizerResult]:
    """
    Choose non-overlapping detections without replacing existing placeholders.

    Prefer higher confidence, then longer spans; start position and entity type break
    remaining ties deterministically. Return selected detections in priority order;
    replacement callers sort them by position before rebuilding the text.
    """
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
        if any(item.start < end and item.end > start for start, end in protected):
            continue
        if any(item.start < r.end and item.end > r.start for r in selected):
            continue
        selected.append(item)
    return selected


def add_recognizers(analyzer: AnalyzerEngine) -> None:
    """
    Apply the configured UK-specific detection policy to a local analyzer.
    """
    from presidio_analyzer import Pattern, PatternRecognizer
    from presidio_analyzer.predefined_recognizers import PhoneRecognizer

    from .recognizers import OfflineEmailRecognizer

    analyzer.registry.remove_recognizer("EmailRecognizer")
    analyzer.registry.add_recognizer(OfflineEmailRecognizer())
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


class PresidioSanitizer:
    """
    Own a lazy local analyzer and apply one deterministic mapping per call.

    Keep analyzer reuse behind a resource lease so loading, sanitization and idle
    unloading cannot race. Related text fields must be submitted together to share
    placeholder identity. No original-to-placeholder mapping survives the call.
    Detection follows the configured policy and may miss identifiers.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        config: PresidioDeployment,
    ) -> None:
        """
        Bind explicit configuration and registry ownership without loading the NLP model.
        """
        self.config = config
        self._registry = registry
        self._resource = None
        self._analyzer = None

    @property
    def resource(self) -> LocalResource[Any]:
        """
        Register the analyzer's cold lifecycle guard once in its owning registry.
        """
        if self._resource is None:
            self._resource = self._registry.register_once(
                self.config.slug,
                lambda: LocalResource(self.config.slug, self._load, self._unload),
            )
        return self._resource

    def _load(self) -> AnalyzerEngine:
        """
        Reuse or create the CPU analyzer from the installed local spaCy model.

        The resource guard calls this loader while leased. Load the configured model
        explicitly to avoid Presidio downloading it during requests, then add the UK
        recognizers. Convert initialization failures to ClassifierUnavailable.
        """
        if self._analyzer is not None:
            return self._analyzer
        try:
            import spacy
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import SpacyNlpEngine
            from thinc.api import require_cpu

            require_cpu()
            # Explicit local loading avoids Presidio's automatic model download.
            nlp = spacy.load(self.config.spacy_model)
            engine = SpacyNlpEngine(
                models=[{"lang_code": "en", "model_name": self.config.spacy_model}]
            )
            engine.nlp = {"en": nlp}
            analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])
            add_recognizers(analyzer)
            self._analyzer = analyzer
            return analyzer
        except Exception as exc:
            raise ClassifierUnavailable(
                "Presidio unavailable; install the configured local spaCy model"
            ) from exc

    def _unload(self) -> None:
        """
        Drop the analyzer reference when the guard releases an inactive resource.
        """
        self._analyzer = None

    def sanitize_strings(self, texts: list[str]) -> list[str]:
        """
        Replace detected entities using one mapping across ordered text fields.

        Reuse placeholders for case-insensitive matches of the same entity type within
        this call. Preserve existing markers and reserve their numbers before assigning
        new ones. Resolve overlapping detections deterministically and retain field order.

        Hold the analyzer lease through the complete operation. Keep mappings local to
        the call and raise ClassifierUnavailable on detection or replacement failure.
        """
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
                        ProtectedSpan(m.start(), m.end())
                        for m in _PLACEHOLDER.finditer(text)
                    ]
                    results = (
                        analyzer.analyze(
                            text=text,
                            language="en",
                            entities=_ENTITIES,
                            score_threshold=self.config.score_threshold,
                        )
                        if text
                        else []
                    )
                    # Highest confidence / longest span wins conflicting detections.
                    selected = select_spans(results, protected)
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
        """
        Run synchronous text sanitization in a worker thread and return attestation.
        """
        return await asyncio.to_thread(self.sanitize_text, value)

    def sanitize_text(self, text: str) -> Sanitized[str]:
        """
        Apply the local policy to one string and seal the resulting snapshot.
        """
        return _attest(self.sanitize_strings([text])[0])

    def sanitize_texts(self, texts: list[str]) -> Sanitized[list[str]]:
        """
        Sanitize related strings with shared placeholder identity and seal the list.
        """
        return _attest(self.sanitize_strings(texts))

    def sanitize_payload(self, payload: dict[str, Any]) -> Sanitized[dict[str, Any]]:
        """
        Copy a chat envelope and sanitize every message's string content together.

        Preserve configuration, roles and schema while leaving the caller's input
        untouched. This adapter does not scan arbitrary metadata or nested multimodal
        content; callers must select such fields through an appropriate adapter.
        """
        from copy import deepcopy

        payload = deepcopy(payload)
        messages = payload["messages"]
        clean = self.sanitize_strings([message["content"] for message in messages])
        for message, content in zip(messages, clean, strict=True):
            message["content"] = content
        return _attest(payload)


def configured_presidio(
    config: PresidioDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> PresidioSanitizer:
    """
    Construct a cold sanitizer owned by the supplied registry.

    Accept the catalog for the common resource factory interface; sanitizer
    configuration is already complete and needs no checkpoint lookup.
    """
    return PresidioSanitizer(config=config, registry=registry)
