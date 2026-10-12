"""Immutable evidence that the local sanitization policy ran.

Attestation records policy execution, not exhaustive detection. Never accept an
attestation supplied by an API client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from pydantic import TypeAdapter

T = TypeVar("T")
POLICY_VERSION = "foi-pii-v1"
_SEAL = object()


@dataclass(frozen=True, init=False)
class Sanitized(Generic[T]):
    """
    Immutable snapshot recording that the local sanitization policy ran.

    Construction requires the internal seal. Serialize the sanitized value at creation
    and return a fresh decoded view on each access, preventing mutations from changing
    the attested snapshot. The policy marker is not proof that every identifier was
    detected and is never accepted as an attestation supplied by an API client.
    """

    _json: bytes
    _type: Any
    policy_version: str

    def __init__(self, value: T, *, _seal: object | None = None) -> None:
        """
        Capture a sanitized snapshot; reject construction without the internal seal.
        """
        if _seal is not _SEAL:
            raise TypeError("Sanitized values must be produced by a sanitizer")
        kind = type(value)
        object.__setattr__(self, "_json", TypeAdapter(kind).dump_json(value))
        object.__setattr__(self, "_type", kind)
        object.__setattr__(self, "policy_version", POLICY_VERSION)

    @property
    def value(self) -> T:
        # Each access gets a fresh view; mutating it cannot alter the attestation.
        """
        Decode a fresh view so callers cannot mutate the stored attestation.
        """
        return cast(T, TypeAdapter(self._type).validate_json(self._json))


class Sanitizer(Protocol[T]):
    """
    Describe an async sanitizer that returns immutable policy evidence.
    """

    async def sanitize(self, value: T) -> Sanitized[T]:
        """
        Apply the local policy and return its attested result.
        """
        ...


class TextSanitizer(Protocol):
    """
    Apply one shared placeholder mapping to related text fields.
    """

    def sanitize_strings(self, texts: list[str]) -> list[str]:
        """
        Apply one placeholder mapping across the supplied ordered text fields.
        """
        ...


class ChatSanitizer(TextSanitizer, Protocol):
    """
    Sanitize message text while preserving a chat transport's JSON envelope.
    """

    def sanitize_payload(self, payload: dict[str, Any]) -> Sanitized[dict[str, Any]]:
        """
        Attest sanitized message text while retaining the transport configuration.
        """
        ...


def require_sanitized(value: Sanitized[T]) -> T:
    """
    Require a locally produced snapshot bearing the current policy version.

    Return a fresh decoded value. Reject raw input and obsolete policy evidence;
    this check cannot establish that detection was exhaustive.
    """
    if not isinstance(value, Sanitized) or value.policy_version != POLICY_VERSION:
        raise TypeError("External inference requires the current sanitization policy")
    return value.value


def _attest(value: T) -> Sanitized[T]:
    """
    Seal output after a trusted local sanitizer has applied the policy.

    This helper performs no detection. Call it only after sanitizing the selected
    fields; exposing it as a caller-facing constructor would bypass that contract.
    """
    return Sanitized(value, _seal=_SEAL)
