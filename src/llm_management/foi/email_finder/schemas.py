from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    computed_field,
    model_validator,
)

from ...web.schemas import PageVisit, SearchResults


class PublicAuthority(BaseModel):
    """
    Identify an authority, with optional details to distinguish similar names.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=300)
    website: HttpUrl | None = None
    previous_email: EmailStr | None = Field(
        default=None,
        description="A former mailbox to prefer an evidenced alternative to, for example after bounces.",
    )
    context: str = Field(default="", max_length=1000)


class Evidence(BaseModel):
    """
    A verbatim passage from a page fetched during this search.
    """

    source_url: HttpUrl
    quote: str = Field(min_length=1)


class FOIFinding(BaseModel):
    """
    A published contact route, preferring FOI/EIR mailboxes over general contact.
    """

    status: Literal["found", "form_only", "not_found"]
    email: EmailStr | None = None
    email_type: Literal["foi_eir", "eir_only", "generic"] | None = Field(
        default=None,
        description=(
            "foi_eir for an FOI or combined FOI/EIR mailbox, eir_only for a route "
            "limited to EIR, generic for general contact; null without an email."
        ),
    )
    request_form_url: HttpUrl | None = None
    evidence: Evidence | None = None
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_submission_route(self) -> Self:
        """
        Require evidence for positive findings and disallow inconsistent routes.
        """
        if self.status == "found":
            if self.email is None or self.email_type is None or self.evidence is None:
                raise ValueError(
                    "A found email requires an address, email type and evidence"
                )
        elif self.email is not None or self.email_type is not None:
            raise ValueError("Only a found result can contain an email or email type")

        if self.status == "form_only":
            if self.request_form_url is None or self.evidence is None:
                raise ValueError("A form-only result requires a form URL and evidence")
        if self.status == "not_found" and (
            self.evidence is not None or self.request_form_url is not None
        ):
            raise ValueError("A not-found result must not claim a submission route")
        return self


class ModelUsage(BaseModel):
    """
    Model usage across all turns, including output-validation retries.
    """

    provider: str
    model: str
    service: str | None = None
    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class SearchUsage(BaseModel):
    """
    External search requests and credits; null credits indicate unknown usage.
    """

    provider: str | None
    requests: int = Field(ge=0)
    credits: int | None = Field(ge=0)


class FOIUsage(BaseModel):
    """
    Measured external-service usage for one completed authority lookup.
    """

    model: ModelUsage
    search: SearchUsage


class FOIResult(BaseModel):
    """
    Keep the original authority and research trail alongside the finding.
    """

    authority: PublicAuthority
    finding: FOIFinding
    checked_at: datetime
    searches: list[SearchResults]
    pages: list[PageVisit]
    usage: FOIUsage

    @computed_field(
        description=(
            "Whether the returned email matches previous_email, ignoring case; "
            "null when either address is absent. This does not verify delivery."
        )
    )
    @property
    def matches_previous_email(self) -> bool | None:
        """
        Compare addresses without treating case-only changes as replacements.
        """
        if self.authority.previous_email is None or self.finding.email is None:
            return None
        return (
            str(self.authority.previous_email).casefold()
            == str(self.finding.email).casefold()
        )
