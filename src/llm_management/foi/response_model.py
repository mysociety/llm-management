"""
Pinned FOI status prompts, tokenizer ownership and remote JSON generation.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx2
from openai import APIError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict

from ..deployments.catalog import DeploymentCatalog, ResponseTokenizerDeployment
from ..inference.errors import ClassifierUnavailable
from ..inference.resources import LocalResource, ResourceRegistry
from ..sanitization import Sanitized, require_sanitized
from ..sanitization.contracts import _attest
from ..settings import settings
from .response_schemas import (
    ExtractionInput,
    ExtractionOutput,
    InformationOutcome,
    ProceduralEvent,
    ProcessReferenceGroup,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT_SHA256 = (
    "9ddffc18c8fe3d5d88fb3d3b5a07387d7bdfe363cb3e7fd01f0a40263e86a7f5"
)


OUTPUT_SCHEMA_SHA256 = (
    "0b73b20340cface4bc00fb0ac697a72e7cd30c75efcb9da8867cfb9eab9806dc"
)


class StatusOutputError(RuntimeError):
    """
    Status generation failed, was incomplete or violated the trained output contract.
    """


class StatusOutput(BaseModel):
    """
    Trained output excludes completion and next-actor enrichment entirely.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    outcomes: list[InformationOutcome]
    events: list[ProceduralEvent]
    process_references: list[ProcessReferenceGroup]

    def result(self) -> ExtractionOutput:
        """
        Project trained fields explicitly; leave process_states unassessed.
        """
        return ExtractionOutput(
            outcomes=self.outcomes,
            events=self.events,
            process_references=self.process_references,
        )


class StatusMessages(NamedTuple):
    """
    Exact system instruction and canonical sanitized input for the trained template.
    """

    system: str
    user: str

    def chat_messages(self) -> list[ChatCompletionMessageParam]:
        """
        Serialize the two messages at the OpenAI transport boundary.
        """
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def output_schema() -> dict[str, Any]:
    """
    Read the published schema, preserving its wire form rather than regenerating it.
    """
    data = (
        files("llm_management.foi")
        .joinpath("status_contract/OUTPUT-SCHEMA.json")
        .read_bytes()
    )
    if hashlib.sha256(data).hexdigest() != OUTPUT_SCHEMA_SHA256:
        raise ClassifierUnavailable(
            "FOI status schema does not match its pinned contract"
        )
    return json.loads(data)


def prepare_request(
    observed: Sanitized[ExtractionInput],
    *,
    tokenizer: LocalResource[PreTrainedTokenizerBase],
    catalog: DeploymentCatalog,
) -> Sanitized[StatusMessages]:
    """
    Count the exact non-thinking template and reserve the full generation allowance.

    Canonicalize only attested model input, including default nulls. Reject oversized
    requests without truncation before remote startup. Run in a worker thread while
    holding the tokenizer's lease; the trusted system instruction stays unchanged.
    """
    clean = require_sanitized(observed)
    output_schema()
    system = (
        files("llm_management.foi")
        .joinpath("status_contract/SYSTEM-PROMPT.txt")
        .read_text()
    )
    if hashlib.sha256(system.encode()).hexdigest() != SYSTEM_PROMPT_SHA256:
        raise ClassifierUnavailable(
            "FOI status instruction does not match its pinned contract"
        )
    messages = StatusMessages(
        system=system,
        user=json.dumps(
            clean.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        ),
    )
    with tokenizer.use():
        token_ids = tokenizer.warmup().apply_chat_template(
            [
                {"role": "system", "content": messages.system},
                {"role": "user", "content": messages.user},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            truncation=False,
        )
    budget = catalog.require_foi().response
    limit = budget.context_tokens - budget.output_tokens
    if len(token_ids) > limit:
        raise ValueError(
            f"FOI status prompt has {len(token_ids)} tokens; limit is {limit}. Input was not truncated."
        )
    return _attest(messages)


def tokenizer_resource(
    config: ResponseTokenizerDeployment,
    *,
    catalog: DeploymentCatalog,
    registry: ResourceRegistry,
) -> LocalResource[PreTrainedTokenizerBase]:
    """
    Register a cold tokenizer for the pinned status checkpoint and its chat template.

    Download tokenizer files only on warm-up, with explicit authentication, then load
    locally. Idle release clears the cache; no model weights are loaded in this process.
    """
    cached: PreTrainedTokenizerBase | None = None

    def load() -> PreTrainedTokenizerBase:
        """
        Reuse the owner's tokenizer or fetch its pinned metadata and vocabulary.
        """
        nonlocal cached
        if cached is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            checkpoint = catalog.model[config.model_ref]
            snapshot = snapshot_download(
                checkpoint.repo,
                revision=checkpoint.revision,
                token=settings.huggingface_token or None,
                cache_dir=settings.classifier_cache_dir,
                allow_patterns=[
                    "config.json",
                    "tokenizer*",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "chat_template.jinja",
                ],
            )
            cached = AutoTokenizer.from_pretrained(
                snapshot, local_files_only=True, fix_mistral_regex=False
            )
        assert cached is not None
        return cached

    def unload() -> None:
        """
        Drop the cached tokenizer after its guard becomes inactive.
        """
        nonlocal cached
        cached = None

    return registry.register(LocalResource(config.slug, load, unload))


async def classify_status(
    messages: Sanitized[StatusMessages],
    *,
    model: str,
    deployment_url: str,
    api_key: str,
    output_tokens: int,
) -> ExtractionOutput:
    """
    Generate constrained JSON with deterministic, non-thinking status settings.

    The SDK owns and closes the httpx2 client, including failed requests. Disable
    retries and reject missing choices, refusals, truncation and malformed output;
    do not repair JSON or infer process_states that this model was not trained for.
    """
    clean = require_sanitized(messages)
    try:
        async with AsyncOpenAI(
            base_url=deployment_url.rstrip("/") + "/",
            api_key=api_key,
            http_client=httpx2.AsyncClient(timeout=120),
            max_retries=0,
        ) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=clean.chat_messages(),
                temperature=0,
                max_tokens=output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ExtractionOutput",
                        "strict": True,
                        "schema": output_schema(),
                    },
                },
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        if len(response.choices) != 1:
            raise StatusOutputError("Status model returned an unexpected choice count")
        choice = response.choices[0]
        if (
            choice.finish_reason != "stop"
            or choice.message.refusal
            or not choice.message.content
        ):
            raise StatusOutputError("Status model did not complete normally")
        return StatusOutput.model_validate_json(choice.message.content).result()
    except (APIError, httpx2.HTTPError, ValueError) as exc:
        raise StatusOutputError(
            "Status model request failed or returned invalid output"
        ) from exc
