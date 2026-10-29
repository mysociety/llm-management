"""
Training-compatible question topic/regime classification for extracted questions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx2
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from ..deployments import TopicTokenizerDeployment, get_catalog
from ..deployments.catalog import DeploymentCatalog
from ..inference.resources import LocalResource, ResourceRegistry
from ..sanitization import Sanitized, presidio, require_sanitized
from ..sanitization.contracts import ChatSanitizer
from ..settings import settings
from .schemas import GeneratedTopicOutput, ReconstructedQuestion, TopicOutput

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


SYSTEM_PROMPT = (
    "For each extracted UK information-request question, identify its access regime "
    "and write a concise topic noun phrase. Return only JSON matching the supplied "
    "schema. Topics describe the subject, not the requesting action, and must omit "
    "names, identifiers, dates, and redaction markers."
)


class TopicOutputError(Exception):
    pass


class QuestionTopicChatModel(OpenAIChatModel):
    """
    Enforce completion policy before the adapter normalizes finish reasons.
    """

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        """
        Reject truncated or abnormal completions before the SDK normalizes stop reasons.
        """
        if isinstance(response, ChatCompletion) and response.choices:
            reason = response.choices[0].finish_reason
            if reason == "length":
                raise TopicOutputError("Topic model exhausted its completion budget")
            if reason != "stop":
                raise TopicOutputError("Topic model did not complete normally")
        return super()._process_response(response)


class TopicDependencies(NamedTuple):
    """
    Prepared sanitized system prompt and ordered IDs expected in topic output.
    """

    system_prompt: str
    question_ids: list[str]


topic_agent = Agent(
    deps_type=TopicDependencies,
    output_type=NativeOutput(GeneratedTopicOutput, strict=True, template=False),
    retries=0,
    model_settings={"timeout": 120},
)


@topic_agent.system_prompt
def topic_system_prompt(ctx: RunContext[TopicDependencies]) -> str:
    return ctx.deps.system_prompt


@topic_agent.output_validator
def validate_question_ids(
    ctx: RunContext[TopicDependencies], output: GeneratedTopicOutput
) -> GeneratedTopicOutput:
    """
    Require output question IDs to match the prepared extraction exactly and in order.
    """
    if [q.question_id for q in output.questions] != ctx.deps.question_ids:
        raise TopicOutputError(
            "Topic model question IDs do not match extracted questions"
        )
    return output


def topic_model(model: str, client: AsyncOpenAI) -> QuestionTopicChatModel:
    """
    Configure the fine-tuned deployment's native JSON schema support.
    """
    return QuestionTopicChatModel(
        model,
        provider=OpenAIProvider(openai_client=client),
        profile=OpenAIModelProfile(
            supports_json_schema_output=True,
            openai_system_prompt_role="system",
        ),
    )


def completion_budget(
    question_count: int, *, catalog: DeploymentCatalog | None = None
) -> int:
    """
    Calculate per-question output allowance within the configured deployment limit.

    Reject zero questions and requests exceeding the limit rather than truncating
    model output or silently reducing the per-question allowance.
    """
    topic = (catalog if catalog is not None else get_catalog()).require_foi().topic
    budget = max(
        topic.min_output_tokens,
        topic.output_overhead + topic.output_tokens_per_question * question_count,
    )
    if question_count < 1 or budget > topic.output_limit:
        raise ValueError(
            f"Topic model supports 1–{topic.max_questions} questions within its {topic.output_limit:,}-token output budget"
        )
    return budget


def question_extractor_tokenizer(
    config: TopicTokenizerDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    cache: dict[str, PreTrainedTokenizerBase] | None = None,
) -> PreTrainedTokenizerBase:
    """
    Load the pinned topic tokenizer through authenticated local-cache access.

    Without a supplied cache, resolve the default application's tokenizer guard and
    warm it. A supplied cache belongs to the explicit resource owner and retains the
    tokenizer for later calls. Preserve the checkpoint's trained chat template.
    """
    catalog = catalog if catalog is not None else get_catalog()
    if cache is None:
        from ..bootstrap import default_application

        return default_application().foi.tokenizer().warmup()
    config = config or catalog.local.get(
        catalog.require_foi().tokenizer, TopicTokenizerDeployment
    )
    if config.slug in cache:
        return cache[config.slug]
    checkpoint = catalog.model[config.model_ref]
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Load locally after an explicitly authenticated download. Transformers may
    # perform additional Hub metadata requests without forwarding token=.
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
            "additional_chat_templates/*.jinja",
        ],
    )
    # Preserve the trained Granite tokenizer; do not apply Mistral regex changes.
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, fix_mistral_regex=False
    )
    cache[config.slug] = tokenizer
    return tokenizer


def tokenizer_resource(
    config: TopicTokenizerDeployment | None = None,
    *,
    catalog: DeploymentCatalog | None = None,
    registry: ResourceRegistry | None = None,
) -> LocalResource[PreTrainedTokenizerBase]:
    """
    Resolve the default application's retained owner or construct an explicit one.

    An explicit registry requires both configuration and catalog. Without a registry,
    use the default assembly's provider and reject explicit resource configuration.
    Construction registers a cold guard; loading remains a warm-up or inference operation.
    """
    if registry is not None:
        if config is None or catalog is None:
            raise ValueError(
                "Explicit resource construction requires config and catalog"
            )
        cache = {}
        return registry.register(
            LocalResource(
                config.slug,
                lambda: question_extractor_tokenizer(
                    config, catalog=catalog, cache=cache
                ),
                cache.clear,
            )
        )
    from ..bootstrap import default_application

    if config is not None:
        raise ValueError(
            "Explicit resource configuration requires catalog and registry"
        )
    return default_application().foi.tokenizer()


def prepare_topic_request(
    request_text: str,
    questions: list[ReconstructedQuestion],
    *,
    catalog: DeploymentCatalog | None = None,
    sanitizer: ChatSanitizer | None = None,
    tokenizer_guard: LocalResource[PreTrainedTokenizerBase] | None = None,
) -> Sanitized[dict[str, Any]]:
    """
    Prepare an attested chat envelope and reject prompts outside the token budget.

    Sanitize system and user message text together, then count the complete trained
    chat template without truncation under the tokenizer's resource lease. Configure
    strict topic output schema and the per-question completion allowance. This runs
    synchronously in a worker thread before a remote topic deployment is started.
    """
    catalog = catalog if catalog is not None else get_catalog()
    sanitizer = sanitizer if sanitizer is not None else presidio
    budget = completion_budget(len(questions), catalog=catalog)
    schema = GeneratedTopicOutput.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\nSchema: " + json.dumps(schema),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request_text": request_text,
                    "questions": [
                        {"question_id": q.question_id, "text": q.text}
                        for q in questions
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    payload = sanitizer.sanitize_payload(
        {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": budget,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "TopicOutput",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
    )
    resource = tokenizer_guard if tokenizer_guard is not None else tokenizer_resource()
    with resource.use():
        tokenizer = resource.warmup()
        token_ids = tokenizer.apply_chat_template(
            payload.value["messages"],
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
        )
    config = catalog.local.get(
        catalog.require_foi().tokenizer, TopicTokenizerDeployment
    )
    if len(token_ids) > config.max_tokens:
        raise ValueError(
            f"Topic model prompt has {len(token_ids)} tokens; limit is {config.max_tokens}. Input was not truncated."
        )
    return payload


async def classify_topics(
    *,
    payload: Sanitized[dict[str, Any]],
    model: str,
    deployment_url: str,
    api_key: str,
    question_ids: list[str],
) -> TopicOutput:
    """
    Run one attested topic request and validate schema, question IDs and completion.

    Use an httpx2 client owned and closed by the OpenAI SDK. Preserve the prepared
    schema and max_tokens policy, disable retries, and translate model API or output
    validation failures into TopicOutputError. The caller owns deployment readiness
    and activity tracking.
    """
    clean_payload = require_sanitized(payload)
    messages = clean_payload["messages"]
    # The SDK owns and closes its HTTP client, including on failed requests.
    async with AsyncOpenAI(
        base_url=deployment_url.rstrip("/") + "/",
        api_key=api_key,
        http_client=httpx2.AsyncClient(timeout=120),
        max_retries=0,
    ) as client:
        try:
            result = await topic_agent.run(
                messages[1]["content"],
                model=topic_model(model, client),
                deps=TopicDependencies(messages[0]["content"], question_ids),
                model_settings={
                    "temperature": clean_payload["temperature"],
                    # Preserve the prepared wire schema (including titles)
                    # and the deployment's original token-budget parameter.
                    "extra_body": {
                        "max_tokens": clean_payload["max_tokens"],
                        "response_format": clean_payload["response_format"],
                    },
                },
            )
        except (ModelAPIError, UnexpectedModelBehavior) as exc:
            raise TopicOutputError(
                "Topic classification failed or returned invalid topic output"
            ) from exc
        # The checkpoint still generates request_topics, but the application uses
        # only question descriptions. Category assignment happens locally afterward.
        return TopicOutput.from_generated(result.output)
