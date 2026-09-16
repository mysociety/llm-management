"""Training-compatible Granite topic/regime classification for extracted questions."""

import json
from functools import lru_cache

import httpx

from .schemas import ExtractedQuestion, TopicOutput
from .model_spec import (
    GRANITE_MERGED,
    GRANITE_MERGED_REVISION,
    GRANITE_INPUT_LIMIT,
    GRANITE_OUTPUT_LIMIT,
    GRANITE_MIN_OUTPUT_TOKENS,
    GRANITE_OUTPUT_OVERHEAD,
    GRANITE_OUTPUT_TOKENS_PER_QUESTION,
    GRANITE_MAX_QUESTIONS,
)
from ..settings import settings

SYSTEM_PROMPT = (
    "For each extracted UK information-request question, identify its access regime "
    "and write a concise topic noun phrase. Return only JSON matching the supplied "
    "schema. Topics describe the subject, not the requesting action, and must omit "
    "names, identifiers, dates, and redaction markers."
)


class TopicOutputError(Exception):
    pass


def completion_budget(question_count: int) -> int:
    budget = max(
        GRANITE_MIN_OUTPUT_TOKENS,
        GRANITE_OUTPUT_OVERHEAD + GRANITE_OUTPUT_TOKENS_PER_QUESTION * question_count,
    )
    if question_count < 1 or budget > GRANITE_OUTPUT_LIMIT:
        raise ValueError(
            f"Granite supports 1–{GRANITE_MAX_QUESTIONS} questions within its {GRANITE_OUTPUT_LIMIT:,}-token output budget"
        )
    return budget


@lru_cache(maxsize=1)
def granite_tokenizer():
    from transformers import AutoTokenizer

    from huggingface_hub import snapshot_download

    # Load locally after an explicitly authenticated download. Transformers may
    # perform additional Hub metadata requests without forwarding token=.
    snapshot = snapshot_download(
        GRANITE_MERGED,
        revision=GRANITE_MERGED_REVISION,
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
    return AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, fix_mistral_regex=False
    )


def prepare_topic_request(
    request_text: str, questions: list[ExtractedQuestion]
) -> dict:
    budget = completion_budget(len(questions))
    schema = TopicOutput.model_json_schema()
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
    tokenizer = granite_tokenizer()
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, truncation=False
    )
    if len(token_ids) > GRANITE_INPUT_LIMIT:
        raise ValueError(
            f"Granite prompt has {len(token_ids)} tokens; limit is {GRANITE_INPUT_LIMIT}. Input was not truncated."
        )
    return {
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


async def classify_topics(
    *,
    payload: dict,
    model: str,
    deployment_url: str,
    api_key: str,
    question_ids: list[str],
) -> TopicOutput:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            deployment_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={**payload, "model": model},
        )
        response.raise_for_status()
    try:
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise TopicOutputError("Granite exhausted its completion budget")
        if choice.get("finish_reason") != "stop":
            raise TopicOutputError("Granite did not complete normally")
        output = TopicOutput.model_validate_json(choice["message"]["content"])
        if [q.question_id for q in output.questions] != question_ids:
            raise TopicOutputError(
                "Granite question IDs do not match extracted questions"
            )
        return output
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise TopicOutputError("Granite returned invalid topic output") from exc
