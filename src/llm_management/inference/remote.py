"""
Remote encoder execution with sanitized inputs and local classification heads.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx2

from ..sanitization import Sanitized, require_sanitized
from .errors import ClassifierOutputError

logger = logging.getLogger(__name__)


def deployment_route_pending(response: httpx2.Response) -> bool:
    """
    Recognize the managed gateway's missing deployment route after replica startup.

    Match the observed structured error, not arbitrary 404s such as an unknown
    model or endpoint. Replica readiness can precede gateway route propagation.
    """
    if response.status_code != 404:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return (
        isinstance(error, dict)
        and error.get("type") == "not_found_error"
        and error.get("message") == "deployment not found"
    )


async def fetch_embeddings(
    client: httpx2.AsyncClient,
    *,
    url: str,
    model: str,
    texts: list[str],
) -> httpx2.Response:
    """
    Submit one batch, allowing the managed gateway up to 30 seconds of retry delay.

    Retry only its specific missing-deployment response, with delays of 1, 2, 4,
    8 and 15 seconds. Such a response has not reached the model. Keep the caller's
    inference lease throughout retries; cancellation propagates through the async
    sleep/request. Other failures and exhausted retries retain HTTPStatusError.
    """
    for attempt in range(6):
        response = await client.post(
            url, json={"model": model, "input": texts, "encoding_format": "float"}
        )
        if attempt < 5 and deployment_route_pending(response):
            delay = min(2**attempt, 15)
            logger.info(
                "Encoder deployment route is not available yet; retrying in %ss.",
                delay,
            )
            await asyncio.sleep(delay)
            continue
        if response.is_error:
            logger.warning(
                "Remote encoder request failed (HTTP %s).", response.status_code
            )
        response.raise_for_status()
        return response
    raise AssertionError("Embedding retry loop must return or raise")


async def classify_remote(
    *,
    texts: Sanitized[list[str]],
    model: str,
    deployment_url: str,
    api_key: str,
    batch_size: int = 8,
    embedding_head: Callable[[list[list[float]]], list[list[float]]],
) -> list[list[float]]:
    """
    Fetch encoder embeddings for attested text and apply the supplied local head.

    Submit ordered batches with server-owned credentials. Validate response indices
    before restoring input order, convert embedding values to floats and run the head
    in a worker thread. Return accumulated probability rows; HTTP errors propagate
    and malformed embedding responses raise ClassifierOutputError.
    """
    clean_texts = require_sanitized(texts)
    base = deployment_url.rstrip("/")
    url = base + "/embeddings"
    rows = []
    async with httpx2.AsyncClient(
        timeout=120, headers={"Authorization": f"Bearer {api_key}"}
    ) as client:
        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            response = await fetch_embeddings(client, url=url, model=model, texts=batch)
            try:
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(batch))):
                    raise ClassifierOutputError(
                        "Remote classifier returned missing or duplicate indices"
                    )
                vectors = [[float(p) for p in item["embedding"]] for item in data]
                rows.extend(await asyncio.to_thread(embedding_head, vectors))
            except (KeyError, TypeError, ValueError) as exc:
                raise ClassifierOutputError(
                    "Remote classifier returned an invalid response"
                ) from exc
    return rows
