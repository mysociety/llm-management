"""
Remote encoder execution with sanitized inputs and local classification heads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx2

from ..sanitization import Sanitized, require_sanitized
from .errors import ClassifierOutputError


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
            payload = {"model": model, "input": batch, "encoding_format": "float"}
            response = await client.post(url, json=payload)
            response.raise_for_status()
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
