"""Config-driven HTTP and native Pydantic AI smoke checks."""

import asyncio
import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, create_model

from .config import TemplateRecipe


def validate_health(health: dict, recipe: TemplateRecipe) -> None:
    for key, expected in {
        "backend": recipe.backend,
        "model": recipe.model,
        "revision": recipe.revision,
        "max_length": recipe.max_length,
    }.items():
        if health.get(key) != expected:
            raise RuntimeError(
                f"Server health {key}={health.get(key)!r}; expected {expected!r}"
            )


def probe(base_url: str, directory: Path, recipe: TemplateRecipe) -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.system_one import SystemOneModel
    from pydantic_ai.providers.system_one import SystemOneProvider

    results = []
    with httpx.Client(timeout=300) as client:
        for case in recipe.smoke:
            body = {
                "model": recipe.model,
                "state": case.state,
                "questions": {
                    "classification": {
                        "type": "choice",
                        "instructions": case.instructions,
                        "criteria": case.criteria,
                    }
                },
            }
            response = client.post(base_url + "/v1/systemone", json=body)
            response.raise_for_status()
            result = response.json()
            results.append({"expected": case.expected, "response": result})
            (directory / "results.json").write_text(json.dumps(results, indent=2))
            if result["answers"]["classification"]["choice"] != case.expected:
                raise RuntimeError(
                    "System One HTTP smoke test returned an unexpected choice"
                )

    case = recipe.smoke[0]
    output_type = create_model(
        "Classification",
        classification=(
            Literal[tuple(case.criteria)],
            Field(
                description=case.instructions
                + " "
                + "; ".join(f"{k}: {v}" for k, v in case.criteria.items())
            ),
        ),
    )

    async def classify():
        async with Agent(
            SystemOneModel(recipe.model, provider=SystemOneProvider(base_url=base_url)),
            output_type=output_type,
            retries=0,
            model_settings={"timeout": 300},
        ) as agent:
            return await agent.run(case.state)

    result = asyncio.run(classify())
    if result.output.classification != case.expected:
        raise RuntimeError(
            "Native Pydantic AI smoke test returned an unexpected choice"
        )
    (directory / "pydantic-ai.json").write_text(result.output.model_dump_json())
