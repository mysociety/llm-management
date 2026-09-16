import asyncio

from pydantic_ai.models.test import TestModel

from pydantic_ai.profiles import ModelProfile
from llm_management.agents.capital_city import capital_city_agent
from llm_management.agents.immigration_detection import (
    Classification,
    immigration_detection_agent,
)


def test_capital_city_native_output():
    async def run_agent():
        model = TestModel(
            custom_output_text='{"country":"France","city":"Paris"}',
            profile=ModelProfile(supports_json_schema_output=True),
        )
        return await capital_city_agent(model=model, country="France")

    result = asyncio.run(run_agent())

    assert result.country == "France"
    assert result.city == "Paris"


def test_immigration_detection_plain_text_output():
    async def run_agent():
        model = TestModel(custom_output_text="IMM")
        return await immigration_detection_agent(
            model=model,
            request="Please provide information about my visa application.",
        )

    result = asyncio.run(run_agent())

    assert result.classification is Classification.IMM
