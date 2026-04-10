from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel


class CapitalCityResponse(BaseModel):
    country: str
    city: str


async def capital_city_agent(
    *, model: OpenAIChatModel, country: str
) -> CapitalCityResponse:
    """
    Example agent endpoint. Takes a country name and returns its capital
    city as structured output via a pydantic-ai Agent running on the
    specified deployment.
    """
    agent = Agent(
        model,
        system_prompt=(
            "Given the name of a country, return the capital city of that country. "
            "e.g. Germany -> Berlin"
        ),
        output_type=CapitalCityResponse,
    )

    result = await agent.run(country)
    return result.output
