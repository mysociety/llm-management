from enum import StrEnum

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel


class Classification(StrEnum):
    IMM = "IMM"
    FOI = "FOI"


class ClassificationResponse(BaseModel):
    classification: Classification


SYSTEM_PROMPT = """
Cutting Knowledge Date: December 2023
    You are an AI assistant tasked with analyzing text to determine if it is an Immigration-related request or a general Freedom of Information (FOI) request.
    An Immigration-related request is any inquiry or correspondence related to immigration processes, visas, residency, citizenship, or any other matter concerning a person's immigration status or application.
    A Freedom of Information (FOI) request is a request for any other type of information held by public authorities, including general government operations, policies, or decisions not related to immigration.
    Instructions:
    - Carefully read the entire text of the request.
    - Identify the main subject or focus of the inquiry.
    - If the request is related to immigration matters, classify it as "IMM".
    - For all other types of requests, classify it as "FOI".
    - Do not include any text or explanation in your response.
    - Your response should be either "IMM" or "FOI" and nothing else.
    Return "IMM" if the request is related to immigration matters, otherwise return "FOI".

    Analyze the following text and determine if it is an Immigration-related request or an FOI request:
"""


async def immigration_detection_agent(
    *, model: OpenAIChatModel, request: str
) -> ClassificationResponse:
    """
    Example agent endpoint. Takes a request and returns its classification
    as a plain text response ("IMM" or "FOI") via a pydantic-ai Agent
    running on the specified deployment.
    """
    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.output_validator
    async def validate_classification(output: str) -> str:
        normalized = output.strip().upper()
        if normalized not in Classification.__members__:
            raise ModelRetry(
                "Response must be exactly 'IMM' or 'FOI' with no other text."
            )
        return normalized

    result = await agent.run(request)

    return ClassificationResponse(classification=Classification(result.output))
