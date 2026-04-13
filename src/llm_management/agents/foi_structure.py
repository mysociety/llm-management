import string
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel


def normalize_text(text: str) -> str:
    """
    Lowercase text and remove all punctuation, preserving whitespace.
    """
    return text.lower().translate(str.maketrans("", "", string.punctuation))


@dataclass
class RequestDeps:
    """Dependencies holding the original request text for validation."""

    original_text: str


class Question(BaseModel):
    text: str
    ir_type: Literal["FOI", "EIR", "SAR", "OTHER_PERSONAL", "OTHER"]


class FOIRequest(BaseModel):
    authority: str = Field(
        ..., description="The public authority to which the FOI request is addressed"
    )
    questions: list[Question] = Field(
        ..., description="A list of questions included in the FOI request"
    )
    additional_info: str | None = Field(
        None, description="Any additional information provided in the request"
    )

    short_description: str = Field(
        ...,
        description="A short description of the request, ideally one sentence - on the main topic of the request",
    )
    keywords: list[str] = Field(
        ..., description="A list of exactly 5 keywords summarizing the request"
    )


SYSTEM_PROMPT = """
You are an assistant to create structured data about information requests from unstructured text.
You will be given an information request and need to extract key information from it to populate a structured response. 

When extracting text about questions, do not paraphrase the question, but extract the exact wording of the question.
If there is additional clarification text for a question, include this in the question.

Information request types:

Freedom of Information (FOI) - request for recorded information held by public authorities in general (not related to environmental information or personal data)
Environmental Information Regulations (EIR) - requests related to environmental information, such as pollution, emissions, land use, waste, water, air, energy, and the effects of policies or projects on the environment
Subject Access Requests (SAR) - request for personal data held about an individual by organizations
Other Personal Information Requests (OTHER_PERSONAL) - request for personal information that does not fall under the SAR regime (e.g. requests for data about own immigration status)
Other (OTHER) - request that does not fall under any of the above categories.

"""

stucture_agent = Agent(
    output_type=FOIRequest,
    deps_type=RequestDeps,
    system_prompt=SYSTEM_PROMPT,
    retries=3,
    model_settings={"temperature": 0.1},
)


@stucture_agent.output_validator
async def validate_extracted_text(
    ctx: RunContext[RequestDeps], result: FOIRequest
) -> FOIRequest:
    original = ctx.deps.original_text
    normalized_original = normalize_text(original)
    missing = []
    for i, question in enumerate(result.questions, 1):
        if normalize_text(question.text) not in normalized_original:
            missing.append(
                f"Question {i}: {question.text!r} not found in original text"
            )
    if missing:
        print("Retrying because of missing question")
        print("\n".join(missing))
        raise ModelRetry(
            "Extracted question text must appear verbatim in the original request. "
            "Do not paraphrase. The following questions were not found:\n"
            + "\n".join(missing)
        )
    return result


async def extract_structure_from_request(
    *, model: OpenAIChatModel, request_text: str
) -> FOIRequest:
    """Helper function to run the structure agent asynchronously."""
    result = await stucture_agent.run(
        request_text, model=model, deps=RequestDeps(original_text=request_text)
    )
    return result.output
