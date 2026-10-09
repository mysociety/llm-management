"""Typed checkpoint identities and training limits for the FOI pipeline."""

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings


# Related runtime resources share the deployment's model-family/version name.
QUESTION_SLICE_DEPLOYMENT = "question_slice_v2"
QUESTION_SLICE_CPU_RESOURCE = f"{QUESTION_SLICE_DEPLOYMENT}_cpu"
QUESTION_SLICE_HEAD_CPU_RESOURCE = f"{QUESTION_SLICE_DEPLOYMENT}_head_cpu"


class FOIModelSettings(BaseSettings):
    question_slice_model: str = "mySociety/modernbert-question-slice-v2"
    question_slice_revision: str = "eb0436d4be96f113f35b5f891f9cc876a0f0bd6b"
    question_slice_input_limit: int = Field(default=768, ge=1)
    foi_topic_model: str = "mySociety/granite-tiny-foi-topic-grounded-v2-merged"
    foi_topic_revision: str = "c2ba6b86e43977bcb71bc90f12dc0cad42ac7e79"
    foi_topic_input_limit: int = Field(default=2048, ge=1)
    foi_topic_output_limit: int = Field(default=2048, ge=1)
    foi_topic_min_output_tokens: int = Field(default=256, ge=1)
    foi_topic_output_overhead: int = Field(default=64, ge=0)
    foi_topic_output_tokens_per_question: int = Field(default=192, ge=1)

    @computed_field
    @property
    def foi_topic_max_questions(self) -> int:
        return (
            self.foi_topic_output_limit - self.foi_topic_output_overhead
        ) // self.foi_topic_output_tokens_per_question

    @model_validator(mode="after")
    def validate_topic_budget(self):
        if self.foi_topic_min_output_tokens > self.foi_topic_output_limit:
            raise ValueError("Minimum topic output budget exceeds the output limit")
        if self.foi_topic_max_questions < 1:
            raise ValueError(
                "Topic output budget must accommodate at least one question"
            )
        return self
