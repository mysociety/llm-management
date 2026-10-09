import pytest
from pydantic import ValidationError

from llm_management.foi import question_extractor
from llm_management.foi.model_spec import FOIModelSettings
from llm_management.settings import Settings


def test_model_settings_read_environment_and_derive_question_limit(monkeypatch):
    monkeypatch.setenv("QUESTION_SLICE_MODEL", "example/question-slice")
    monkeypatch.setenv("QUESTION_SLICE_INPUT_LIMIT", "512")
    monkeypatch.setenv("FOI_TOPIC_MODEL", "example/question-topics")
    monkeypatch.setenv("FOI_TOPIC_REVISION", "pinned-revision")
    monkeypatch.setenv("FOI_TOPIC_OUTPUT_LIMIT", "1024")
    configured = Settings(_env_file=None)
    assert isinstance(configured, FOIModelSettings)
    assert configured.question_slice_model == "example/question-slice"
    assert configured.question_slice_input_limit == 512
    assert configured.foi_topic_model == "example/question-topics"
    assert configured.foi_topic_revision == "pinned-revision"
    assert configured.foi_topic_max_questions == 5
    monkeypatch.setattr(question_extractor, "settings", configured)
    assert question_extractor.completion_budget(5) == 1024
    with pytest.raises(ValueError, match="1–5"):
        question_extractor.completion_budget(6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"question_slice_input_limit": 0},
        {"foi_topic_input_limit": 0},
        {"foi_topic_output_tokens_per_question": 0},
        {"foi_topic_output_overhead": -1},
        {"foi_topic_min_output_tokens": 2049},
        {"foi_topic_output_overhead": 2048},
    ],
)
def test_invalid_model_limits_rejected(overrides):
    with pytest.raises(ValidationError):
        FOIModelSettings(**overrides)
