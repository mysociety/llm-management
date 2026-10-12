"""FOI model identities and budgets come from the validated deployment catalog."""

import pytest
from pydantic import ValidationError

from llm_management.deployments import DeploymentCatalog
from llm_management.foi import topic_classification


def test_catalog_drives_topic_budget_and_checkpoint(monkeypatch):
    data = DeploymentCatalog.load().model_dump()
    data["model"]["foi_topic"]["repo"] = "example/question-topics"
    for deployment in data["exoscale"]["deployment"]:
        if deployment.get("model_ref") == "foi_topic":
            deployment.pop("model")
    data["foi"]["topic"]["output_limit"] = 1024
    configured = DeploymentCatalog.model_validate(data)
    assert configured.get("foi_topic").model == "example/question-topics"
    assert configured.require_foi().topic.max_questions == 5
    monkeypatch.setattr(topic_classification, "get_catalog", lambda: configured)
    assert topic_classification.completion_budget(5) == 1024
    with pytest.raises(ValueError, match="1–5"):
        topic_classification.completion_budget(6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_limit": 0},
        {"output_tokens_per_question": 0},
        {"output_overhead": -1},
        {"min_output_tokens": 2049},
        {"output_overhead": 2048},
    ],
)
def test_invalid_model_limits_rejected(overrides):
    data = DeploymentCatalog.load().model_dump()
    data["foi"]["topic"].update(overrides)
    with pytest.raises(ValidationError):
        DeploymentCatalog.model_validate(data)
