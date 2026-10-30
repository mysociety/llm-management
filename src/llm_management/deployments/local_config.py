"""
Data-only schemas for local resources; importing them never imports owners.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LocalDeployment(BaseModel):
    """
    Identity and validation policy shared by all local resource configurations.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str = Field(min_length=1)


class PresidioDeployment(LocalDeployment):
    """
    Configure a spaCy pipeline directly, without a catalog checkpoint reference.
    """

    loader: Literal["presidio"]
    spacy_model: str = Field(min_length=1)
    score_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)


class LocalModelDeployment(LocalDeployment):
    """
    Add a catalog checkpoint reference for model-backed local resources.

    The concrete resource schemas below inherit this field. Catalog validation
    uses this base type to identify resources whose model_ref must resolve.
    """

    model_ref: str = Field(min_length=1)


class SequenceClassifierDeployment(LocalModelDeployment):
    """
    Sequence-classifier settings; the current loader validates QuestionSlice labels.
    """

    loader: Literal["modernbert_classifier"]
    max_tokens: int = Field(gt=0)


class QuestionSliceHeadDeployment(LocalModelDeployment):
    loader: Literal["modernbert_head"]


class TopicTokenizerDeployment(LocalModelDeployment):
    loader: Literal["topic_tokenizer"]
    max_tokens: int = Field(gt=0)


class CategoryMatcherDeployment(LocalModelDeployment):
    """
    Data-only category index paired with a pinned local embedding encoder.
    """

    loader: Literal["foi_category_matcher"]
    encoder_ref: str = Field(min_length=1)
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ResponseTokenizerDeployment(LocalModelDeployment):
    """
    Pinned tokenizer for the trained FOI status chat template.
    """

    loader: Literal["response_tokenizer"]


class SARLogisticDeployment(LocalModelDeployment):
    """
    Artifact configuration for the SAR-specific logistic loader.
    """

    loader: Literal["sar_logistic"]
    artifact: str = Field(min_length=1)


class SARClassifierDeployment(LocalModelDeployment):
    loader: Literal["sar_deberta"]
    max_tokens: int = Field(gt=0)


class AutoLogisticDeployment(LocalModelDeployment):
    loader: Literal["auto_logistic"]
    artifact: str = "model.json"
    standalone_artifact: str = "standalone-model.json"


class AutoClassifierDeployment(LocalModelDeployment):
    loader: Literal["auto_deberta"]
    max_tokens: Literal[512] = 512
