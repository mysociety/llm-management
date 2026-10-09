# SPDX-License-Identifier: MIT
"""
Standalone V1 binary text logistic inference; requires only Pydantic 2 and Python's stdlib.

The V1 types describe the binary JSON format, independently of model release
names. Local adaptation: the runtime Unicode version check is omitted to support
the service's Python 3.11 runtime; normalization and tokenization are unchanged.
Future formats should have separate types and explicit conversion functions
before being passed to this predictor. Raw correspondence is normalized here.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class V1Model(BaseModel):
    """
    Keep the V1 file contract immutable and reject unrecognized fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class LogisticTrainingConfigV1(V1Model):
    """
    Preserve training metadata without depending on a training library.
    """

    minimum_ngram: int = Field(default=1, ge=1)
    maximum_ngram: int = Field(default=3, ge=1)
    minimum_document_frequency: int = Field(default=2, ge=1)
    maximum_features: int = Field(default=30000, ge=1)
    inverse_regularization: FiniteFloat = Field(default=1.0, gt=0)
    maximum_iterations: int = Field(default=500, ge=1)
    class_weight: Literal["balanced"] | None = "balanced"
    random_seed: int = Field(default=0, ge=0, le=4294967295)
    solver: Literal["liblinear"] = "liblinear"

    @model_validator(mode="after")
    def validate_ngram_range(self) -> Self:
        """
        Reject reversed training feature ranges.
        """

        if self.maximum_ngram < self.minimum_ngram:
            raise ValueError("Maximum ngram size must be at least the minimum.")
        return self


class LogisticProvenanceV1(V1Model):
    """
    Retain the original training identity and historical-data boundary.
    """

    training_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sklearn_version: str
    training_documents: int = Field(ge=1)
    config: LogisticTrainingConfigV1
    data_role: Literal["training"] = "training"
    frozen_test_consumed: Literal[False] = False


class LogisticFeaturesV1(V1Model):
    """
    Fix the tokenization contract and ordered binary word-ngram vocabulary.
    """

    tokenizer: Literal["python-unicode-word-v1"] = "python-unicode-word-v1"
    lowercase: Literal[True] = True
    token_pattern: Literal[r"(?u)\b\w\w+\b"] = r"(?u)\b\w\w+\b"
    binary: Literal[True] = True
    unicode_version: str
    minimum_ngram: int = Field(ge=1)
    maximum_ngram: int = Field(ge=1)
    vocabulary: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_vocabulary(self) -> Self:
        """
        Validate the feature range and unique nonempty vocabulary entries.
        """

        if self.maximum_ngram < self.minimum_ngram:
            raise ValueError("Maximum ngram size must be at least the minimum.")
        if len(set(self.vocabulary)) != len(self.vocabulary) or any(
            not term for term in self.vocabulary
        ):
            raise ValueError("Vocabulary entries must be unique and nonempty.")
        return self

    @cached_property
    def vocabulary_indices(self) -> dict[str, int]:
        """
        Build lookup indices once per loaded model, outside serialized fields.
        """

        return {term: index for index, term in enumerate(self.vocabulary)}

    @cached_property
    def token_regex(self) -> re.Pattern[str]:
        """
        Compile the fixed Unicode tokenizer once per loaded model.
        """

        return re.compile(self.token_pattern)


class LogisticClassifierV1(V1Model):
    """
    Describe the positive-class output, with constant-model compatibility.
    """

    label: str = Field(min_length=1, pattern=r"\S")
    coefficients: tuple[FiniteFloat, ...]
    intercept: FiniteFloat
    constant_probability: Literal[0, 1] | None = None
    decision_threshold: FiniteFloat = Field(default=0.0, ge=0, le=0)
    decision_comparison: Literal["strictly_greater"] = "strictly_greater"

    @model_validator(mode="after")
    def validate_constant(self) -> Self:
        """
        Reject inconsistent constant-output metadata.
        """

        if self.constant_probability is not None and (
            self.intercept != 0 or any(self.coefficients)
        ):
            raise ValueError("Constant models require zero weights and intercept.")
        return self


class LogisticWeightsV1(V1Model):
    """
    Validate a single-output V1 weights payload using the training export layout.
    """

    schema_version: Literal["1"]
    model_kind: Literal["multilabel_logistic_regression"]
    input_kind: Literal["prepared_text"]
    features: LogisticFeaturesV1
    classifiers: tuple[LogisticClassifierV1, ...] = Field(min_length=1, max_length=1)
    provenance: LogisticProvenanceV1

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """
        Ensure each vocabulary column has a corresponding coefficient.
        """

        if len(self.classifiers[0].coefficients) != len(self.features.vocabulary):
            raise ValueError("Coefficient dimensions must match the vocabulary.")
        return self


class BinaryTextLogisticModelV1(V1Model):
    """
    Explicit V1 input preserving weights, outcome labels and threshold policy.

    The score always belongs to true_label. A true result means that score meets
    the threshold; false_label names the other outcome. Renaming labels does not
    reverse the learned classifier. Multilabel unions require explicit conversion.
    """

    schema_version: Literal["1"]
    name: str
    kind: Literal["logistic"]
    true_label: str = Field(min_length=1, pattern=r"\S")
    false_label: str = Field(min_length=1, pattern=r"\S")
    logistic: LogisticWeightsV1
    rules: None = None
    constant: None = None
    probability_threshold: FiniteFloat = Field(default=0.5, ge=0, le=1)
    inclusive: bool = True
    training_ids_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    historical_output_union: Literal[False] = False

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        """
        Require distinct outcome labels and weights aligned with the true label.
        """

        if self.true_label == self.false_label:
            raise ValueError("True and false labels must differ.")
        if self.logistic.classifiers[0].label != self.true_label:
            raise ValueError("Classifier label must match true_label.")
        return self


class BinaryPredictionV1(V1Model):
    """
    Return the positive-class score, boolean result and selected outcome label.
    """

    model_name: str
    positive_score: FiniteFloat = Field(ge=0, le=1)
    result: bool
    label: str = Field(min_length=1, pattern=r"\S")
    threshold: FiniteFloat = Field(ge=0, le=1)


def load_model_v1(
    path: Path, expected_sha256: str | None = None
) -> BinaryTextLogisticModelV1:
    """
    Load labelled binary V1 JSON, optionally checking a pinned artifact checksum first.
    """

    payload = path.read_bytes()
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("Logistic model checksum mismatch.")
    return BinaryTextLogisticModelV1.model_validate_json(payload)


def normalize_text_v1(text: str) -> str:
    """
    Preserve V1 placeholder removal, NFKC, case folding and whitespace collapse.
    """

    without_placeholders = re.sub(r"<[^<>]+>", " ", text, flags=re.MULTILINE)
    return " ".join(
        unicodedata.normalize("NFKC", without_placeholders).casefold().split()
    )


def predict_binary(text: str, model: BinaryTextLogisticModelV1) -> BinaryPredictionV1:
    """
    Classify raw text using an explicitly supplied V1 model and cutoff.

    Cache vocabulary indices on the model so callers can load once and reuse it.
    Future file versions must be explicitly converted to a supported model type.
    """

    if not isinstance(model, BinaryTextLogisticModelV1):
        raise TypeError(
            "Expected BinaryTextLogisticModelV1; convert other formats explicitly."
        )
    features = model.logistic.features
    classifier = model.logistic.classifiers[0]
    if classifier.constant_probability is not None:
        probability = float(classifier.constant_probability)
    else:
        tokens = features.token_regex.findall(normalize_text_v1(text).lower())
        active: set[int] = set()
        vocabulary = features.vocabulary_indices
        for size in range(features.minimum_ngram, features.maximum_ngram + 1):
            for start in range(len(tokens) - size + 1):
                index = vocabulary.get(" ".join(tokens[start : start + size]))
                if index is not None:
                    active.add(index)
        margin = 0.0
        for index in sorted(active):
            margin += classifier.coefficients[index]
        margin += classifier.intercept
        if margin >= 0:
            probability = 1.0 / (1.0 + math.exp(-margin))
        else:
            exponential = math.exp(margin)
            probability = exponential / (1.0 + exponential)
    result = (
        probability >= model.probability_threshold
        if model.inclusive
        else probability > model.probability_threshold
    )
    return BinaryPredictionV1(
        model_name=model.name,
        positive_score=probability,
        result=result,
        label=model.true_label if result else model.false_label,
        threshold=model.probability_threshold,
    )
