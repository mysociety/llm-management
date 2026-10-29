"""
Regression comparisons must reject missing or extra items without enforcing order.
"""

from __future__ import annotations

import pytest

from llm_management.foi.examples import validation_examples
from tests.foi_validation import (
    assert_extraction_matches,
    assert_response_matches,
    assert_topics_match,
)


def test_response_comparison_accepts_reordering():
    """
    Out-of-order decisions preserve their full question and source references.
    """
    expected = validation_examples().responses[-1].expected
    actual = expected.model_copy(deep=True)
    actual.outcomes.reverse()
    assert_response_matches(actual, expected)


@pytest.mark.parametrize("change", ["missing", "duplicate", "scope"])
def test_response_comparison_rejects_changed_items(change):
    """
    Counts and full nested item contents remain part of the semantic expectation.
    """
    expected = validation_examples().responses[-1].expected
    actual = expected.model_copy(deep=True)
    if change == "missing":
        actual.outcomes.pop()
    elif change == "duplicate":
        actual.outcomes.append(actual.outcomes[0].model_copy(deep=True))
    else:
        actual.outcomes[0].scope.question_id = "q9"
    with pytest.raises(AssertionError, match="Semantic outcomes mismatch"):
        assert_response_matches(actual, expected)


def test_extraction_comparison_reports_whitespace_separately():
    """
    Whitespace changes fail exact comparison with a distinct diagnostic category.
    """
    expected = validation_examples().requests[0].expected
    actual = expected.model_copy(deep=True)
    actual.questions[0].text += "  "
    with pytest.raises(AssertionError, match="Whitespace-only"):
        assert_extraction_matches(actual, expected)


@pytest.mark.parametrize(
    "example", validation_examples().requests, ids=lambda case: case.id
)
def test_reviewed_topic_wording_passes_and_records_gold_difference(example):
    """
    Accept each approved complete output while keeping exact dataset fidelity visible.
    """
    properties = {}
    for alternative in example.accepted_classifications:
        assert_topics_match(
            alternative.output,
            example.classification,
            properties.__setitem__,
            accepted=example.accepted_classifications,
        )
        assert properties["exact_topic_match"] is False
        assert properties["topic_match"] == "reviewed"
    assert_topics_match(
        example.classification,
        example.classification,
        properties.__setitem__,
        accepted=example.accepted_classifications,
    )
    assert properties["exact_topic_match"] is True
    assert properties["topic_match"] == "gold"


def test_reviewed_r3_reconstruction_passes_and_records_gold_difference():
    """
    Both R3 placements are accepted; the dataset reconstruction remains distinguishable.
    """
    example = validation_examples().requests[2]
    properties = {}
    assert_extraction_matches(
        example.accepted_extractions[0].output,
        example.expected,
        accepted=example.accepted_extractions,
        record_property=properties.__setitem__,
    )
    assert properties["exact_extraction_match"] is False
    assert properties["extraction_match"] == "reviewed"
    assert_extraction_matches(
        example.expected,
        example.expected,
        accepted=example.accepted_extractions,
        record_property=properties.__setitem__,
    )
    assert properties["exact_extraction_match"] is True
    assert properties["extraction_match"] == "gold"


@pytest.mark.parametrize(
    "change", ["subject", "regime", "question_id", "extra_question"]
)
def test_reviewed_topics_still_reject_unapproved_changes(change):
    """
    Approval of one wording variant does not permit other subjects or contract changes.
    """
    example = validation_examples().requests[0]
    actual = example.accepted_classifications[0].output.model_copy(deep=True)
    if change == "subject":
        actual.questions[0].topic = "school admissions"
    elif change == "regime":
        actual.questions[0].regime = "SAR"
    elif change == "question_id":
        actual.questions[0].question_id = "q9"
    else:
        actual.questions.append(
            actual.questions[0].model_copy(update={"question_id": "q99"})
        )
    with pytest.raises(AssertionError):
        assert_topics_match(
            actual,
            example.classification,
            lambda key, value: None,
            accepted=example.accepted_classifications,
        )


@pytest.mark.parametrize("change", ["missing_qualification", "extra_question"])
def test_reviewed_extraction_still_rejects_content_changes(change):
    """
    Moving qualifiers is allowed only in the reviewed complete reconstruction.
    """
    example = validation_examples().requests[2]
    actual = example.accepted_extractions[0].output.model_copy(deep=True)
    if change == "missing_qualification":
        actual.questions[0].text = example.expected.questions[0].text
    else:
        extra = actual.questions[0].model_copy(deep=True)
        extra.question_id = "q2"
        actual.questions.append(extra)
    with pytest.raises(AssertionError, match="Semantic extraction mismatch"):
        assert_extraction_matches(
            actual, example.expected, accepted=example.accepted_extractions
        )
