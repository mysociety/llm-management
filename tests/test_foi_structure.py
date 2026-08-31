import pytest
from pydantic import ValidationError

from llm_management.agents.foi_structure import (
    Extraction,
    QuestionSpan,
    Unit,
    _validate_indices,
    normalize_text,
    reconstruct_units,
    resolve_extraction,
    segment,
)


def test_segment_is_line_first_and_ignores_blank_lines():
    units = segment("Dear Council,\n\n1. First ask\n- Second ask\nThank you")

    assert [unit.idx for unit in units] == [0, 1, 2, 3]
    assert [unit.kind for unit in units] == [
        "line",
        "list_item",
        "list_item",
        "line",
    ]
    assert units[1].text == "1. First ask"


def test_segment_long_prose_without_breaking_foi_abbreviations():
    text = (
        "Please provide records from Acme Ltd. "
        "Also provide records under s.1(1)(a). Finally give file No. 12."
    )

    units = segment(text, long_line_chars=20)

    assert [unit.text for unit in units] == [
        "Please provide records from Acme Ltd.",
        "Also provide records under s.1(1)(a).",
        "Finally give file No. 12.",
    ]
    assert all(unit.kind == "sentence" for unit in units)


def test_list_items_are_not_sentence_split_even_when_long():
    text = "1. " + "A long enumerated request. " * 20

    units = segment(text, long_line_chars=20)

    assert len(units) == 1
    assert units[0].kind == "list_item"


def test_normalize_text_handles_unicode_punctuation_and_hard_wraps():
    source = "The council’s “air–quality”\nrecords"
    plain = "The councils air-quality records"

    assert normalize_text(source) == normalize_text(plain)


def test_reconstruct_units_removes_only_list_markers():
    units = [
        Unit(idx=0, text="1. What records are held?", kind="list_item"),
        Unit(idx=1, text="For the 2024-25 year.", kind="line"),
    ]

    assert reconstruct_units(units, [0, 1], remove_list_markers=True) == (
        "What records are held?\nFor the 2024-25 year."
    )


def test_resolve_extraction_uses_source_text_and_preserves_classification():
    units = segment("1. Air-quality monitoring reports\nFor the calendar year 2024")
    extraction = Extraction(
        questions=[QuestionSpan(units=[0, 1], ir_type="EIR")],
    )

    questions, additional_info = resolve_extraction(extraction, units)

    assert questions[0].text == (
        "Air-quality monitoring reports\nFor the calendar year 2024"
    )
    assert questions[0].ir_type == "EIR"
    assert additional_info is None


@pytest.mark.parametrize(
    ("extraction", "expected_error"),
    [
        (
            Extraction(questions=[QuestionSpan(units=[1, 0], ir_type="FOI")]),
            "unique and ascending",
        ),
        (
            Extraction(questions=[QuestionSpan(units=[2], ir_type="FOI")]),
            "invalid indices",
        ),
        (
            Extraction(
                questions=[
                    QuestionSpan(units=[0], ir_type="FOI"),
                    QuestionSpan(units=[0], ir_type="FOI"),
                ]
            ),
            "reuses indices",
        ),
        (
            Extraction(
                questions=[QuestionSpan(units=[0], ir_type="FOI")],
                additional_info_units=[0],
            ),
            "overlap",
        ),
    ],
)
def test_invalid_span_selections_are_rejected(extraction, expected_error):
    errors = _validate_indices(extraction, unit_count=2)

    assert any(expected_error in error for error in errors)


def test_every_list_item_must_be_selected_as_its_own_question():
    units = segment("1. First ask\n2. Second ask")

    missing = Extraction(
        questions=[QuestionSpan(units=[0], ir_type="FOI")],
    )
    combined = Extraction(
        questions=[QuestionSpan(units=[0, 1], ir_type="FOI")],
    )

    missing_errors = _validate_indices(
        missing,
        unit_count=len(units),
        source_units=units,
    )
    combined_errors = _validate_indices(
        combined,
        unit_count=len(units),
        source_units=units,
    )

    assert any("missing indices: [1]" in error for error in missing_errors)
    assert any(
        "combines separate list items: [0, 1]" in error for error in combined_errors
    )


def test_list_item_question_rejects_preceding_but_allows_following_prose():
    units = segment("Request context\n1. First ask\nPlease use calendar years")
    preceding = Extraction(
        questions=[QuestionSpan(units=[0, 1], ir_type="FOI")],
    )
    following = Extraction(
        questions=[QuestionSpan(units=[1, 2], ir_type="FOI")],
    )

    preceding_errors = _validate_indices(preceding, len(units), source_units=units)
    following_errors = _validate_indices(following, len(units), source_units=units)

    assert any("includes prose before" in error for error in preceding_errors)
    assert not following_errors


def test_scottish_specific_regime_labels_are_not_exposed():
    with pytest.raises(ValidationError):
        QuestionSpan.model_validate({"units": [0], "ir_type": "FOISA"})

    with pytest.raises(ValidationError):
        QuestionSpan.model_validate({"units": [1], "ir_type": "EISR"})


def test_prose_only_question_is_rejected_when_request_has_list_items():
    units = segment("Request context\n1. First ask")
    extraction = Extraction(
        questions=[
            QuestionSpan(units=[0], ir_type="FOI"),
            QuestionSpan(units=[1], ir_type="FOI"),
        ]
    )

    errors = _validate_indices(extraction, len(units), source_units=units)

    assert any("not anchored to a list item" in error for error in errors)
