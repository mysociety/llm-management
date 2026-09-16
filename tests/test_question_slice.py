import pytest
from llm_management.foi.question_slice import (
    contextual_inputs,
    segment_request,
    build_extraction_result,
)


def test_wrapped_context_and_list_boundaries():
    units = segment_request(
        "Dear Council,\n\nPlease provide the number of\ncomplaints received.\n\nA. Waste collection\nB. Road maintenance\n\nThank you."
    )
    assert [u.text for u in units] == [
        "Dear Council,",
        "Please provide the number of complaints received.",
        "A. Waste collection",
        "B. Road maintenance",
        "Thank you.",
    ]
    assert [u.kind for u in units][2:4] == ["list_item", "list_item"]
    assert (
        contextual_inputs(units)[1]
        == "[PREVIOUS] Dear Council,\n[CURRENT] Please provide the number of complaints received.\n[NEXT] A. Waste collection"
    )


@pytest.mark.parametrize(
    ("labels", "expected_indices", "promoted", "status"),
    [
        ([1, 3, 0, 3], [[1, 3]], 1, "questions_found"),
        ([3, 3, 3, 3], [[0, 1, 2, 3]], 0, "questions_found"),
        ([3, 2, 0, 3], [[1, 3]], None, "uncertain"),
        ([1, 0, 1, 0], [], None, "no_questions_found"),
        ([1, 2, 3, 0], [[1, 2]], None, "questions_found"),
    ],
)
def test_continuation_recovery_preserves_predictions(
    labels, expected_indices, promoted, status
):
    from llm_management.foi.schemas import UNIT_LABELS

    rows = [[0.97 if i == label else 0.01 for i in range(4)] for label in labels]
    texts = [
        "Dear Council,",
        "Please provide the totals.",
        "Thank you.",
        "For last year.",
    ]
    result = build_extraction_result(
        segment_request("\n\n".join(texts)),
        rows,
        backend="cpu",
        model="test",
        revision="pinned",
    ).model_dump(mode="json")
    assert result["extraction_status"] == status
    assert result["promoted_continuation_index"] == promoted
    assert [q["unit_indices"] for q in result["questions"]] == expected_indices
    assert [q["text"] for q in result["questions"]] == [
        " ".join(texts[i] for i in indices) for indices in expected_indices
    ]
    assert [p["label"] for p in result["unit_predictions"]] == [
        UNIT_LABELS[i] for i in labels
    ]
    for prediction in result["unit_predictions"]:
        assert prediction["confidence"] == 0.97
        assert prediction["probabilities"][prediction["label"]] == 0.97
    if promoted is not None:
        assert result["orphan_continuation_indices"] == [
            i for i, label in enumerate(labels) if label == 3
        ]
        assert all(
            texts[i] not in result["additional_text"] for i in expected_indices[0]
        )
