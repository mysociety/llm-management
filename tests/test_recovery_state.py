"""
Recovery manifests retain JSON compatibility and validate identity before use.
"""

import json

import pytest
from pydantic import ValidationError

from llm_management.templates.state import ComputeState, TemplateState


def test_legacy_compute_manifest_round_trips_unknown_fields(tmp_path):
    path = tmp_path / "state.json"
    legacy = {
        "name": "llm-compute-at-vie-2-test",
        "zone": "at-vie-2",
        "create_operation": "pending",
        "future_metadata": {"keep": True},
    }
    path.write_text(json.dumps(legacy))

    state = ComputeState.load(path)
    assert state.default_user == "ubuntu"
    assert not state.cleanup_complete
    state.save(path)

    assert json.loads(path.read_text()) == legacy
    assert not path.with_name("state.json.tmp").exists()


def test_compute_reset_clears_previous_creation_without_losing_identity(tmp_path):
    path = tmp_path / "state.json"
    state = ComputeState(
        name="llm-compute-at-vie-2-test",
        zone="at-vie-2",
        template_id="release",
        create_operation="previous",
        instance_id="deleted-vm",
        cleanup_complete=True,
    )

    state.reset_creation()
    state.save(path)

    assert json.loads(path.read_text()) == {
        "name": state.name,
        "zone": state.zone,
        "template_id": "release",
    }
    assert ComputeState.load(path).create_operation is None


def test_invalid_manifest_does_not_replace_existing_json(tmp_path):
    path = tmp_path / "state.json"
    original = '{"name": "llm-template-test", "zone": "at-vie-2"}'
    path.write_text(original)
    state = TemplateState.load(path)

    with pytest.raises(ValidationError):
        state.instance_id = ["invalid-id"]

    assert path.read_text() == original
    assert state.instance_id is None


def test_manifest_requires_identity_before_recovery(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"create_operation": "pending"}')

    with pytest.raises(ValidationError):
        ComputeState.load(path)
