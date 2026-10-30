"""
Cleanup and operation failures for temporary template builder and test VMs.
"""

from __future__ import annotations

import json

import pytest

from llm_management.templates import compute
from llm_management.templates.state import TemplateState


class FakeCompute:
    def __init__(self, name, fail_delete=False):
        self.instances = [
            {"name": name, "id": "ours"},
            {"name": "unrelated", "id": "other"},
        ]
        self.groups = [{"name": name, "id": "our-group"}]
        self.keys = [{"name": name}]
        self.deleted = []
        self.fail_delete = fail_delete

    def list_instances(self):
        return {"instances": self.instances}

    def list_security_groups(self):
        return {"security-groups": self.groups}

    def list_ssh_keys(self):
        return {"ssh-keys": self.keys}

    def wait(self, operation_id, max_wait_time):
        pass

    def delete_instance(self, id):
        if self.fail_delete:
            raise RuntimeError("Provider unavailable")
        self.instances = [i for i in self.instances if i["id"] != id]
        self.deleted.append(id)
        return {"id": "operation"}

    def delete_security_group(self, id):
        self.groups = [g for g in self.groups if g["id"] != id]
        self.deleted.append(id)
        return {"id": "operation"}

    def delete_ssh_key(self, name):
        self.keys = [k for k in self.keys if k["name"] != name]
        self.deleted.append(name)
        return {"id": "operation"}


def test_cleanup_recovers_unrecorded_resources_and_is_idempotent(tmp_path):
    # A killed creation can leave a resource before its ID reaches the manifest.
    state = TemplateState(name="llm-template-test", zone="at-vie-2")
    client = FakeCompute(state.name)
    path = tmp_path / "state.json"
    compute.cleanup(client, state, path)
    assert client.deleted == ["ours", "our-group", state.name]
    assert client.instances == [{"name": "unrelated", "id": "other"}]
    assert json.loads(path.read_text())["cleanup_complete"] is True
    compute.cleanup(client, state, path)
    assert len(client.deleted) == 3


def test_failed_vm_deletion_preserves_its_access_resources(tmp_path):
    state = TemplateState(name="llm-template-test", zone="at-vie-2")
    client = FakeCompute(state.name, fail_delete=True)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        compute.cleanup(client, state, tmp_path / "state.json")
    assert client.keys and client.groups
    assert not state.cleanup_complete


def test_cleanup_refuses_unrelated_manifest(tmp_path):
    client = FakeCompute("production")
    with pytest.raises(ValueError, match="Not an llm-management template"):
        compute.cleanup(
            client,
            TemplateState(name="production", zone="at-vie-2"),
            tmp_path / "state.json",
        )
    assert client.deleted == []


@pytest.mark.parametrize(
    ("state", "detail"),
    [
        ("failure", "Insufficient capacity: no suitable host found"),
        ("timeout", "Instance creation timed out"),
        ("failure", None),
    ],
)
def test_wait_recovers_message_from_sdk_missing_reason(
    state: str, detail: str | None
) -> None:
    """
    Exercise the installed SDK's failure path without contacting the provider.
    """
    from unittest.mock import Mock

    from exoscale.api.v2 import Client

    from llm_management.deployments.errors import LLMManagementError

    operation = {"id": "create-vm", "state": state}
    if detail is not None:
        operation["message"] = detail
    client = Mock(spec=Client)
    client.get_operation.return_value = operation
    client.wait.side_effect = lambda operation_id, **kwargs: Client.wait(
        client, operation_id, **kwargs
    )

    with pytest.raises(LLMManagementError) as caught:
        compute.wait(client, {"id": "create-vm"})

    assert str(caught.value) == (
        f"Compute operation create-vm {state}: "
        f"{detail or 'Provider supplied no failure detail'}"
    )
    assert isinstance(caught.value.__cause__, KeyError)
    client.wait.assert_called_once_with("create-vm", max_wait_time=1200)


def test_wait_preserves_successful_sdk_result() -> None:
    """
    Return completed operations through the existing SDK polling path.
    """
    from unittest.mock import Mock

    from exoscale.api.v2 import Client

    client = Mock(spec=Client)
    completed = {"id": "create-vm", "state": "success"}
    client.get_operation.return_value = completed
    client.wait.side_effect = lambda operation_id, **kwargs: Client.wait(
        client, operation_id, **kwargs
    )
    assert compute.wait(client, {"id": "create-vm"}) == completed
    client.get_operation.assert_called_once_with(id="create-vm")


def test_wait_does_not_hide_unrelated_sdk_key_errors() -> None:
    """
    Only handle the SDK's missing failure-detail field.
    """
    from unittest.mock import Mock

    client = Mock()
    original = KeyError("state")
    client.wait.side_effect = original
    with pytest.raises(KeyError) as caught:
        compute.wait(client, {"id": "create-vm"})
    assert caught.value is original
    client.get_operation.assert_not_called()


def test_wait_requires_terminal_failure_before_recovering_detail() -> None:
    """
    Do not turn a pending operation into a confirmed provisioning failure.
    """
    from unittest.mock import Mock

    client = Mock()
    original = KeyError("reason")
    client.wait.side_effect = original
    client.get_operation.return_value = {"state": "pending"}
    with pytest.raises(KeyError) as caught:
        compute.wait(client, {"id": "create-vm"})
    assert caught.value is original
