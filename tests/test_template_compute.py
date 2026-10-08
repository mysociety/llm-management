"""Cleanup checks for temporary template builder and test VMs."""

import json

import pytest

from llm_management.templates import compute


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
    state = {"name": "llm-template-test", "zone": "at-vie-2"}
    client = FakeCompute(state["name"])
    path = tmp_path / "state.json"
    compute.cleanup(client, state, path)
    assert client.deleted == ["ours", "our-group", state["name"]]
    assert client.instances == [{"name": "unrelated", "id": "other"}]
    assert json.loads(path.read_text())["cleanup_complete"] is True
    compute.cleanup(client, state, path)
    assert len(client.deleted) == 3


def test_failed_vm_deletion_preserves_its_access_resources(tmp_path):
    state = {"name": "llm-template-test"}
    client = FakeCompute(state["name"], fail_delete=True)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        compute.cleanup(client, state, tmp_path / "state.json")
    assert client.keys and client.groups
    assert not state.get("cleanup_complete")


def test_cleanup_refuses_unrelated_manifest(tmp_path):
    client = FakeCompute("production")
    with pytest.raises(ValueError, match="Not an llm-management template"):
        compute.cleanup(client, {"name": "production"}, tmp_path / "state.json")
    assert client.deleted == []
