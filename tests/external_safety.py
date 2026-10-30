"""
Isolation and ownership checks for tests that use real cloud resources.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from uuid import uuid4

import pytest

from llm_management.deployments.compute import ComputeDeployment
from llm_management.settings import settings


@contextmanager
def isolated_external_run() -> Generator[str, None, None]:
    """
    Reject operational roles and retain recovery state under a unique test role.
    """
    if not re.fullmatch(r"test(?:_[A-Za-z0-9_-]+)?", settings.server_role):
        raise pytest.UsageError("Live tests require SERVER_ROLE=test or test_<name>")
    role = f"test_{uuid4().hex}"
    directory = settings.compute_state_dir / "pytest" / role
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "server_role", role)
        patch.setattr(settings, "compute_state_dir", directory)
        patch.setenv("SERVER_ROLE", role)
        patch.setenv("COMPUTE_STATE_DIR", str(directory))
        print(f"Live test role: {role}; recovery directory: {directory}")
        yield role


class OwnedComputeDeployment(ComputeDeployment):
    """
    Refuse existing resources and check VM identity on every cleanup path.
    """

    def __init__(self, config, role: str) -> None:
        if not re.fullmatch(r"test_[a-f0-9]{32}", role) or settings.server_role != role:
            raise RuntimeError("Compute tests require an isolated external test run")
        super().__init__(config)
        self.claimed = False

    def ensure(self) -> None:
        """
        Refuse to reuse an existing VM before opening a tunnel or sending requests.
        """
        from llm_management.deployments.compute import get_client

        if not self.claimed:
            self.require_empty(get_client(self.recipe.zone))
        super().ensure()

    def require_empty(self, client) -> None:
        """
        Never adopt a VM, access resource or manifest from another run.
        """
        if (
            self.path.exists()
            or self.instance(client) is not None
            or any(
                key.get("name") == self.name
                for key in client.list_ssh_keys().get("ssh-keys", [])
            )
            or any(
                group.get("name") == self.name
                for group in client.list_security_groups().get("security-groups", [])
            )
        ):
            raise RuntimeError("Refusing pre-existing Compute test resources")

    def provision(self, client, state):
        self.require_empty(client)
        self.claimed = True
        return super().provision(client, state)

    def delete_resources(self, client, state) -> None:
        """
        Guard explicit deletion, app shutdown and failed-startup cleanup alike.

        An unrecorded VM is retained for manual recovery rather than deleted by name.
        """
        if not self.claimed:
            return
        if state.name != self.name or state.zone != self.recipe.zone:
            raise RuntimeError("Refusing cleanup of an unrelated Compute manifest")
        self.settle_creation(client, state, self.instance(client))
        instance = self.instance(client)
        if instance is not None and instance["id"] != state.instance_id:
            raise RuntimeError("Refusing cleanup of a VM not recorded by this test")
        super().delete_resources(client, state)
