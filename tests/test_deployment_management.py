"""
Direct management callers share HTTP command semantics without constructing an API.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from llm_management.bootstrap import assemble_service
from llm_management.deployments.errors import DeploymentServiceError
from llm_management.deployments.types import (
    GroupActionMemberOutcome,
    GroupActionOutcome,
    GroupMemberResult,
    GroupWarmUpResponse,
)


def test_direct_group_warmup_preserves_order_and_partial_failure(monkeypatch):
    service = assemble_service()
    outcome = GroupWarmUpResponse(
        slug="group_example",
        success=False,
        deployments=[
            GroupMemberResult(slug="second", success=True, replicas=1),
            GroupMemberResult(
                slug="first", success=False, error="Resource could not be warmed"
            ),
        ],
    )
    ensure = AsyncMock(return_value=outcome)
    monkeypatch.setattr(service.management, "warm_up_group", ensure)
    result = asyncio.run(service.management.warm_up("group_example"))
    assert isinstance(result, GroupWarmUpResponse)
    assert result.model_dump() == outcome.model_dump()
    ensure.assert_awaited_once_with("group_example")
    assert service.resources.all() == []


def test_direct_group_replica_override_is_rejected_before_execution(monkeypatch):
    service = assemble_service()
    ensure = AsyncMock()
    monkeypatch.setattr(service.management, "warm_up_group", ensure)
    with pytest.raises(DeploymentServiceError) as caught:
        asyncio.run(service.management.warm_up("group_example", 2))
    assert caught.value.status_code == 422
    ensure.assert_not_awaited()


@pytest.mark.parametrize(
    "command,action", [("stop", "scale-to-zero"), ("cool_down", "cool-down")]
)
def test_direct_group_actions_use_shared_service_execution(
    monkeypatch, command, action
):
    service = assemble_service()
    outcome = GroupActionOutcome(
        slug="group_example",
        success=False,
        deployments=[
            GroupActionMemberOutcome(
                slug="busy", success=False, error="Resource in use"
            )
        ],
    )
    change = AsyncMock(return_value=outcome)
    monkeypatch.setattr(service.management, "change_group", change)
    result = asyncio.run(getattr(service.management, command)("group_example"))
    assert result == outcome
    change.assert_awaited_once_with("group_example", action)


def test_local_status_does_not_construct_configured_owners():
    service = assemble_service()
    assert service.resources.names()
    assert service.management.local_status() == []
    assert service.resources.all() == []
    assert service.backends == {}
