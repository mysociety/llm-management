"""
Management commands and result contracts shared by HTTP and direct callers.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class WarmUpAction(StrEnum):
    """
    Outcome of warming a local resource or remote deployment.
    """

    WARMED = "warmed"
    SCALED = "scaled"
    ALREADY_RUNNING = "already_running"
    RESUMED = "resumed"
    CREATED = "created"


class WarmUpResponse(BaseModel):
    slug: str
    action: WarmUpAction
    replicas: int | None


class GroupMemberResult(BaseModel):
    slug: str
    success: bool
    replicas: int | None = None
    error: str | None = None


class GroupWarmUpResponse(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupMemberResult]


class LifecycleCommand(StrEnum):
    """
    Management commands supported for a resource group.
    """

    COOL_DOWN = "cool-down"
    STOP = "scale-to-zero"


class LifecycleAction(StrEnum):
    """
    Outcome of stopping a resource or shortening its idle timer.
    """

    ALREADY_STOPPED = "already_stopped"
    COOLING_DOWN = "cooling_down"
    SCALED_TO_ZERO = "scaled_to_zero"


class GroupActionMemberOutcome(BaseModel):
    slug: str
    success: bool
    action: LifecycleAction | None = None
    seconds_until_scale_to_zero: float | None = None
    error: str | None = None


class GroupActionOutcome(BaseModel):
    slug: str
    success: bool
    deployments: list[GroupActionMemberOutcome]


class DeploymentStatusResponse(BaseModel):
    slug: str
    exists: bool
    replicas: int


class ScaleToZeroResponse(BaseModel):
    slug: str
    action: LifecycleAction


class CoolDownResponse(BaseModel):
    slug: str
    action: LifecycleAction
    seconds_until_scale_to_zero: float | None


class DeploymentOverview(BaseModel):
    """
    Summary of a single deployment's current state and idle timer.
    """

    slug: str
    deployment_name: str
    exists: bool
    replicas: int
    idle_seconds: float | None
    seconds_until_scale_to_zero: float | None


class AllDeploymentsResponse(BaseModel):
    """
    Overview of all configured deployments with their current state
    and time remaining before auto-scaling to zero.
    """

    idle_timeout_minutes: int
    deployments: list[DeploymentOverview]
