"""
Typed JSON records used to recover deployments and inspect template runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from .config import TemplateRecipe


class JsonRecord(BaseModel):
    """
    Preserve unknown fields when reading manifests written by other versions.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Validate an existing JSON record before using it for recovery.
        """
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        """
        Replace the JSON record atomically within its existing directory.
        """
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(self.model_dump_json(indent=2, exclude_unset=True) + "\n")
        temporary.replace(path)


class RecoveryState(JsonRecord):
    """
    Cloud resource identity and progress shared by Compute and template runs.
    """

    name: str
    zone: str
    create_operation: str | None = None
    instance_id: str | None = None
    cleanup_complete: bool = False


class ComputeState(RecoveryState):
    """
    Durable identity needed to reconnect to a single Compute deployment.
    """

    template_id: str | None = None
    default_user: str = "ubuntu"

    def reset_creation(self) -> None:
        """
        Clear progress from a previous VM while preserving its JSON shape.
        """
        self.cleanup_complete = False
        self.create_operation = None
        self.instance_id = None
        self.model_fields_set.difference_update(
            {"cleanup_complete", "create_operation", "instance_id"}
        )


class TemplateState(RecoveryState):
    """
    Recovery manifest and measurements for one template build or smoke test.
    """

    image: str | None = None
    source_template_id: str | None = None
    recipe: TemplateRecipe | None = None
    snapshot_operation: str | None = None
    snapshot_id: str | None = None
    template_name: str | None = None
    template_operation: str | None = None
    template_id: str | None = None
    provision_seconds: float | None = None
    ssh_seconds: float | None = None
    health_seconds: float | None = None
    smoke_complete_seconds: float | None = None
    inference_passed: bool = False
    vm_deleted: bool = False
    error_type: str | None = None


class TemplateReport(JsonRecord):
    """
    Recipe, builder outcome and benchmark runs retained for inspection.
    """

    recipe: TemplateRecipe
    builder: TemplateState | None = None
    template_id: str | None = None
    benchmarks: list[TemplateState] = Field(default_factory=list[TemplateState])
