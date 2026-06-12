# © Artur Czarnecki. All rights reserved.

"""Environment diff contracts for Tier-3 deploy review (APP-EVOL-6 · §49.6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DiffRiskLevel(StrEnum):
    """Aggregate risk for a deploy diff review."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FieldChange(BaseModel):
    """Single field-level change between two structured documents."""

    model_config = ConfigDict(extra="forbid")

    path: str
    left: Any = None
    right: Any = None


class StructuredDiff(BaseModel):
    """Field-level diff between two JSON-like documents."""

    model_config = ConfigDict(extra="forbid")

    changes: list[FieldChange] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.changes)


class RosterChangeKind(StrEnum):
    """Roster mutation category."""

    ADDED = "added"
    REMOVED = "removed"
    CAPABILITIES_CHANGED = "capabilities_changed"


class RosterEntryChange(BaseModel):
    """Agent roster mutation between two manifests."""

    model_config = ConfigDict(extra="forbid")

    agent_key: str
    kind: RosterChangeKind
    left_capabilities: list[str] = Field(default_factory=list)
    right_capabilities: list[str] = Field(default_factory=list)


class ApplicationEnvironmentDiff(BaseModel):
    """Diff artifact for pre-deploy and incident review (§49.6.1)."""

    model_config = ConfigDict(extra="forbid")

    left_snapshot_id: str
    right_snapshot_id: str
    left_app_version: str
    right_app_version: str
    profile_diff: StructuredDiff
    graph_diff: StructuredDiff | None = None
    envelope_diff: StructuredDiff | None = None
    roster_diff: list[RosterEntryChange] = Field(default_factory=list)
    risk_level: DiffRiskLevel = DiffRiskLevel.LOW
    breaking_changes: list[str] = Field(default_factory=list)
