# © Artur Czarnecki. All rights reserved.

"""Declarative Tier-3 environment migration contracts (APP-EVOL-2 · §49.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MigrationStepTarget(StrEnum):
    """Migration step primitive target (§49.2.1)."""

    PROFILE = "profile"
    GRAPH_SPEC = "graph_spec"
    ORG_ENVELOPE = "org_envelope"
    ROSTER = "roster"
    HOOKS = "hooks"


class MigrationStepAction(StrEnum):
    """How a migration step mutates or validates a target."""

    TRANSFORM = "transform"
    REPLACE = "replace"
    VALIDATE_ONLY = "validate_only"


class RemovedNodesPolicy(StrEnum):
    """Policy when graph migration removes nodes (§49.2.4)."""

    FAIL = "fail"
    ORPHAN_AUDIT = "orphan_audit"


class FieldTransform(BaseModel):
    """Profile field transform entry."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    action: Literal["set", "delete", "rename"]
    value: Any | None = None
    rename_to: str | None = None


class EdgeRewrite(BaseModel):
    """Graph edge rewrite entry."""

    model_config = ConfigDict(extra="forbid")

    from_source: str = Field(min_length=1)
    from_target: str = Field(min_length=1)
    to_source: str = Field(min_length=1)
    to_target: str = Field(min_length=1)


class ProfileMigration(BaseModel):
    """Typed profile sub-migration (§49.2.4)."""

    model_config = ConfigDict(extra="forbid")

    migration_id: str = Field(min_length=1)
    from_spec_version: str = Field(min_length=1)
    to_spec_version: str = Field(min_length=1)
    field_transforms: list[FieldTransform] = Field(default_factory=list)
    default_injection: dict[str, Any] = Field(default_factory=dict)
    breaking: bool = False
    golden_replay_ref: str | None = None


class GraphSpecMigration(BaseModel):
    """Typed graph-spec sub-migration (§49.2.4)."""

    model_config = ConfigDict(extra="forbid")

    migration_id: str = Field(min_length=1)
    from_graph_version: str = Field(min_length=1)
    to_graph_version: str = Field(min_length=1)
    node_renames: dict[str, str] = Field(default_factory=dict)
    edge_rewrites: list[EdgeRewrite] = Field(default_factory=list)
    removed_nodes_policy: RemovedNodesPolicy = RemovedNodesPolicy.FAIL
    breaking: bool = False
    golden_replay_ref: str | None = None


class OrgEnvelopeMigration(BaseModel):
    """Typed organizational envelope sub-migration (§49.2.4)."""

    model_config = ConfigDict(extra="forbid")

    migration_id: str = Field(min_length=1)
    from_envelope_version: str = Field(min_length=1)
    to_envelope_version: str = Field(min_length=1)
    playbook_id_map: dict[str, str] = Field(default_factory=dict)
    tool_deny_additions: list[str] = Field(default_factory=list)
    tool_deny_removals: list[str] = Field(default_factory=list)
    breaking: bool = False
    golden_replay_ref: str | None = None


class MigrationStep(BaseModel):
    """Single migration step against a deploy primitive."""

    model_config = ConfigDict(extra="forbid")

    target: MigrationStepTarget
    action: MigrationStepAction
    script_ref: str | None = None
    breaking: bool = False

    @model_validator(mode="after")
    def _breaking_requires_script(self) -> MigrationStep:
        if self.breaking and self.action is not MigrationStepAction.VALIDATE_ONLY:
            if not (self.script_ref and self.script_ref.strip()):
                raise ValueError("breaking migration steps require script_ref unless action=validate_only")
        return self


class ApplicationMigration(BaseModel):
    """Declarative application environment migration A → B (§49.2.1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "application_migration.v1"
    migration_id: str = Field(min_length=1)
    from_app_version: str = Field(min_length=1)
    to_app_version: str = Field(min_length=1)
    steps: list[MigrationStep] = Field(default_factory=list)
    profile_migration: ProfileMigration | None = None
    graph_spec_migration: GraphSpecMigration | None = None
    org_envelope_migration: OrgEnvelopeMigration | None = None
    rollback_supported: bool = False

    @model_validator(mode="after")
    def _typed_sub_migrations_present_in_steps(self) -> ApplicationMigration:
        if self.profile_migration is not None and not any(
            step.target is MigrationStepTarget.PROFILE for step in self.steps
        ):
            raise ValueError("profile_migration requires a profile MigrationStep")
        if self.graph_spec_migration is not None and not any(
            step.target is MigrationStepTarget.GRAPH_SPEC for step in self.steps
        ):
            raise ValueError("graph_spec_migration requires a graph_spec MigrationStep")
        if self.org_envelope_migration is not None and not any(
            step.target is MigrationStepTarget.ORG_ENVELOPE for step in self.steps
        ):
            raise ValueError("org_envelope_migration requires an org_envelope MigrationStep")
        return self
