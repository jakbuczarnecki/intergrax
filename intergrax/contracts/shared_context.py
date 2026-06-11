# © Artur Czarnecki. All rights reserved.

"""Author-facing shared context view for graph handoffs (ACP-STATE-1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SharedContextConflictError(ValueError):
    """Optimistic concurrency failure on shared context version."""


class SharedContextConflictPolicy(StrEnum):
    """Conflict resolution posture (architecture §40.5.3 · ACP-PROD-5)."""

    LAST_WRITE_WINS = "last_write_wins"
    OPTIMISTIC_LOCK = "optimistic_lock"
    HITL_ON_CONFLICT = "hitl_on_conflict"


class PublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    key: str
    version: int
    conflict: bool = False


class SharedArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    size_bytes: int = 0


class SharedContextView(BaseModel):
    """
    Typed read/write facade for cross-agent handoffs (architecture §34).

    Persistence goes through ``shared_context_bridge`` — not raw task metadata.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    version: int = 1
    memory_namespace: str = "shared"
    conflict_policy: SharedContextConflictPolicy = SharedContextConflictPolicy.OPTIMISTIC_LOCK
    artifacts: dict[str, SharedArtifactRef] = Field(default_factory=dict)
    structured_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    entry_versions: dict[str, int] = Field(default_factory=dict)
    updated_by: dict[str, str] = Field(default_factory=dict)

    def get_structured_output(self, key: str) -> dict[str, Any] | None:
        return self.structured_outputs.get(key)

    def get(self, key: str) -> tuple[dict[str, Any] | None, int]:
        """Return value and per-key version (architecture §40.5.2)."""
        return self.structured_outputs.get(key), self.entry_versions.get(key, 0)

    def publish(
        self,
        key: str,
        value: dict[str, Any],
        *,
        expected_version: int | None = None,
        updated_by: str = "",
        strict_cas: bool = False,
    ) -> PublishResult:
        current_version = self.entry_versions.get(key, 0)
        if strict_cas and expected_version is None:
            raise SharedContextConflictError(
                "STRICT shared context publish requires expected_version"
            )
        if (
            self.conflict_policy == SharedContextConflictPolicy.OPTIMISTIC_LOCK
            and expected_version is not None
            and expected_version != current_version
        ):
            raise SharedContextConflictError(
                f"shared context key {key!r} version mismatch: "
                f"expected {expected_version}, got {current_version}"
            )
        if (
            self.conflict_policy == SharedContextConflictPolicy.HITL_ON_CONFLICT
            and expected_version is not None
            and expected_version != current_version
        ):
            raise SharedContextConflictError(
                f"HITL required for shared context conflict on key {key!r}"
            )
        self.structured_outputs[key] = value
        next_version = current_version + 1
        self.entry_versions[key] = next_version
        if updated_by:
            self.updated_by[key] = updated_by
        self.version += 1
        return PublishResult(accepted=True, key=key, version=next_version)

    def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_value: dict[str, Any],
        *,
        updated_by: str = "",
    ) -> bool:
        try:
            self.publish(
                key,
                new_value,
                expected_version=expected_version,
                updated_by=updated_by,
            )
            return True
        except SharedContextConflictError:
            return False

    def put_structured_output(
        self,
        key: str,
        payload: dict[str, Any],
        *,
        expected_version: int | None = None,
    ) -> None:
        if expected_version is not None and self.version != expected_version:
            raise SharedContextConflictError(
                f"shared context version mismatch: expected {expected_version}, got {self.version}"
            )
        if expected_version is None and self.conflict_policy in {
            SharedContextConflictPolicy.OPTIMISTIC_LOCK,
            SharedContextConflictPolicy.HITL_ON_CONFLICT,
        }:
            entry_expected = self.entry_versions.get(key, 0)
            self.publish(key, payload, expected_version=entry_expected if key in self.structured_outputs else 0)
            return
        self.publish(key, payload)

    def register_artifact(self, key: str, entry: SharedArtifactRef) -> None:
        self.artifacts[key] = entry
        self.version += 1
