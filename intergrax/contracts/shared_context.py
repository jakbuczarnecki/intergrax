# © Artur Czarnecki. All rights reserved.

"""Author-facing shared context view for graph handoffs (ACP-STATE-1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SharedContextConflictError(ValueError):
    """Optimistic concurrency failure on shared context version."""


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
    artifacts: dict[str, SharedArtifactRef] = Field(default_factory=dict)
    structured_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def get_structured_output(self, key: str) -> dict[str, Any] | None:
        return self.structured_outputs.get(key)

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
        self.structured_outputs[key] = payload
        self.version += 1

    def register_artifact(self, key: str, entry: SharedArtifactRef) -> None:
        self.artifacts[key] = entry
        self.version += 1
