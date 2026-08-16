# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-layer checkpoint contract (§42.9.2, Phase G.1)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    validate_attempt_id,
    validate_run_id,
)

RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint.v1"
UAEP_STEP_CURSOR_KEY = "uaep_step_cursor"
PLAN_SNAPSHOT_KEY = "plan_snapshot.v1"


class GraphNodeCheckpoint(BaseModel):
    node_id: str
    status: str
    agent_id: Optional[str] = None


class _RuntimeCheckpointStateFields(BaseModel):
    schema_version: str = "runtime_checkpoint.v1"
    plan_id: Optional[str] = None
    graph_id: Optional[str] = None
    graph_node_id: Optional[str] = None
    agent_id: Optional[str] = None
    uaep_step_index: int = 0
    uaep_step_id: Optional[str] = None
    uaep_step_completed: bool = False
    uaep_step_cursor: Optional[Dict[str, Any]] = None
    paused_phase: Optional[str] = None
    plan_snapshot: Optional[Dict[str, Any]] = None
    graph_snapshot: Optional[Dict[str, Any]] = None
    node_states: Dict[str, str] = Field(default_factory=dict)
    prior_node_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    pending_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    pending_human_request: Optional[Dict[str, Any]] = None
    last_step_output: Optional[Dict[str, Any]] = None


class RuntimeCheckpointExecutionState(_RuntimeCheckpointStateFields):
    """Metadata/structured execution-state snapshot (legacy read boundary).

  ``run_id`` / ``attempt_id`` when present are non-authoritative carry-over fields.
  Canonical resume identity is ``TaskCheckpoint.runtime`` only — never metadata.
    """

    run_id: Optional[str] = None
    attempt_id: Optional[str] = None


class RuntimeCheckpoint(_RuntimeCheckpointStateFields):
    """Canonical persisted checkpoint identity carrier (TRACE-1A)."""

    run_id: RunId
    attempt_id: AttemptId

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)


RuntimeCheckpointStateView = Union[RuntimeCheckpoint, RuntimeCheckpointExecutionState]


def _execution_state_from_raw(raw: object) -> Optional[RuntimeCheckpointExecutionState]:
    if isinstance(raw, RuntimeCheckpointExecutionState):
        return raw
    if isinstance(raw, RuntimeCheckpoint):
        return RuntimeCheckpointExecutionState.model_validate(
            raw.model_dump(mode="json", exclude={"run_id", "attempt_id"})
        )
    if isinstance(raw, dict):
        return RuntimeCheckpointExecutionState.model_validate(raw)
    return None


def runtime_checkpoint_from_metadata(
    metadata: Dict[str, Any],
) -> Optional[RuntimeCheckpointExecutionState]:
    """Read execution-state snapshot from task metadata (identity NOT authoritative)."""
    return _execution_state_from_raw(metadata.get(RUNTIME_CHECKPOINT_KEY))


def attach_runtime_checkpoint_to_metadata(
    metadata: Dict[str, Any],
    checkpoint: RuntimeCheckpointStateView,
) -> None:
    metadata[RUNTIME_CHECKPOINT_KEY] = checkpoint.model_dump(mode="json")


def runtime_checkpoint_from_execution_structured(
    structured: Dict[str, Any],
) -> Optional[RuntimeCheckpointExecutionState]:
    """Read execution-state snapshot from agent structured output (identity NOT authoritative)."""
    return _execution_state_from_raw(structured.get(RUNTIME_CHECKPOINT_KEY))
