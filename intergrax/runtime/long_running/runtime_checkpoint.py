# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-layer checkpoint contract (§42.9.2, Phase G.1)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint.v1"
UAEP_STEP_CURSOR_KEY = "uaep_step_cursor"
PLAN_SNAPSHOT_KEY = "plan_snapshot.v1"


class GraphNodeCheckpoint(BaseModel):
    node_id: str
    status: str
    agent_id: Optional[str] = None


class RuntimeCheckpoint(BaseModel):
    schema_version: str = "runtime_checkpoint.v1"
    run_id: Optional[str] = None
    attempt_id: Optional[str] = None
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


def runtime_checkpoint_from_metadata(metadata: Dict[str, Any]) -> Optional[RuntimeCheckpoint]:
    raw = metadata.get(RUNTIME_CHECKPOINT_KEY)
    if isinstance(raw, dict):
        return RuntimeCheckpoint.model_validate(raw)
    return None


def attach_runtime_checkpoint_to_metadata(
    metadata: Dict[str, Any],
    checkpoint: RuntimeCheckpoint,
) -> None:
    metadata[RUNTIME_CHECKPOINT_KEY] = checkpoint.model_dump(mode="json")


def runtime_checkpoint_from_execution_structured(
    structured: Dict[str, Any],
) -> Optional[RuntimeCheckpoint]:
    raw = structured.get(RUNTIME_CHECKPOINT_KEY)
    if isinstance(raw, dict):
        return RuntimeCheckpoint.model_validate(raw)
    return None
