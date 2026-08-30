# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-layer checkpoint contract (§42.9.2, Phase G.1)."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    validate_attempt_id,
    validate_run_id,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import ExecutionTreeSnapshot

UAEP_STEP_CURSOR_KEY = "uaep_step_cursor"
PLAN_SNAPSHOT_KEY = "plan_snapshot.v1"


class GraphNodeCheckpoint(BaseModel):
    node_id: str
    status: str
    agent_id: Optional[str] = None


class UaepStepCursor(BaseModel):
    values: Dict[str, bool] = Field(default_factory=dict)


class UaepStepOutput(BaseModel):
    step_id: str
    summary: str


class PendingDecision(BaseModel):
    type: str
    agent_id: str
    payload: Dict[str, object] = Field(default_factory=dict)


class RuntimeCheckpoint(BaseModel):
    """Canonical persisted checkpoint identity carrier (TRACE-1A, UE-9C)."""

    schema_version: str = "runtime_checkpoint.v2"
    run_id: RunId
    attempt_id: AttemptId
    execution_tree: ExecutionTreeSnapshot
    plan_id: Optional[str] = None
    graph_id: Optional[str] = None
    graph_node_id: Optional[str] = None
    agent_id: Optional[str] = None
    uaep_step_index: int = 0
    uaep_step_id: Optional[str] = None
    uaep_step_completed: bool = False
    uaep_step_cursor: Optional[UaepStepCursor] = None
    paused_phase: Optional[str] = None
    plan_snapshot: Optional[Dict[str, object]] = None
    graph_snapshot: Optional[Dict[str, object]] = None
    node_states: Dict[str, str] = Field(default_factory=dict)
    prior_node_outputs: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    pending_decisions: List[PendingDecision] = Field(default_factory=list)
    pending_human_request: Optional[Dict[str, object]] = None
    last_step_output: Optional[UaepStepOutput] = None

    @field_validator("run_id", mode="before")
    @classmethod
    def _validate_run_id(cls, value: object) -> RunId:
        return validate_run_id(value)

    @field_validator("attempt_id", mode="before")
    @classmethod
    def _validate_attempt_id(cls, value: object) -> AttemptId:
        return validate_attempt_id(value)

    def validate_canonical(self) -> None:
        self.execution_tree.validate_for_task(
            task_id=self.execution_tree.task_id,
            run_id=self.run_id,
        )
        if self.execution_tree.attempt_id != self.attempt_id:
            raise ValueError(
                "runtime checkpoint attempt_id mismatch with execution tree: "
                f"{self.attempt_id!r} != {self.execution_tree.attempt_id!r}"
            )


def _resolve_task_contract_forward_refs() -> None:
    from intergrax.runtime.task import task_contract

    task_contract.TaskOrchestrationState.model_rebuild(
        _types_namespace={"RuntimeCheckpoint": RuntimeCheckpoint},
    )


_resolve_task_contract_forward_refs()
