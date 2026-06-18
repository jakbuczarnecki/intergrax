# © Artur Czarnecki. All rights reserved.

"""Vendor-neutral execution boundary event schema (execution_boundary_event.v1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionBoundaryLineageV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str
    type: Literal["execution_record"] = "execution_record"


class ExecutionBoundaryRuntimeRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = "intergrax"
    runtime_version: str = ""


class HarnessBoundaryPolicyVerdictV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal["pre", "post"]
    action: str
    reason: str = ""
    policy_rule_id: str = ""


class HarnessBoundaryStepOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "denied", "failed", "paused"]
    next_action: str = ""
    error_code: str | None = None
    outcome_applied: bool = False


BoundaryActionStatus = Literal[
    "executed",
    "failed",
    "completed",
    "denied",
    "paused",
]


class ExecutionBoundaryEventV1(BaseModel):
    """Unsigned harness boundary fact for external receipt adapters."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["execution_boundary_event.v1"] = "execution_boundary_event.v1"
    event_id: str
    event_sequence: int = 0
    boundary_type: Literal["tool_execution", "harness_step"]
    signed: Literal[False] = False
    tool_id: str | None = None
    agent_id: str
    run_id: str
    step_id: str
    task_id: str = ""
    tenant_id: str = ""
    action_status: BoundaryActionStatus
    side_effects: bool | None = None
    risk_level: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None
    error_message: str | None = None
    policy_verdicts: list[HarnessBoundaryPolicyVerdictV1] = Field(default_factory=list)
    step_outcome: HarnessBoundaryStepOutcomeV1 | None = None
    occurred_at: str
    lineage: ExecutionBoundaryLineageV1
    runtime_ref: ExecutionBoundaryRuntimeRefV1 = Field(
        default_factory=ExecutionBoundaryRuntimeRefV1,
    )
