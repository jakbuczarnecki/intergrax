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


class ExecutionBoundaryEventV1(BaseModel):
    """Unsigned harness tool-boundary fact for external receipt adapters."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["execution_boundary_event.v1"] = "execution_boundary_event.v1"
    event_id: str
    boundary_type: Literal["tool_execution"] = "tool_execution"
    signed: Literal[False] = False
    tool_id: str
    agent_id: str
    run_id: str
    step_id: str
    task_id: str = ""
    tenant_id: str = ""
    action_status: Literal["executed", "failed"]
    side_effects: bool
    risk_level: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None
    error_message: str | None = None
    occurred_at: str
    lineage: ExecutionBoundaryLineageV1
    runtime_ref: ExecutionBoundaryRuntimeRefV1 = Field(
        default_factory=ExecutionBoundaryRuntimeRefV1,
    )
