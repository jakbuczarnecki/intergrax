# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent step boundary contract (architecture §42.6)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import AgentDecision


class AgentStep(BaseModel):
    step_id: str
    step_name: str
    step_index: int = 0
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    allowed_tools: List[str] = Field(default_factory=list)
    max_duration_ms: int = 120_000
    max_retries: int = 0
    idempotent: bool = True
    trace_label: str = ""
    schema_version: str = "agent_step.v1"


class StepOutput(BaseModel):
    step_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    summary: str = ""


class StepExecutionResult(BaseModel):
    output: Optional[StepOutput] = None
    decision: Optional[AgentDecision] = None
    duration_ms: Optional[int] = None
