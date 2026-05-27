# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import AgentDecision, HumanRequest
from intergrax.contracts.execution_interrupt import ExecutionInterrupt


class AgentExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"


class AgentExecutionResult(BaseModel):
    """
    Structured agent output at the Agent→Nexus boundary (canonical architecture §14).
    """

    agent_id: str
    run_id: str
    status: AgentExecutionStatus
    summary: str = ""
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    used_tools: List[str] = Field(default_factory=list)
    cost: Optional[float] = None
    duration_seconds: Optional[float] = None
    next_recommendations: List[str] = Field(default_factory=list)
    agent_decision: Optional[AgentDecision] = None
    human_request: Optional[HumanRequest] = None
    execution_interrupt: Optional[ExecutionInterrupt] = None
    policy_rule_id: Optional[str] = None
