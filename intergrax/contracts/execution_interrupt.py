# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Structured execution interrupts (architecture §42.8)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import AgentDecisionType


class InterruptType(str, Enum):
    POLICY_REVIEW_REQUIRED = "policy_review_required"
    SAFETY_VIOLATION = "safety_violation"
    COST_CEILING_BREACH = "cost_ceiling_breach"
    VALIDATION_CRITICAL_FAILURE = "validation_critical_failure"
    EXTERNAL_DEPENDENCY_FAILURE = "external_dependency_failure"
    HUMAN_JUDGMENT_REQUIRED = "human_judgment_required"
    PLAN_OBSOLESCENCE = "plan_obsolescence"
    AGENT_HANDOFF_REQUIRED = "agent_handoff_required"
    RUNTIME_RECOVERY_REQUIRED = "runtime_recovery_required"


class ExecutionInterrupt(BaseModel):
    interrupt_id: str = Field(default_factory=lambda: f"int_{uuid4().hex[:12]}")
    interrupt_type: InterruptType
    source_agent_id: str
    source_step_id: Optional[str] = None
    task_id: str
    run_id: str
    blocking: bool = True
    recommended_action: AgentDecisionType = AgentDecisionType.REQUEST_HUMAN
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = "execution_interrupt.v1"
