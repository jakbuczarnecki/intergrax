# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hook context and results (architecture §42.3)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from intergrax.runtime.events.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent


class HookAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"


class HookContext(BaseModel):
    task_id: str
    run_id: str
    node_id: Optional[str] = None
    agent_id: Optional[str] = None
    step_id: Optional[str] = None
    phase: ExecutionPhase = ExecutionPhase.STEP_EXECUTION
    runtime_state: Dict[str, Any] = Field(default_factory=dict)
    event: Optional[RuntimeEvent] = None

    model_config = {"arbitrary_types_allowed": True}


class HookResult(BaseModel):
    action: HookAction = HookAction.ALLOW
    modified_payload: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
