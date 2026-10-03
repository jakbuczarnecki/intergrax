# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hook context and results (architecture §42.3).

``HookContext`` satisfies :class:`~intergrax.contracts.middleware_hook_semantics.MiddlewareHookInvocationContext`
for cross-layer middleware. ``runtime_state`` remains a runtime-internal compatibility carrier for
hook-registry mutation and is not part of the Tier-0 middleware ABI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.middleware_hook_point import HookPoint
from intergrax.contracts.middleware_hook_semantics import (
    EmptyMiddlewareHookPayload,
    MiddlewareExecutionSubjectFacet,
    MiddlewareHookPayload,
)
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
    hook_point: HookPoint = HookPoint.BEFORE_TASK_INTAKE
    payload: MiddlewareHookPayload = Field(default_factory=EmptyMiddlewareHookPayload)
    subject: MiddlewareExecutionSubjectFacet = Field(
        default_factory=MiddlewareExecutionSubjectFacet,
    )
    runtime_state: Dict[str, Any] = Field(default_factory=dict)
    event: Optional[RuntimeEvent] = None

    model_config = {"arbitrary_types_allowed": True}


class HookResult(BaseModel):
    action: HookAction = HookAction.ALLOW
    modified_payload: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
