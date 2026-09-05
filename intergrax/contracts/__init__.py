# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from intergrax.contracts.agent_contract_meta import AgentContract, AgentExecutionMode, AgentRiskLevel
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_interrupt import ExecutionInterrupt, InterruptType
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision, EnforcementLevel
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.contracts.validation import ValidationResult
from intergrax.contracts.validation_contract import ExtendedValidationResult, ValidationContract

if TYPE_CHECKING:
    from intergrax.contracts.runtime_mapping import runtime_answer_to_agent_result

__all__ = [
    "AgentContract",
    "AgentDecision",
    "AgentDecisionType",
    "AgentExecutionMode",
    "AgentExecutionResult",
    "AgentExecutionStatus",
    "AgentRiskLevel",
    "AgentStep",
    "CapabilityMatchResult",
    "EnforcementLevel",
    "EventSeverity",
    "ExecutionInterrupt",
    "ExtendedValidationResult",
    "InterruptType",
    "PolicyAction",
    "PolicyDecision",
    "RuntimeExecutionContext",
    "StepExecutionResult",
    "StepOutput",
    "ToolRequest",
    "ToolResponse",
    "ToolResponseStatus",
    "ValidationContract",
    "ValidationResult",
    "runtime_answer_to_agent_result",
]


def __getattr__(name: str) -> Any:
    if name == "runtime_answer_to_agent_result":
        from intergrax.contracts.runtime_mapping import runtime_answer_to_agent_result

        return runtime_answer_to_agent_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
