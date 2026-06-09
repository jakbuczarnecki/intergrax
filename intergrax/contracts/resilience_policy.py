# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composable resilience policy contract (architecture REL §34)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FailureResponse(str, Enum):
    RETRY = "retry"
    RETRY_ALTERNATE = "retry_alternate"
    CIRCUIT_BREAK = "circuit_break"
    FAIL = "fail"
    DEGRADE = "degrade"
    REQUEST_HUMAN = "request_human"
    PARTIAL = "partial"
    RETRY_RUN = "retry_run"
    RETRY_GRAPH = "retry_graph"
    RECOVERY_REBOOT = "recovery_reboot"
    ESCALATE = "escalate"


class RebootStrategy(str, Enum):
    NONE = "none"
    RE_EXECUTE_NODE = "re_execute_node"
    RE_EXECUTE_GRAPH = "re_execute_graph"
    COLD_AGENT_RELOAD = "cold_agent_reload"


class FailureClass(str, Enum):
    USER_ERROR = "user_error"
    POLICY_ERROR = "policy_error"
    DEPENDENCY_ERROR = "dependency_error"
    RUNTIME_ERROR = "runtime_error"
    QUALITY_ERROR = "quality_error"


class ResiliencePolicy(BaseModel):
    """Modular fault-tolerance policy resolved at host assembly time."""

    policy_id: str = "harness.default"
    version: str = "1"
    on_dependency_error: FailureResponse = FailureResponse.RETRY
    on_quality_error: FailureResponse = FailureResponse.RETRY_ALTERNATE
    on_timeout: FailureResponse = FailureResponse.RETRY_RUN
    on_runtime_error: FailureResponse = FailureResponse.RECOVERY_REBOOT
    max_attempts: int = Field(default=3, ge=0, le=32)
    backoff: Literal["fixed", "exponential", "none"] = "exponential"
    alternate_agent_ids: list[str] = Field(default_factory=list)
    allow_partial_result: bool = False
    checkpoint_on_pause: bool = True
    reboot_strategy: RebootStrategy = RebootStrategy.RE_EXECUTE_NODE

    def action_for(self, failure_class: FailureClass) -> FailureResponse:
        if failure_class is FailureClass.USER_ERROR:
            return FailureResponse.FAIL
        if failure_class is FailureClass.POLICY_ERROR:
            return FailureResponse.REQUEST_HUMAN
        if failure_class is FailureClass.DEPENDENCY_ERROR:
            return self.on_dependency_error
        if failure_class is FailureClass.QUALITY_ERROR:
            return self.on_quality_error
        if failure_class is FailureClass.RUNTIME_ERROR:
            return self.on_runtime_error
        return FailureResponse.FAIL


def default_resilience_policy() -> ResiliencePolicy:
    return ResiliencePolicy()
