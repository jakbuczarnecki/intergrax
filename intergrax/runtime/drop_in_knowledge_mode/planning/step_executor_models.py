# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol

from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import (
    ExecutionStep,
    StepAction,
    StepId,
)


class StepStatus(str, Enum):
    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"
    REPLAN_REQUESTED = "replan_requested"
    

class ReplanCode(str, Enum):
    VERIFICATION_FAILED = "verification_failed"
    TOOL_SCHEMA_MISMATCH = "tool_schema_mismatch"
    MISSING_CAPABILITY = "missing_capability"
    PLANNER_INVARIANT_VIOLATION = "planner_invariant_violation"
    STEP_POLICY_REPLAN = "step_policy_replan"
    UNKNOWN = "unknown"


class ReplanReason(str, Enum):
    RETRY_EXHAUSTED = "retry_exhausted"
    INVALID_OUTPUT = "invalid_output"
    DEPENDENCY_FAILED = "dependency_failed"
    USER_CONSTRAINT_VIOLATION = "user_constraint_violation"
    INTERNAL_ERROR = "internal_error"
    UNSPECIFIED = "unspecified"


class StepErrorCode(str, Enum):
    HANDLER_EXCEPTION = "handler_exception"
    INVALID_HANDLER_RESULT = "invalid_handler_result"
    DEPENDENCY_MISSING = "dependency_missing"
    DEPENDENCY_FAILED = "dependency_failed"
    RETRY_EXHAUSTED = "retry_exhausted"
    REPLAN = "replan"
    INTERNAL = "internal"
    UNKNOWN = "unknown"



@dataclass(frozen=True)
class StepError:
    code: StepErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class StepExecutionResult:
    step_id: StepId
    action: StepAction
    status: StepStatus
    output: Optional[Any] = None
    error: Optional[StepError] = None
    attempts: int = 1


@dataclass(frozen=True)
class PlanExecutionReport:
    plan_id: str
    ok: bool
    step_results: Dict[StepId, StepExecutionResult]
    final_output: Optional[Any] = None
    replan_reason: Optional[str] = None


class StepReplanRequested(RuntimeError):
    def __init__(
        self,
        *,
        code: ReplanCode,
        reason: ReplanReason,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"{code.value}: {reason.value}")
        self.code = code
        self.reason = reason
        self.details = details



class StepExecutionContext(Protocol):
    """
    Minimal contract executor needs from the host system (runtime/supervisor).
    Keep it tiny and testable.

    Notes:
    - state is intentionally opaque to the executor; it's owned by runtime.
    - results store can be used by handlers to read dependency outputs.
    """
    @property
    def state(self) -> Any: ...

    @property
    def results(self) -> Mapping[StepId, StepExecutionResult]: ...

    def set_result(self, result: StepExecutionResult) -> None: ...


StepHandler = Callable[[ExecutionStep, StepExecutionContext], Awaitable[StepExecutionResult]]


@dataclass(frozen=True)
class StepExecutorConfig:
    """
    Executor-level behavior. Keep deterministic.
    """
    # Global hard cap safety (even if step says retry many times)
    max_attempts_hard_cap: int = 3

    # Whether to stop execution immediately on FAILED step (even if more steps exist)
    fail_fast: bool = True


@dataclass(frozen=True)
class StepHandlerRegistry:
    """
    Explicit mapping StepAction -> handler.
    This replaces getattr/reflection and makes wiring obvious and testable.
    """
    handlers: Dict[StepAction, StepHandler]

    def get(self, action: StepAction) -> StepHandler:
        h = self.handlers.get(action)
        if h is None:
            raise KeyError(f"No handler registered for action={action.value}")
        return h
