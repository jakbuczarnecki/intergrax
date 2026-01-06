# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
    
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


class PlanStopReason(str, Enum):
    COMPLETED = "completed"
    NEEDS_USER_INPUT = "needs_user_input"
    REPLAN_REQUIRED = "replan_required"
    FAILED = "failed"

@dataclass(frozen=True)
class UserInputRequest:
    """
    Human-in-the-loop request emitted by a plan step (e.g. ASK_CLARIFYING_QUESTION).
    """
    question: str
    must_answer_to_continue: bool = True

    # Optional metadata
    context_key: Optional[str] = None
    origin_step_id: Optional[StepId] = None

@dataclass(frozen=True)
class ReplanFailedStep:
    step_id: str
    action: str
    error_code: Optional[str]
    error_message: Optional[str]


@dataclass(frozen=True)
class ReplanContext:
    """
    Production-grade structured feedback for EnginePlanner replanning.

    This object is derived from PlanExecutionReport and MUST be safe to serialize.
    It should contain only stable, low-volume diagnostic signals (no large outputs).
    """
    attempt: int
    last_plan_id: str
    replan_reason: str
    executed_order: List[str]
    failed_steps: List[ReplanFailedStep]
    skipped_with_error_steps: List[ReplanFailedStep]

    @staticmethod
    def from_report(*, report: PlanExecutionReport, attempt: int, last_plan_id: str) -> ReplanContext:
        reason = (report.replan_reason or "").strip() or "replan_requested"

        executed_order = [sid.value if hasattr(sid, "value") else str(sid) for sid in (report.executed_order or [])]

        failed_steps: List[ReplanFailedStep] = []
        skipped_with_error: List[ReplanFailedStep] = []

        for r in (report.step_results or {}).values():
            if not isinstance(r.status, StepStatus):
                raise TypeError(
                    "StepReport.status must be StepStatus. "
                    f"Got {type(r.status).__name__}: {r.status!r}"
                )

            if r.status == StepStatus.FAILED:
                failed_steps.append(
                    ReplanFailedStep(
                        step_id=r.step_id,
                        action=r.step_id.value,
                        error_code=r.error.code if r.error else None,
                        error_message=r.error.message if r.error else None,
                    )
                )

            if r.status == StepStatus.SKIPPED and r.error is not None:
                skipped_with_error.append(
                    ReplanFailedStep(
                        step_id=r.step_id,
                        action=r.step_id.value,
                        error_code=r.error.code if r.error else None,
                        error_message=r.error.message if r.error else None,
                    ))

        return ReplanContext(
            attempt=attempt,
            last_plan_id=last_plan_id,
            replan_reason=reason,
            executed_order=executed_order,
            failed_steps=failed_steps,
            skipped_with_error_steps=skipped_with_error,
        )

    def to_prompt_dict(self) -> Dict[str, Any]:
        """
        Stable JSON-ish representation for prompts.
        """
        return {
            "attempt": self.attempt,
            "last_plan_id": self.last_plan_id,
            "replan_reason": self.replan_reason,
            "executed_order": self.executed_order,
            "failed_steps": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "error_code": s.error_code,
                    "error_message": s.error_message,
                }
                for s in self.failed_steps
            ],
            "skipped_with_error_steps": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "error_code": s.error_code,
                    "error_message": s.error_message,
                }
                for s in self.skipped_with_error_steps
            ],
        }


@dataclass(frozen=True)
class StepError:
    code: StepErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass(frozen=False)
class StepExecutionResult:
    step_id: StepId
    action: StepAction
    status: StepStatus

    # --- new: production telemetry ---
    sequence: int
    started_at_utc: str
    ended_at_utc: str
    duration_ms: int

    # --- new: structured data for audit/debug ---
    validated_params: Optional[Dict[str, Any]] = None
    output: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None

    error: Optional[StepError] = None
    attempts: int = 1


@dataclass(frozen=True)
class PlanExecutionReport:
    plan_id: str
    ok: bool

    step_results: Dict[StepId, StepExecutionResult]
    final_output: Optional[Any] = None
    replan_reason: Optional[str] = None


    stop_reason: PlanStopReason = PlanStopReason.COMPLETED
    user_input_request: Optional[UserInputRequest] = None

    started_at_utc: str = ""
    ended_at_utc: str = ""
    duration_ms: int = 0

    step_order: Optional[list[StepId]] = None
    executed_order: Optional[list[StepId]] = None



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
    @property
    def state(self) -> RuntimeState: ...

    @property
    def current_step(self) -> Optional[ExecutionStep]: ...

    @property
    def results(self) -> Mapping[StepId, StepExecutionResult]: ...

    @property
    def ordered_results(self) -> Sequence[StepExecutionResult]: ...

    def set_current_step(self, step: Optional[ExecutionStep]) -> None: ...

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
