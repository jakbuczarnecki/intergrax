# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.stepplan_models import ExecutionStep, StepAction
from intergrax.runtime.nexus.planning.step_executor_models import (
    StepError,
    StepErrorCode,
    StepExecutionContext,
    StepExecutionResult,
    StepHandler,
    StepHandlerRegistry,
    StepStatus,
)

# Your existing RuntimeStep protocol
from typing import Protocol

from intergrax.utils.time_provider import SystemTimeProvider


class RuntimeStep(Protocol):
    async def run(self, state: RuntimeState) -> None:
        ...


# -----------------------------
# Result helpers (minimal + deterministic)
# -----------------------------

def _now_iso() -> str:
    return SystemTimeProvider.utc_now().isoformat()


def _ok_result(*, step: ExecutionStep, attempts: int = 1) -> StepExecutionResult:
    # Executor overwrites telemetry anyway (sequence/timestamps/duration) when it wraps results,
    # but StepExecutionResult requires these fields, so provide deterministic placeholders.
    ts = _now_iso()
    return StepExecutionResult(
        step_id=step.step_id,
        action=step.action,
        status=StepStatus.OK,
        sequence=0,
        started_at_utc=ts,
        ended_at_utc=ts,
        duration_ms=0,
        validated_params=None,
        output=None,
        meta=None,
        error=None,
        attempts=attempts,
    )


def _failed_result(*, step: ExecutionStep, message: str, details: Optional[dict] = None, attempts: int = 1) -> StepExecutionResult:
    ts = _now_iso()
    return StepExecutionResult(
        step_id=step.step_id,
        action=step.action,
        status=StepStatus.FAILED,
        sequence=0,
        started_at_utc=ts,
        ended_at_utc=ts,
        duration_ms=0,
        validated_params=None,
        output=None,
        meta=None,
        error=StepError(
            code=StepErrorCode.HANDLER_EXCEPTION,
            message=message,
            details=details,
        ),
        attempts=attempts,
    )


# -----------------------------
# RuntimeStep -> StepHandler adapter
# -----------------------------

@dataclass(frozen=True)
class RuntimeStepBinding:
    """
    One explicit binding:
      StepAction -> factory that returns a RuntimeStep instance
    """
    action: StepAction
    factory: Callable[[], RuntimeStep]


def make_runtime_step_handler(*, action: StepAction, factory: Callable[[], RuntimeStep]) -> StepHandler:
    """
    Adapter: ExecutionStep + ctx -> executes RuntimeStep.run(state)
    """
    async def _handler(step: ExecutionStep, ctx: StepExecutionContext) -> StepExecutionResult:
        # Defensive: executor already validates action/step_id consistency, but keep it tight.
        if step.action != action:
            return _failed_result(
                step=step,
                message="Handler called with unexpected action.",
                details={"expected": action.value, "got": step.action.value},
            )

        # Instantiate step (fresh instance to keep handlers stateless by default)
        runtime_step = factory()

        # Run
        try:
            await runtime_step.run(ctx.state)
            return _ok_result(step=step, attempts=1)
        except Exception as e:
            return _failed_result(
                step=step,
                message=f"RuntimeStep exception: {type(e).__name__}: {e}",
                details={"step_id": step.step_id.value, "action": step.action.value},
                attempts=1,
            )

    return _handler


def build_runtime_step_registry(*, bindings: Dict[StepAction, Callable[[], RuntimeStep]]) -> StepHandlerRegistry:
    """
    Explicit, strongly-typed mapping StepAction -> handler.

    bindings: Dict[StepAction, factory()]
    """
    handlers: Dict[StepAction, StepHandler] = {}
    for action, factory in bindings.items():
        handlers[action] = make_runtime_step_handler(action=action, factory=factory)

    return StepHandlerRegistry(handlers=handlers)
