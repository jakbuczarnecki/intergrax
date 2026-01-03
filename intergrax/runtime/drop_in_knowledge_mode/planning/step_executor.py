# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import (
    ExecutionPlan,
    ExecutionStep,
    FailurePolicyKind,
    StepAction,
    StepId,
)
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import (
    PlanExecutionReport,
    ReplanCode,
    ReplanReason,
    StepError,
    StepErrorCode,
    StepExecutionContext,
    StepExecutionResult,
    StepExecutorConfig,
    StepHandlerRegistry,
    StepReplanRequested,
    StepStatus,
)


@dataclass
class _DefaultExecContext(StepExecutionContext):
    _state: RuntimeState
    _results: Dict[StepId, StepExecutionResult]

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def results(self) -> Mapping[StepId, StepExecutionResult]:
        return self._results

    def set_result(self, result: StepExecutionResult) -> None:
        self._results[result.step_id] = result


class StepExecutor:
    """
    Executes an ExecutionPlan deterministically.

    The executor:
      - does NOT know runtime internals,
      - uses injected handlers (StepAction -> handler),
      - maintains a StepId -> StepExecutionResult store for dependencies.
    """

    def __init__(
        self,
        *,
        registry: StepHandlerRegistry,
        cfg: Optional[StepExecutorConfig] = None,
    ) -> None:
        self._registry = registry
        self._cfg = cfg or StepExecutorConfig()

    async def execute(self, *, plan: ExecutionPlan, state: RuntimeState) -> PlanExecutionReport:
        results: Dict[StepId, StepExecutionResult] = {}
        ctx = _DefaultExecContext(_state=state, _results=results)

        replan_reason: Optional[str] = None

        for step in plan.steps:
            if not step.enabled:
                res = StepExecutionResult(
                    step_id=step.step_id,
                    action=step.action,
                    status=StepStatus.SKIPPED,
                    output=None,
                    error=None,
                    attempts=1,
                )
                ctx.set_result(res)
                continue

            # Dependency gate: if any dep FAILED/REPLAN -> propagate stop
            if not self._deps_ok(step=step, results=results):
                res = StepExecutionResult(
                    step_id=step.step_id,
                    action=step.action,
                    status=StepStatus.SKIPPED,
                    output=None,
                    error=StepError(
                        code=StepErrorCode.DEPENDENCY_FAILED,
                        message="Skipped due to failed dependency.",
                        details={"depends_on": [d.value for d in step.depends_on]},
                    ),
                    attempts=1,
                )
                ctx.set_result(res)
                if self._cfg.fail_fast:
                    break
                continue

            try:
                res = await self._run_step_with_policy(step=step, ctx=ctx)
                ctx.set_result(res)

                if res.status in (StepStatus.FAILED, StepStatus.REPLAN_REQUESTED) and self._cfg.fail_fast:
                    if res.status == StepStatus.REPLAN_REQUESTED:
                        replan_reason = (res.error.message if res.error else None) or "replan_requested"
                    break

            except StepReplanRequested as e:
                res = StepExecutionResult(
                    step_id=step.step_id,
                    action=step.action,
                    status=StepStatus.REPLAN_REQUESTED,
                    output=None,
                    error=StepError(
                        code=StepErrorCode.REPLAN,
                        message=f"{e.code.value}: {e.reason.value}",
                        details={
                            "replan_code": e.code.value,
                            "replan_reason": e.reason.value,
                            "replan_details": e.details,
                        },
                    ),
                    attempts=1,
                )
                ctx.set_result(res)
                continue


        # Final output: last OK output if any (runtime can override this later)
        final_output: Optional[Any] = None
        final_step = next(
            (s for s in reversed(plan.steps) if s.action == StepAction.FINALIZE_ANSWER),
            None,
        )
        if final_step is not None:
            r = results.get(final_step.step_id)
            if r is not None and r.status == StepStatus.OK:
                final_output = r.output

        has_failed = any(r.status == StepStatus.FAILED for r in results.values())
        has_replan = any(r.status == StepStatus.REPLAN_REQUESTED for r in results.values())
        has_skipped_error = any(
            (r.status == StepStatus.SKIPPED and r.error is not None) for r in results.values()
        )

        ok = (replan_reason is None) and (not has_failed) and (not has_replan) and (not has_skipped_error)

        # If plan includes FINALIZE_ANSWER, require it to be OK for ok=True
        final_step = next(
            (s for s in reversed(plan.steps) if s.action == StepAction.FINALIZE_ANSWER),
            None,
        )
        if ok and final_step is not None:
            fr = results.get(final_step.step_id)
            if fr is None or fr.status != StepStatus.OK:
                ok = False


        return PlanExecutionReport(
            plan_id=plan.plan_id,
            ok=ok,
            step_results=results,
            final_output=final_output,
            replan_reason=replan_reason,
        )

    def _deps_ok(self, *, step: ExecutionStep, results: Mapping[StepId, StepExecutionResult]) -> bool:
        for dep in step.depends_on:
            r = results.get(dep)
            if r is None:
                raise RuntimeError(
                    f"Missing dependency result: step={step.step_id.value} depends_on={dep.value}"
                )
            if r.status in (StepStatus.FAILED, StepStatus.REPLAN_REQUESTED):
                return False
        return True


    async def _run_step_with_policy(self, *, step: ExecutionStep, ctx: StepExecutionContext) -> StepExecutionResult:
        policy = step.on_failure.policy
        max_retries = int(step.on_failure.max_retries)
        backoff_ms = int(step.on_failure.retry_backoff_ms)

        # attempts = 1 + retries, but hard capped
        allowed_attempts = min(1 + max_retries, int(self._cfg.max_attempts_hard_cap))

        last_err: Optional[StepError] = None

        for attempt in range(1, allowed_attempts + 1):
            try:
                handler = self._registry.get(step.action)
                res = await handler(step, ctx)

                if res is None:
                    raise RuntimeError("StepHandler returned None (expected StepExecutionResult).")

                if res.step_id != step.step_id:
                    raise RuntimeError(
                        f"StepHandler returned mismatched step_id: got={res.step_id.value} expected={step.step_id.value}"
                    )

                if res.action != step.action:
                    raise RuntimeError(
                        f"StepHandler returned mismatched action: got={res.action.value} expected={step.action.value}"
                    )

                # normalize attempts
                if res.attempts != attempt:
                    res = StepExecutionResult(
                        step_id=res.step_id,
                        action=res.action,
                        status=res.status,
                        output=res.output,
                        error=res.error,
                        attempts=attempt,
                    )
                return res

            except StepReplanRequested:
                # bubble up; executor will record it
                raise

            except Exception as e:
                last_err = StepError(
                    code=StepErrorCode.HANDLER_EXCEPTION,
                    message=str(e),
                    details={"action": step.action.value, "step_id": step.step_id.value},
                )

                if attempt < allowed_attempts and policy == FailurePolicyKind.RETRY:
                    if backoff_ms > 0:
                        await asyncio.sleep(backoff_ms / 1000.0)
                    continue

                # terminal based on policy
                if policy == FailurePolicyKind.SKIP:
                    return StepExecutionResult(
                        step_id=step.step_id,
                        action=step.action,
                        status=StepStatus.SKIPPED,
                        output=None,
                        error=last_err,
                        attempts=attempt,
                    )

                if policy == FailurePolicyKind.REPLAN:
                    raise StepReplanRequested(
                        code=ReplanCode.STEP_POLICY_REPLAN,
                        reason=ReplanReason.INTERNAL_ERROR,
                        details={"step_id": step.step_id.value, "action": step.action.value}
                    )

                # default FAIL
                return StepExecutionResult(
                    step_id=step.step_id,
                    action=step.action,
                    status=StepStatus.FAILED,
                    output=None,
                    error=last_err,
                    attempts=attempt,
                )

        # should never reach, but keep safe
        return StepExecutionResult(
            step_id=step.step_id,
            action=step.action,
            status=StepStatus.FAILED,
            output=None,
            error=last_err or StepError(code=StepErrorCode.UNKNOWN, message="Unknown failure"),
            attempts=allowed_attempts,
        )
