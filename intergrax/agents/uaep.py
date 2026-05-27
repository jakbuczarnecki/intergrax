# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Agent Execution Protocol (§42.5) — runtime-controlled agent loop."""

from __future__ import annotations

import time
from typing import Any, List, Optional
from uuid import uuid4

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.middleware.trace_middleware import TraceEmittingMiddleware
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest


class UAEPBlockedError(RuntimeError):
    """Raised when middleware/hooks block UAEP execution."""


def supports_uaep(agent: Agent) -> bool:
    """True when agent implements the optional UAEP step protocol (§42.32)."""
    return callable(getattr(agent, "get_steps", None)) and callable(
        getattr(agent, "run_step", None)
    )


class _BusEventEmitter:
    def __init__(self, bus: RuntimeEventBus) -> None:
        self._bus = bus

    async def emit(self, event: RuntimeEvent) -> None:
        await self._bus.publish(event)


class UAEPExecutor:
    """
    Executes agents through the Unified Agent Execution Protocol (§42.5).

    Legacy pipeline agents (no ``get_steps`` / ``run_step``) continue via
    :meth:`AgentEngine._execute_agent_impl` fallback.
    """

    def __init__(
        self,
        *,
        middleware: Optional[MiddlewarePipeline] = None,
        event_bus: Optional[RuntimeEventBus] = None,
    ) -> None:
        self._event_bus = event_bus
        if middleware is not None:
            self._middleware = middleware
        elif event_bus is not None:
            self._middleware = MiddlewarePipeline(
                middleware=[TraceEmittingMiddleware(event_bus)],
            )
        else:
            self._middleware = MiddlewarePipeline()

    @property
    def middleware(self) -> MiddlewarePipeline:
        return self._middleware

    async def execute(
        self,
        agent: Agent,
        request: RuntimeRequest,
    ) -> tuple[RuntimeAnswer, ValidationResult, RuntimeContext]:
        contract = agent.get_contract()
        run_id = str(request.metadata.get("run_id") or request.metadata.get("task_id") or uuid4().hex)
        task_id = str(request.metadata.get("task_id") or run_id)
        node_id = request.metadata.get("graph_node_id")

        exec_ctx = RuntimeExecutionContext(
            task_id=task_id,
            run_id=run_id,
            node_id=str(node_id) if node_id else None,
            agent_id=contract.id,
            correlation_id=task_id,
            phase=ExecutionPhase.CONTEXT_BUILDING,
            contract=contract,
            request=request,
            event_emitter=_BusEventEmitter(self._event_bus) if self._event_bus else None,
        )

        hook_base = HookContext(
            task_id=task_id,
            run_id=run_id,
            node_id=exec_ctx.node_id,
            agent_id=contract.id,
            phase=ExecutionPhase.CONTEXT_BUILDING,
        )
        await self._guard_hook(
            await self._middleware.run_before(HookPoint.BEFORE_CONTEXT_BUILD, hook_base)
        )

        runtime_context = agent.build_context(request)
        exec_ctx.domain_context = runtime_context

        await self._guard_hook(
            await self._middleware.run_after(
                HookPoint.AFTER_CONTEXT_BUILD,
                hook_base.model_copy(update={"phase": ExecutionPhase.CONTEXT_BUILDING}),
            )
        )
        await self._emit(
            exec_ctx,
            RuntimeEventType.CONTEXT_BUILT,
            ExecutionPhase.CONTEXT_BUILDING,
            {"agent_id": contract.id},
        )

        steps = self._resolve_steps(agent, runtime_context, contract.max_steps)
        last_output: Optional[StepOutput] = None

        for index, step in enumerate(steps):
            exec_ctx.phase = ExecutionPhase.STEP_EXECUTION
            hook_step = hook_base.model_copy(
                update={"step_id": step.step_id, "phase": ExecutionPhase.STEP_EXECUTION},
            )
            await self._guard_hook(
                await self._middleware.run_before(HookPoint.BEFORE_STEP, hook_step)
            )

            started = time.perf_counter()
            step_result = await self.execute_step(agent, step, exec_ctx)
            step_result.duration_ms = int((time.perf_counter() - started) * 1000)

            await self._guard_hook(
                await self._middleware.run_after(HookPoint.AFTER_STEP, hook_step)
            )

            if step_result.output is not None:
                last_output = step_result.output

            decision = step_result.decision or self._decide_after_step(
                agent, step, step_result.output, exec_ctx
            )
            await self._emit(
                exec_ctx,
                RuntimeEventType.DECISION_EMITTED,
                ExecutionPhase.STEP_EXECUTION,
                {"step_id": step.step_id, "decision": decision.type.value},
            )

            if decision.type != AgentDecisionType.CONTINUE:
                break

        answer = self._build_answer(exec_ctx, last_output, run_id)

        exec_ctx.phase = ExecutionPhase.VALIDATION
        hook_val = hook_base.model_copy(update={"phase": ExecutionPhase.VALIDATION})
        await self._guard_hook(
            await self._middleware.run_before(HookPoint.BEFORE_VALIDATION, hook_val)
        )
        validation = agent.validate(answer, context=runtime_context)
        await self._guard_hook(
            await self._middleware.run_after(HookPoint.AFTER_VALIDATION, hook_val)
        )
        await self._emit(
            exec_ctx,
            RuntimeEventType.VALIDATION_PASSED if validation.valid else RuntimeEventType.VALIDATION_FAILED,
            ExecutionPhase.VALIDATION,
            {"errors": validation.errors},
        )

        if not validation.valid and validation.errors and answer.route is not None:
            answer.route.extra.setdefault("agent_validation_errors", validation.errors)

        return answer, validation, runtime_context

    async def execute_step(
        self,
        agent: Agent,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepExecutionResult:
        output = await agent.run_step(step, ctx)
        return StepExecutionResult(output=output)

    @staticmethod
    def _resolve_steps(
        agent: Agent,
        runtime_context: RuntimeContext,
        max_steps: Optional[int],
    ) -> List[AgentStep]:
        steps = list(agent.get_steps(runtime_context))
        if not steps:
            raise ValueError(f"{type(agent).__name__}.get_steps() returned no steps.")
        limit = max_steps if max_steps is not None else len(steps)
        return steps[: max(1, limit)]

    @staticmethod
    def _decide_after_step(
        agent: Agent,
        step: AgentStep,
        output: Optional[StepOutput],
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        decide = getattr(agent, "decide_after_step", None)
        if callable(decide):
            return decide(step, output, ctx)
        return AgentDecision(type=AgentDecisionType.CONTINUE)

    @staticmethod
    def _build_answer(
        ctx: RuntimeExecutionContext,
        output: Optional[StepOutput],
        run_id: str,
    ) -> RuntimeAnswer:
        cached = ctx.metadata.get("runtime_answer")
        if isinstance(cached, RuntimeAnswer):
            return cached
        summary = output.summary if output else ""
        return RuntimeAnswer(run_id=run_id, answer=summary)

    async def _emit(
        self,
        ctx: RuntimeExecutionContext,
        event_type: RuntimeEventType,
        phase: ExecutionPhase,
        payload: dict[str, Any],
    ) -> None:
        if self._event_bus is None:
            return
        event = RuntimeEvent(
            task_id=ctx.task_id,
            run_id=ctx.run_id,
            node_id=ctx.node_id,
            agent_id=ctx.agent_id,
            event_type=event_type,
            phase=phase,
            payload=payload,
            correlation_id=ctx.correlation_id or ctx.task_id,
        )
        await self._event_bus.publish(event)

    @staticmethod
    async def _guard_hook(result: Any) -> None:
        if result.action != HookAction.ALLOW:
            raise UAEPBlockedError(result.reason or f"hook blocked: {result.action.value}")
