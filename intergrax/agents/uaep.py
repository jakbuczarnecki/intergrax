# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Agent Execution Protocol (§42.5) — runtime-controlled agent loop."""

from __future__ import annotations

import time
from typing import Any, List, Optional
from uuid import uuid4

from intergrax.agents.agent_contract import Agent
from intergrax.agents.uaep_protocol import (
    UAEPAgent,
    UAEPAgentWithDecide,
    UAEPAgentWithResume,
    is_uaep_agent,
    supports_uaep,
)
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.governance_hooks import hook_context_for_task, run_hook_pair
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler, GovernanceResolution
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.middleware.trace_middleware import TraceEmittingMiddleware
from intergrax.runtime.nexus.tools.uaep_tool_gateway import BoundToolGateway
from intergrax.runtime.policy.policy_engine import PolicyEngine, coerce_policy_engine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.nexus.engine.runtime import RuntimeEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RouteInfo, RuntimeRequest
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task_metadata_bridge import execution_options_for_request
from intergrax.runtime.cancellation.coordinator import (
    CANCELLATION_REQUESTED_KEY,
    CancellationCoordinator,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    should_resume_uaep_step,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.runtime_checkpoint import (
    RUNTIME_CHECKPOINT_KEY,
    UAEP_STEP_CURSOR_KEY,
    PLAN_SNAPSHOT_KEY,
    RuntimeCheckpoint,
    attach_runtime_checkpoint_to_metadata,
    runtime_checkpoint_from_metadata,
)
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import SHADOW_WORKSPACE_ID_KEY
from intergrax.runtime.task_memory.limits import TaskMemoryLimits
from intergrax.runtime.task_memory.memory_view import PolicyScopedMemoryView
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.task_memory.policy import (
    MemoryAccessPolicy,
    memory_access_policy_from_metadata,
)
from intergrax.runtime.task_memory.delegation_memory import apply_delegation_memory_namespace
from intergrax.runtime.nexus.context.shared_context_bridge import hydrate_shared_context_memory
from intergrax.runtime.nexus.context.shared_task_context import (
    DEFAULT_SHARED_MEMORY_NAMESPACE,
    load_shared_task_context_from_metadata,
)


class UAEPBlockedError(RuntimeError):
    """Raised when middleware/hooks block UAEP execution."""


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
        policy_engine: PolicyEngine | RuntimePolicyEngine | None = None,
        interrupt_handler: Optional[ExecutionInterruptHandler] = None,
        shadow_manager: Optional[ShadowWorkspaceManager] = None,
        sandbox_manager: Optional[SandboxSessionManager] = None,
        task_memory_store: Optional[TaskMemoryPersistence] = None,
        memory_limits: Optional[TaskMemoryLimits] = None,
        critic_hooks: Any = None,
        verify_uaep_step: bool = False,
    ) -> None:
        resolved_policy = coerce_policy_engine(policy_engine)
        self._event_bus = event_bus
        self._interrupt_handler = interrupt_handler or ExecutionInterruptHandler(
            policy_engine=resolved_policy,
        )
        self._shadow_manager = shadow_manager or ShadowWorkspaceManager()
        self._sandbox_manager = sandbox_manager or SandboxSessionManager()
        self._task_memory_store = task_memory_store
        self._memory_limits = memory_limits or TaskMemoryLimits()
        self._critic_hooks = critic_hooks
        self._verify_uaep_step = verify_uaep_step
        if middleware is not None:
            self._middleware = middleware
        elif event_bus is not None:
            self._middleware = MiddlewarePipeline(
                middleware=[TraceEmittingMiddleware(event_bus)],
            )
        else:
            self._middleware = MiddlewarePipeline()

    def set_critic_hooks(self, hooks: Any, *, verify_uaep_step: bool = False) -> None:
        self._critic_hooks = hooks
        self._verify_uaep_step = verify_uaep_step

    @staticmethod
    def _retention_days_from_metadata(metadata: dict[str, Any]) -> int | None:
        raw = metadata.get("memory_retention_days")
        if raw is None:
            return None
        try:
            days = int(raw)
        except (TypeError, ValueError):
            return None
        return days if days >= 1 else None

    @property
    def middleware(self) -> MiddlewarePipeline:
        return self._middleware

    @property
    def interrupt_handler(self) -> ExecutionInterruptHandler:
        return self._interrupt_handler

    async def execute(
        self,
        agent: Agent,
        request: RuntimeRequest,
    ) -> tuple[RuntimeAnswer, ValidationResult, RuntimeContext, Optional[GovernanceResolution]]:
        contract = agent.get_contract()
        task_options = execution_options_for_request(request)
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
        self._attach_shadow_workspace(exec_ctx, request, task_id=task_id)
        self._attach_sandbox_session(exec_ctx, request, task_id=task_id)
        self._attach_shared_context(exec_ctx, request)
        self._attach_memory_view(exec_ctx, request, task_id=task_id)

        hook_base = HookContext(
            task_id=task_id,
            run_id=run_id,
            node_id=exec_ctx.node_id,
            agent_id=contract.id,
            phase=ExecutionPhase.CONTEXT_BUILDING,
            runtime_state={
                "prompt": request.message or "",
                "tenant_id": request.tenant_id or "",
            },
        )
        await self._guard_hook(
            await self._middleware.run_before(HookPoint.BEFORE_CONTEXT_BUILD, hook_base)
        )

        runtime_context = agent.build_context(request)
        exec_ctx.domain_context = runtime_context
        exec_ctx.tool_gateway = BoundToolGateway(
            exec_ctx,
            allowed_tools=list(contract.allowed_tools),
            middleware=self._middleware,
        )

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
        governance: Optional[GovernanceResolution] = None
        runtime_ckpt = runtime_checkpoint_from_metadata(request.metadata)
        human_approved = task_options.human.is_resumed or bool(request.metadata.get("human_approved"))

        for index, step in enumerate(steps):
            if CancellationCoordinator.is_requested(request.metadata):
                validation = ValidationResult(valid=False, errors=["task_cancelled"])
                answer = self._build_answer(exec_ctx, last_output, run_id)
                if answer.route is None:
                    answer.route = RouteInfo(extra={})
                answer.route.extra[CANCELLATION_REQUESTED_KEY] = True
                return answer, validation, runtime_context, None

            exec_ctx.phase = ExecutionPhase.STEP_EXECUTION
            hook_step = hook_base.model_copy(
                update={"step_id": step.step_id, "phase": ExecutionPhase.STEP_EXECUTION},
            )
            await self._guard_hook(
                await self._middleware.run_before(HookPoint.BEFORE_STEP, hook_step)
            )

            started = time.perf_counter()
            if should_skip_uaep_step(
                step_index=index,
                step_id=step.step_id,
                checkpoint=runtime_ckpt,
                human_approved=human_approved,
            ):
                last_output = StepOutput.model_validate(runtime_ckpt.last_step_output)
                step_result = StepExecutionResult(output=last_output)
            elif should_resume_uaep_step(
                step_index=index,
                step_id=step.step_id,
                checkpoint=runtime_ckpt,
                human_approved=human_approved,
            ):
                assert runtime_ckpt is not None
                exec_ctx.metadata[UAEP_STEP_CURSOR_KEY] = dict(runtime_ckpt.uaep_step_cursor or {})
                step_result = await self._execute_step_with_resume(
                    agent,
                    step,
                    exec_ctx,
                    runtime_ckpt.uaep_step_cursor or {},
                )
            else:
                step_result = await self.execute_step(agent, step, exec_ctx)
            step_result.duration_ms = int((time.perf_counter() - started) * 1000)

            await self._guard_hook(
                await self._middleware.run_after(HookPoint.AFTER_STEP, hook_step)
            )
            if step_result.output is not None:
                await self._scan_step_output_hooks(
                    hook_step,
                    step_result.output,
                    request,
                    exec_ctx=exec_ctx,
                )

            if step_result.output is not None:
                last_output = step_result.output

            critic_resolution = self._verify_uaep_step_critic(
                contract=contract,
                step=step,
                step_result=step_result,
                request=request,
                run_id=run_id,
                task_id=task_id,
                task_options=task_options,
                exec_ctx=exec_ctx,
            )
            if critic_resolution is not None:
                governance = critic_resolution
                exec_ctx.metadata["governance_resolution"] = critic_resolution
                step_cursor = exec_ctx.metadata.get(UAEP_STEP_CURSOR_KEY)
                runtime_snapshot = self._build_runtime_checkpoint(
                    request=request,
                    contract_id=contract.id,
                    step_index=index,
                    step=step,
                    last_output=last_output,
                    resolution=critic_resolution,
                    step_cursor=step_cursor if isinstance(step_cursor, dict) else None,
                )
                exec_ctx.metadata[RUNTIME_CHECKPOINT_KEY] = runtime_snapshot
                break

            decision = step_result.decision or self._decide_after_step(
                agent, step, step_result.output, exec_ctx
            )
            decision_ctx = hook_context_for_task(
                task_id=task_id,
                run_id=run_id,
                agent_id=contract.id,
                step_id=step.step_id,
                phase=ExecutionPhase.STEP_EXECUTION,
                runtime_state={"decision_type": decision.type.value},
            )
            await run_hook_pair(
                self._middleware,
                HookPoint.BEFORE_DECISION,
                HookPoint.AFTER_DECISION,
                decision_ctx,
            )
            decision_context: dict[str, object] = {
                "require_human_on_critical": task_options.governance.require_human_on_critical,
                "has_unresolved_critical_interrupt": exec_ctx.metadata.get(
                    "has_unresolved_critical_interrupt", False
                ),
            }
            replan_policy = request.metadata.get("replan_policy.v1")
            if isinstance(replan_policy, dict):
                decision_context.update(replan_policy)
            resolution = self._interrupt_handler.resolve_decision(
                decision,
                task_id=task_id,
                run_id=run_id,
                agent_id=contract.id,
                step_id=step.step_id,
                context=decision_context,
            )
            if resolution.interrupt is not None:
                interrupt_ctx = hook_context_for_task(
                    task_id=task_id,
                    run_id=run_id,
                    agent_id=contract.id,
                    step_id=step.step_id,
                    phase=ExecutionPhase.INTERRUPT_HANDLING,
                    runtime_state={"interrupt_type": resolution.interrupt.type.value},
                )
                await run_hook_pair(
                    self._middleware,
                    HookPoint.BEFORE_INTERRUPT,
                    HookPoint.AFTER_INTERRUPT,
                    interrupt_ctx,
                )
            governance = resolution
            exec_ctx.metadata["governance_resolution"] = resolution

            from intergrax.contracts.decision_record import DecisionRecord

            tenant_id = str(request.tenant_id or request.metadata.get("tenant_id") or "default")
            decision_record = DecisionRecord(
                trace_id=run_id,
                run_id=run_id,
                tenant_id=tenant_id,
                task_id=task_id,
                agent_id=contract.id,
                step_id=step.step_id,
                decision_type=decision.type.value,
                rationale=decision.reason,
                policy_action=resolution.policy_decision.action.value,
            )
            await self._emit(
                exec_ctx,
                RuntimeEventType.DECISION_EMITTED,
                ExecutionPhase.STEP_EXECUTION,
                {
                    "step_id": step.step_id,
                    "decision": decision.type.value,
                    "policy_action": resolution.policy_decision.action.value,
                    "decision_record": decision_record.model_dump(mode="json"),
                },
            )
            await self._emit_governance(exec_ctx, resolution)

            if resolution.should_pause or resolution.should_fail:
                step_cursor = exec_ctx.metadata.get(UAEP_STEP_CURSOR_KEY)
                runtime_snapshot = self._build_runtime_checkpoint(
                    request=request,
                    contract_id=contract.id,
                    step_index=index,
                    step=step,
                    last_output=last_output,
                    resolution=resolution,
                    step_cursor=step_cursor if isinstance(step_cursor, dict) else None,
                )
                exec_ctx.metadata[RUNTIME_CHECKPOINT_KEY] = runtime_snapshot
                break
            if decision.type != AgentDecisionType.CONTINUE:
                break

        answer = self._build_answer(exec_ctx, last_output, run_id)
        runtime_snapshot = exec_ctx.metadata.get(RUNTIME_CHECKPOINT_KEY)
        if isinstance(runtime_snapshot, RuntimeCheckpoint):
            if answer.route is None:
                answer.route = RouteInfo(extra={})
            attach_runtime_checkpoint_to_metadata(answer.route.extra, runtime_snapshot)
        self._annotate_answer_with_shadow(answer, exec_ctx)
        self._annotate_answer_with_sandbox(answer, exec_ctx)

        if governance is not None and governance.should_pause:
            validation = ValidationResult(valid=False, errors=["awaiting human input"])
            return answer, validation, runtime_context, governance

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

        return answer, validation, runtime_context, governance

    def _verify_uaep_step_critic(
        self,
        *,
        contract: Any,
        step: AgentStep,
        step_result: StepExecutionResult,
        request: RuntimeRequest,
        run_id: str,
        task_id: str,
        task_options: Any,
        exec_ctx: RuntimeExecutionContext,
    ) -> GovernanceResolution | None:
        if not self._verify_uaep_step or self._critic_hooks is None or step_result.output is None:
            return None

        from intergrax.runtime.critic.contracts import CriticAction
        from intergrax.runtime.critic.critic_wiring import validate_uaep_step_with_critic_detail

        execution = AgentExecutionResult(
            agent_id=contract.id,
            run_id=run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary=step_result.output.summary,
            structured_data=dict(step_result.output.data),
        )
        tenant_id = str(request.tenant_id or request.metadata.get("tenant_id") or "default")
        critic_context: dict[str, object] = {}
        guardrail_scan = exec_ctx.metadata.get("guardrail_scan")
        if isinstance(guardrail_scan, dict):
            critic_context["guardrail_scan"] = guardrail_scan

        validation, verdict = validate_uaep_step_with_critic_detail(
            execution,
            contract=contract,
            hooks=self._critic_hooks,
            run_id=run_id,
            tenant_id=tenant_id,
            step_id=step.step_id,
            extra_context=critic_context or None,
        )
        if validation.valid:
            return None

        if verdict.recommended_action is CriticAction.ESCALATE_HITL:
            decision = AgentDecision(
                type=AgentDecisionType.REQUEST_HUMAN,
                reason="critic_uaep_escalate_hitl",
                payload={"blocking": True, "failure_reasons": verdict.failure_reasons},
            )
        else:
            decision = AgentDecision(
                type=AgentDecisionType.FAIL,
                reason="critic_uaep_verification_failed",
                payload={"failure_reasons": verdict.failure_reasons},
            )
        return self._interrupt_handler.resolve_decision(
            decision,
            task_id=task_id,
            run_id=run_id,
            agent_id=contract.id,
            step_id=step.step_id,
            context={
                "require_human_on_critical": task_options.governance.require_human_on_critical,
            },
        )

    @staticmethod
    def _build_runtime_checkpoint(
        *,
        request: RuntimeRequest,
        contract_id: str,
        step_index: int,
        step: AgentStep,
        last_output: Optional[StepOutput],
        resolution: GovernanceResolution,
        step_cursor: Optional[dict[str, Any]] = None,
    ) -> RuntimeCheckpoint:
        step_completed = last_output is not None and step_cursor is None
        pending_decisions: list[dict[str, Any]] = []
        if resolution.human_request is not None:
            pending_decisions.append(
                {
                    "type": "human_request",
                    "agent_id": contract_id,
                    "payload": resolution.human_request.model_dump(mode="json"),
                }
            )
        return RuntimeCheckpoint(
            plan_id=str(request.metadata.get("plan_id") or "") or None,
            graph_id=str(request.metadata.get("graph_id") or "") or None,
            graph_node_id=str(request.metadata.get("graph_node_id") or "") or None,
            agent_id=contract_id,
            uaep_step_index=step_index,
            uaep_step_id=step.step_id,
            uaep_step_completed=step_completed,
            uaep_step_cursor=step_cursor,
            paused_phase=ExecutionPhase.HUMAN_APPROVAL.value,
            plan_snapshot=(
                request.metadata.get(PLAN_SNAPSHOT_KEY)
                if isinstance(request.metadata.get(PLAN_SNAPSHOT_KEY), dict)
                else None
            ),
            pending_decisions=pending_decisions,
            pending_human_request=(
                resolution.human_request.model_dump(mode="json")
                if resolution.human_request is not None
                else None
            ),
            last_step_output=last_output.model_dump(mode="json") if last_output else None,
        )

    async def _execute_step_with_resume(
        self,
        agent: Agent,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
        cursor: dict[str, Any],
    ) -> StepExecutionResult:
        if isinstance(agent, UAEPAgentWithResume):
            output = await agent.resume_step(step, ctx, cursor)
            return StepExecutionResult(output=output)
        uaep_agent = agent if isinstance(agent, UAEPAgent) else None
        if uaep_agent is None:
            raise TypeError(f"{type(agent).__name__} is not a UAEPAgent")
        output = await uaep_agent.run_step(step, ctx)
        return StepExecutionResult(output=output)

    async def _emit_governance(
        self,
        ctx: RuntimeExecutionContext,
        resolution: GovernanceResolution,
    ) -> None:
        await self._emit(
            ctx,
            RuntimeEventType.POLICY_DECISION,
            ExecutionPhase.STEP_EXECUTION,
            {
                "policy_action": resolution.policy_decision.action.value,
                "policy_rule_id": resolution.policy_decision.policy_rule_id,
                "reason": resolution.policy_decision.reason,
            },
        )
        if resolution.interrupt is not None:
            await self._emit(
                ctx,
                RuntimeEventType.INTERRUPT_REQUESTED,
                ExecutionPhase.STEP_EXECUTION,
                {
                    "interrupt_id": resolution.interrupt.interrupt_id,
                    "interrupt_type": resolution.interrupt.interrupt_type.value,
                    "blocking": resolution.interrupt.blocking,
                },
            )
        if resolution.should_pause and resolution.human_request is not None:
            await self._emit(
                ctx,
                RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
                ExecutionPhase.HUMAN_APPROVAL,
                {
                    "request_id": resolution.human_request.request_id,
                    "prompt": resolution.human_request.prompt,
                    "options": resolution.human_request.options,
                },
            )

    async def execute_step(
        self,
        agent: Agent,
        step: AgentStep,
        ctx: RuntimeExecutionContext,
    ) -> StepExecutionResult:
        if not isinstance(agent, UAEPAgent):
            raise TypeError(
                f"execute_step requires UAEPAgent, got {type(agent).__name__}"
            )
        output = await agent.run_step(step, ctx)
        return StepExecutionResult(output=output)

    @staticmethod
    def _resolve_steps(
        agent: Agent,
        runtime_context: RuntimeContext,
        max_steps: Optional[int],
    ) -> List[AgentStep]:
        if not isinstance(agent, UAEPAgent):
            raise TypeError(f"{type(agent).__name__} is not a UAEPAgent")
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
        if isinstance(agent, UAEPAgentWithDecide):
            return agent.decide_after_step(step, output, ctx)
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

    @staticmethod
    def _attach_shared_context(
        exec_ctx: RuntimeExecutionContext,
        request: RuntimeRequest,
    ) -> None:
        shared = load_shared_task_context_from_metadata(request.metadata)
        if shared is not None:
            exec_ctx.metadata["shared_task_context"] = shared.model_dump(mode="json")

    def _attach_memory_view(
        self,
        exec_ctx: RuntimeExecutionContext,
        request: RuntimeRequest,
        *,
        task_id: str,
    ) -> None:
        if self._task_memory_store is None:
            return
        tenant_id = request.tenant_id or "default"
        access_policy = self._memory_access_policy_for_request(request.metadata)
        exec_ctx.memory_view = PolicyScopedMemoryView(
            exec_ctx,
            self._task_memory_store,
            tenant_id=tenant_id,
            task_id=task_id,
            access_policy=access_policy,
            limits=self._memory_limits,
            hook_registry=self._middleware.hooks,
            retention_days=self._retention_days_from_metadata(request.metadata),
        )
        shared = load_shared_task_context_from_metadata(request.metadata)
        if shared is not None:
            hydrate_shared_context_memory(
                self._task_memory_store,
                tenant_id=tenant_id,
                task_id=task_id,
                shared=shared,
                limits=self._memory_limits,
            )

    @staticmethod
    def _memory_access_policy_for_request(metadata: dict[str, Any]) -> MemoryAccessPolicy:
        policy = memory_access_policy_from_metadata(metadata)
        policy = apply_delegation_memory_namespace(policy, metadata)
        protected = frozenset({DEFAULT_SHARED_MEMORY_NAMESPACE})
        denied = (policy.write_denied_namespaces or frozenset()) | protected
        return MemoryAccessPolicy(
            allowed_namespaces=policy.allowed_namespaces,
            read_only=policy.read_only,
            write_denied_namespaces=denied,
            list_limit=policy.list_limit,
            scope_boundary=policy.scope_boundary,
        )

    def _attach_shadow_workspace(
        self,
        exec_ctx: RuntimeExecutionContext,
        request: RuntimeRequest,
        *,
        task_id: str,
    ) -> None:
        if not execution_options_for_request(request).isolation.shadow_workspace:
            return
        tenant_id = request.tenant_id or "default"
        workspace = self._shadow_manager.open_or_create(
            tenant_id=tenant_id,
            task_id=task_id,
        )
        exec_ctx.metadata["shadow_workspace"] = workspace
        exec_ctx.metadata[SHADOW_WORKSPACE_ID_KEY] = workspace.workspace_id

    @staticmethod
    def _annotate_answer_with_shadow(
        answer: RuntimeAnswer,
        exec_ctx: RuntimeExecutionContext,
    ) -> None:
        workspace_id = exec_ctx.metadata.get(SHADOW_WORKSPACE_ID_KEY)
        if not workspace_id:
            return
        if answer.route is None:
            answer.route = RouteInfo(extra={})
        answer.route.extra[SHADOW_WORKSPACE_ID_KEY] = workspace_id
        workspace = exec_ctx.metadata.get("shadow_workspace")
        if workspace is not None:
            answer.route.extra["shadow_artifact_count"] = len(workspace.list_artifacts())

    def _attach_sandbox_session(
        self,
        exec_ctx: RuntimeExecutionContext,
        request: RuntimeRequest,
        *,
        task_id: str,
    ) -> None:
        if not execution_options_for_request(request).isolation.sandbox:
            return
        tenant_id = request.tenant_id or "default"
        session = self._sandbox_manager.open_or_create(
            tenant_id=tenant_id,
            task_id=task_id,
        )
        exec_ctx.metadata["sandbox_session"] = session
        exec_ctx.metadata[SANDBOX_SESSION_ID_KEY] = session.session_id

    @staticmethod
    def _annotate_answer_with_sandbox(
        answer: RuntimeAnswer,
        exec_ctx: RuntimeExecutionContext,
    ) -> None:
        session_id = exec_ctx.metadata.get(SANDBOX_SESSION_ID_KEY)
        if not session_id:
            return
        if answer.route is None:
            answer.route = RouteInfo(extra={})
        answer.route.extra[SANDBOX_SESSION_ID_KEY] = session_id
        session = exec_ctx.metadata.get("sandbox_session")
        if session is not None:
            answer.route.extra["sandbox_operation_count"] = len(session.audit_log)

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

    async def _scan_step_output_hooks(
        self,
        hook_step: HookContext,
        step_output: StepOutput,
        request: RuntimeRequest,
        *,
        exec_ctx: RuntimeExecutionContext,
    ) -> None:
        output_text = step_output.summary or str(step_output.data.get("answer", ""))
        if not output_text:
            output_text = str(step_output.data.get("text", ""))
        if not output_text:
            return
        ctx = hook_step.model_copy(
            update={
                "runtime_state": {
                    **hook_step.runtime_state,
                    "prompt": request.message or "",
                    "llm_output": output_text,
                    "output": output_text,
                },
            },
        )
        hook_result = await self._middleware.run_after(HookPoint.AFTER_LLM_OUTPUT, ctx)
        if hook_result.modified_payload and "guardrail_scan" in hook_result.modified_payload:
            exec_ctx.metadata["guardrail_scan"] = hook_result.modified_payload["guardrail_scan"]
        await self._guard_hook(hook_result)

    @staticmethod
    async def _guard_hook(result: Any) -> None:
        if result.action != HookAction.ALLOW:
            raise UAEPBlockedError(result.reason or f"hook blocked: {result.action.value}")
