# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified Agent Execution Protocol (§42.5) — runtime-controlled agent loop."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, List, Optional
from uuid import uuid4

from intergrax.agents.agent_contract import Agent
from intergrax.agents.authoring.uaep_step_bridge import (
    build_kernel_session,
    execute_uaep_step_via_kernel,
    trace_summary_from_kernel,
)
from intergrax.agents.uaep_protocol import (
    UAEPAgent,
    UAEPAgentWithDecide,
    UAEPAgentWithResume,
)
from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.uaep_bridge_keys import UaepBridgeMetadataKey
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
from intergrax.contracts.execution_identity import require_active_execution_identity, require_active_execution_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.runtime_policy_context import AgentDecisionPolicyContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.canonical import ContextAssemblyPayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.hooks.governance_hooks import hook_context_for_task, run_hook_pair
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler, GovernanceResolution
from intergrax.runtime.policy.agent_decision_enforcement import (
    agent_decision_failure_from_resolution,
)
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.middleware.trace_middleware import TraceEmittingMiddleware
from intergrax.runtime.nexus.tools.uaep_tool_gateway import BoundToolGateway
from intergrax.runtime.decision_flow import DecisionFlowGate
from intergrax.runtime.task.task_contract import TaskExecutionOptions
from intergrax.runtime.policy.policy_engine import PolicyEngine, coerce_policy_engine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RouteInfo, RuntimeRequest
from intergrax.runtime.sandbox.manager import SandboxSessionManager

if TYPE_CHECKING:
    from intergrax.runtime.migration.critic_shadow_adapter import CriticShadowAdapter
    from intergrax.runtime.migration.decision_critic_parity import DecisionCriticParityObserver
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_SESSION_ID_KEY
from intergrax.runtime.task.task_metadata_bridge import execution_options_for_request
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.cancellation.coordinator import (
    CANCELLATION_REQUESTED_KEY,
    CancellationCoordinator,
)
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    validate_task_id,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    should_resume_uaep_step,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import ExecutionTreeRecorder
from intergrax.runtime.long_running.runtime_checkpoint import (
    PLAN_SNAPSHOT_KEY,
    PendingDecision,
    RuntimeCheckpoint,
    UAEP_STEP_CURSOR_KEY,
    UaepStepCursor,
    UaepStepOutput,
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


def _tenant_id_from_ctx(ctx: RuntimeExecutionContext) -> str:
    request = ctx.request
    if isinstance(request, RuntimeRequest):
        tenant_id = request.tenant_id or request.metadata.get("tenant_id")
        if tenant_id:
            return str(tenant_id)
    elif request is not None:
        tenant_id = request.metadata.get("tenant_id")
        if tenant_id:
            return str(tenant_id)
    raw = ctx.metadata.get("tenant_id")
    if raw:
        return str(raw)
    return "default"


class _BusEventEmitter:
    def __init__(self, bus: RuntimeEventBus) -> None:
        self._bus = bus

    async def emit(self, event: RuntimeEvent) -> None:
        await self._bus.publish(event)


class UAEPExecutor:
    """
    Executes agents through the Unified Agent Execution Protocol (§42.5).

    Executes UAEP-capable agents via the harness kernel step bridge.
    """

    def __init__(
        self,
        *,
        middleware: Optional[MiddlewarePipeline] = None,
        event_bus: Optional[RuntimeEventBus] = None,
        policy_engine: PolicyEngine | RuntimePolicyEngine | None = None,
        interrupt_handler: Optional[ExecutionInterruptHandler] = None,
        governance_service: Any = None,
        shadow_manager: Optional[ShadowWorkspaceManager] = None,
        sandbox_manager: Optional[SandboxSessionManager] = None,
        task_memory_store: Optional[TaskMemoryPersistence] = None,
        memory_limits: Optional[TaskMemoryLimits] = None,
        decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None = None,
        verify_uaep_step_decision: bool = False,
        context_engine: Any = None,
        llm_adapter: Any = None,
    ) -> None:
        resolved_policy = coerce_policy_engine(policy_engine)
        self._policy_engine = resolved_policy
        self._event_bus = event_bus
        self._interrupt_handler = interrupt_handler or ExecutionInterruptHandler(
            policy_engine=resolved_policy,
        )
        self._governance_service = governance_service
        self._shadow_manager = shadow_manager or ShadowWorkspaceManager()
        self._sandbox_manager = sandbox_manager or SandboxSessionManager()
        self._task_memory_store = task_memory_store
        self._memory_limits = memory_limits or TaskMemoryLimits()
        self._decision_flow_gate = decision_flow_gate
        self._verify_uaep_step_decision = verify_uaep_step_decision
        self._critic_parity_shadow = None
        self._parity_observer = None
        self._context_engine = context_engine
        self._llm_adapter = llm_adapter
        if middleware is not None:
            self._middleware = middleware
        elif event_bus is not None:
            self._middleware = MiddlewarePipeline(
                middleware=[TraceEmittingMiddleware(event_bus)],
            )
        else:
            self._middleware = MiddlewarePipeline()

    def set_decision_flow_gate(
        self,
        gate: DecisionFlowGate[AgentExecutionResult] | None,
        *,
        verify_uaep_step: bool = False,
    ) -> None:
        self._decision_flow_gate = gate
        self._verify_uaep_step_decision = verify_uaep_step

    def set_critic_parity_shadow(
        self,
        shadow: CriticShadowAdapter | None,
        *,
        observer: DecisionCriticParityObserver | None = None,
    ) -> None:
        """Attach observational Critic shadow for Decision parity (migration-only)."""
        self._critic_parity_shadow = shadow
        self._parity_observer = observer

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

    @property
    def shadow_manager(self) -> ShadowWorkspaceManager:
        return self._shadow_manager

    @property
    def sandbox_manager(self) -> SandboxSessionManager:
        return self._sandbox_manager

    async def execute(
        self,
        agent: Agent,
        request: RuntimeRequest,
        *,
        contract: AgentContract | None = None,
    ) -> tuple[RuntimeAnswer, ValidationResult, Optional[GovernanceResolution]]:
        contract = contract or agent.get_contract()
        task_options = execution_options_for_request(request)
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        task_id = request.task_id
        node_id = request.metadata.get("graph_node_id")

        exec_ctx = RuntimeExecutionContext(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
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

        kernel_ctx = build_kernel_session(
            agent_id=contract.id,
            run_id=run_id,
            task_id=task_id,
            tenant_id=str(request.tenant_id or request.metadata.get("tenant_id") or "default"),
            max_steps=contract.max_steps,
            policy_engine=self._interrupt_handler.policy_engine,
            request=request,
        )
        exec_ctx.metadata[UaepBridgeMetadataKey.KERNEL_SESSION] = kernel_ctx

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
        try:
            from intergrax.agents.authoring.acp_uaep_shim import apply_host_tool_invoker_to_runtime_context
    
            apply_host_tool_invoker_to_runtime_context(runtime_context, request.metadata)
            if self._context_engine is not None:
                runtime_context.config.context_engine = self._context_engine
            from intergrax.runtime.attestation.kernel_wiring import apply_boundary_export_to_kernel
    
            apply_boundary_export_to_kernel(kernel_ctx, runtime_context.config)
            from intergrax.runtime.nexus.context.memory_context_invocation import (
                populate_request_memory_recall_metadata,
            )
    
            await populate_request_memory_recall_metadata(
                request,
                config=runtime_context.config,
                session_manager=runtime_context.session_manager,
            )
            if self._context_engine is not None and self._llm_adapter is not None:
                from intergrax.llm.messages import StructuredModelInputRequiredError, STRUCTURED_MODEL_INPUT_REQUIRED_REASON
                from intergrax.runtime.nexus.context.graph_assembly import text_from_assembled_messages
                from intergrax.runtime.nexus.context.uaep_assemble import assemble_uaep_session_messages
    
                assembled_messages = await assemble_uaep_session_messages(
                    request,
                    agent_id=contract.id,
                    engine=self._context_engine,
                    llm_adapter=self._llm_adapter,
                    event_bus=self._event_bus,
                )
                try:
                    assembled_prompt = text_from_assembled_messages(assembled_messages)
                except StructuredModelInputRequiredError as exc:
                    raise UAEPBlockedError(STRUCTURED_MODEL_INPUT_REQUIRED_REASON) from exc
                if assembled_prompt and assembled_prompt != (request.message or ""):
                    request = replace(request, message=assembled_prompt)
            exec_ctx.domain_context = runtime_context
            from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
    
            exec_ctx.metadata["runtime_state"] = RuntimeState(
                context=runtime_context,
                request=request,
                run_id=run_id,
                declarative_hitl_grant=request.declarative_hitl_grant,
            )
            from intergrax.runtime.wiring.llm_routing_runtime_bridge import (
                sync_llm_routing_snapshot_for_state,
                wire_llm_routing_observability_on_state,
            )
    
            runtime_state = exec_ctx.metadata["runtime_state"]
            assert isinstance(runtime_state, RuntimeState)
            wire_llm_routing_observability_on_state(runtime_state)
            sync_llm_routing_snapshot_for_state(runtime_state)
            exec_ctx.tool_gateway = BoundToolGateway(
                exec_ctx,
                allowed_tools=list(contract.allowed_tools),
                middleware=self._middleware,
            )
            from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
                attach_functional_evidence_recorder_from_runtime_state,
            )

            attach_functional_evidence_recorder_from_runtime_state(exec_ctx)
    
            await self._guard_hook(
                await self._middleware.run_after(
                    HookPoint.AFTER_CONTEXT_BUILD,
                    hook_base.model_copy(update={"phase": ExecutionPhase.CONTEXT_BUILDING}),
                )
            )
            prompt_text = str(request.message or "")
            context_chars = len(prompt_text)
            await self._emit_context_assembled(
                exec_ctx,
                node_id=exec_ctx.node_id or contract.id,
                agent_id=contract.id,
                context_original_chars=context_chars,
                context_final_chars=context_chars,
                engine_id="default",
            )
    
            steps = self._resolve_steps(agent, runtime_context, contract.max_steps)
            last_output: Optional[StepOutput] = None
            governance: Optional[GovernanceResolution] = None
            runtime_ckpt = request.runtime_checkpoint
            uaep_resume_approval = None
            pause_record = request.hitl_pause_record
            if pause_record is not None and request.task_id:
                uaep_resume_approval = HumanPauseCoordinator.approved_resolution_for_resume(
                    task_id=request.task_id,
                    resolution=request.hitl_resolution,
                    expected_pause_id=pause_record.pause_id,
                    expected_human_request_id=pause_record.human_request_id,
                    run_id=run_id,
                )
    
            for index, step in enumerate(steps):
                if CancellationCoordinator.is_requested(request.metadata):
                    validation = ValidationResult(valid=False, errors=["task_cancelled"])
                    answer = self._build_answer(exec_ctx, last_output, run_id)
                    if answer.route is None:
                        answer.route = RouteInfo(extra={})
                    answer.route.extra[CANCELLATION_REQUESTED_KEY] = True
                    return answer, validation, None
    
                exec_ctx.phase = ExecutionPhase.STEP_EXECUTION
                hook_step = hook_base.model_copy(
                    update={"step_id": step.step_id, "phase": ExecutionPhase.STEP_EXECUTION},
                )
                await self._guard_hook(
                    await self._middleware.run_before(HookPoint.BEFORE_STEP, hook_step)
                )
    
                runtime_state = exec_ctx.metadata.get("runtime_state")
                if isinstance(runtime_state, RuntimeState):
                    from intergrax.runtime.wiring.llm_routing_runtime_bridge import (
                        sync_llm_routing_snapshot_for_state,
                    )
    
                    request.metadata["step_index"] = index
                    sync_llm_routing_snapshot_for_state(runtime_state)
    
                started = time.perf_counter()
                try:
                    if should_skip_uaep_step(
                        step_index=index,
                        step_id=step.step_id,
                        checkpoint=runtime_ckpt,
                        approval=uaep_resume_approval,
                    ):
                        last_output = StepOutput.model_validate(runtime_ckpt.last_step_output.model_dump())
                        step_result = StepExecutionResult(output=last_output)
                    elif should_resume_uaep_step(
                        step_index=index,
                        step_id=step.step_id,
                        checkpoint=runtime_ckpt,
                        approval=uaep_resume_approval,
                    ):
                        assert runtime_ckpt is not None
                        exec_ctx.metadata[UAEP_STEP_CURSOR_KEY] = dict(runtime_ckpt.uaep_step_cursor.values)
                        step_result = await self._execute_step_with_resume(
                            agent,
                            step,
                            exec_ctx,
                            runtime_ckpt.uaep_step_cursor.values if runtime_ckpt.uaep_step_cursor else {},
                        )
                    else:
                        step_result = await self.execute_step(agent, step, exec_ctx)
                except Exception as exc:
                    from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
                        DeclarativePolicyHitlPauseRequired,
                    )
    
                    if not isinstance(exc, DeclarativePolicyHitlPauseRequired):
                        raise
                    governance = exc.governance.model_copy(
                        update={"declarative_hitl_pending": exc.pending}
                    )
                    exec_ctx.metadata["governance_resolution"] = governance
                    runtime_snapshot = self._build_runtime_checkpoint(
                        request=request,
                        contract_id=contract.id,
                        step_index=index,
                        step=step,
                        last_output=last_output,
                        resolution=governance,
                        step_cursor=exec_ctx.metadata.get(UAEP_STEP_CURSOR_KEY)
                        if isinstance(exec_ctx.metadata.get(UAEP_STEP_CURSOR_KEY), dict)
                        else None,
                    )
                    request.runtime_checkpoint = runtime_snapshot
                    answer = self._build_answer(exec_ctx, last_output, run_id)
                    validation = ValidationResult(valid=False, errors=["awaiting human input"])
                    return answer, validation, governance
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
    
                critic_resolution = await self._verify_uaep_step_authority(
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
                    request.runtime_checkpoint = runtime_snapshot
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
                replan_context: dict[str, object] = {}
                replan_policy = request.metadata.get("replan_policy.v1")
                if isinstance(replan_policy, dict):
                    replan_context.update(replan_policy)
                resolution = self._interrupt_handler.resolve_decision(
                    decision,
                    task_id=task_id,
                    run_id=run_id,
                    agent_id=contract.id,
                    step_id=step.step_id,
                    context=replan_context or None,
                    decision_policy_context=AgentDecisionPolicyContext(
                        require_human_on_critical=task_options.governance.require_human_on_critical,
                        has_unresolved_critical_interrupt=bool(
                            exec_ctx.metadata.get("has_unresolved_critical_interrupt", False)
                        ),
                    ),
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
    
                from intergrax.contracts.uaep_decision_record import DecisionRecord
    
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
    
                if (
                    decision.type is AgentDecisionType.MODIFY_PLAN
                    and not resolution.should_fail
                    and not resolution.should_block_execution
                ):
                    from intergrax.contracts.agent_handoff import handoff_from_decision
                    from intergrax.runtime.execution.budget.consumption import consume_replan
    
                    if handoff_from_decision(decision) is None:
                        consume_replan()
    
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
                    request.runtime_checkpoint = runtime_snapshot
                    break
                if resolution.should_block_execution:
                    governance = resolution
                    exec_ctx.metadata["governance_resolution"] = resolution
                    last_output = StepOutput(
                        step_id=step.step_id,
                        summary=resolution.policy_decision.reason or "policy_denied",
                        data={"policy_action": resolution.policy_decision.action.value},
                    )
                    decision = agent_decision_failure_from_resolution(resolution)
                    break
                if decision.type != AgentDecisionType.CONTINUE:
                    break
    
            answer = self._build_answer(exec_ctx, last_output, run_id)
            if last_output is not None and isinstance(last_output.data, dict):
                # Promote typed domain summaries into TaskResult.structured_data via route.extra.
                for key in ("search_summary", "ingest_summary", "domain_summary"):
                    value = last_output.data.get(key)
                    if isinstance(value, dict) and value:
                        if answer.route is None:
                            answer.route = RouteInfo(extra={})
                        answer.route.extra[key] = dict(value)
            bridged_kernel = exec_ctx.metadata.get(UaepBridgeMetadataKey.KERNEL_SESSION)
            if bridged_kernel is not None:
                if answer.route is None:
                    answer.route = RouteInfo(extra={})
                answer.route.extra[AcpStructuredDataKey.TRACE_SUMMARY] = trace_summary_from_kernel(
                    bridged_kernel
                )
            self._annotate_answer_with_shadow(answer, exec_ctx)
            self._annotate_answer_with_sandbox(answer, exec_ctx)
    
            if governance is not None and governance.should_pause:
                validation = ValidationResult(valid=False, errors=["awaiting human input"])
                return answer, validation, governance
    
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
    
            from intergrax.runtime.governance.post_run_governance_bridge import (
                invoke_post_run_governance,
            )
    
            invoke_post_run_governance(
                self._governance_service,
                run_id=run_id,
                agent_id=contract.id,
            )
    
            return answer, validation, governance

        finally:
            runtime_context.close()

    async def _verify_uaep_step_authority(
        self,
        *,
        contract: AgentContract,
        step: AgentStep,
        step_result: StepExecutionResult,
        request: RuntimeRequest,
        run_id: str,
        task_id: str,
        task_options: TaskExecutionOptions,
        exec_ctx: RuntimeExecutionContext,
    ) -> GovernanceResolution | None:
        if step_result.output is None:
            return None
        if (
            self._decision_flow_gate is not None
            and self._verify_uaep_step_decision
        ):
            from intergrax.runtime.decision_flow import DecisionFlowScope

            if self._decision_flow_gate.supports_scope(DecisionFlowScope.UAEP_STEP):
                return await self._verify_uaep_step_decision_flow(
                    contract=contract,
                    step=step,
                    step_result=step_result,
                    request=request,
                    run_id=run_id,
                    task_id=task_id,
                    task_options=task_options,
                    exec_ctx=exec_ctx,
                )
        return None

    async def _verify_uaep_step_decision_flow(
        self,
        *,
        contract: AgentContract,
        step: AgentStep,
        step_result: StepExecutionResult,
        request: RuntimeRequest,
        run_id: str,
        task_id: str,
        task_options: TaskExecutionOptions,
        exec_ctx: RuntimeExecutionContext,
    ) -> GovernanceResolution | None:
        from intergrax.contracts.execution_identity import require_active_execution_identity
        from intergrax.runtime.decision_flow import DecisionFlowHostAction, DecisionFlowScope
        from intergrax.runtime.decision_flow_host import (
            agent_execution_decision_context,
            agent_execution_identity_seed,
            build_agent_execution_flow_request,
            evaluate_agent_execution_flow,
        )

        gate = self._decision_flow_gate
        if gate is None:
            return None

        execution = AgentExecutionResult(
            agent_id=contract.id,
            run_id=run_id,
            status=AgentExecutionStatus.COMPLETED,
            summary=step_result.output.summary,
            structured_data=dict(step_result.output.data),
        )
        tenant_id = str(request.tenant_id or request.metadata.get("tenant_id") or "default")
        active_run_id, active_attempt_id = require_active_execution_identity()
        decision_context = agent_execution_decision_context(
            task_id=task_id,
            run_id=active_run_id,
            attempt_id=active_attempt_id,
            tenant_id=tenant_id,
        )
        identity_seed = agent_execution_identity_seed(
            context=decision_context,
            namespace="uaep.step",
            subject=step.step_id,
        )
        flow_request = build_agent_execution_flow_request(
            execution=execution,
            identity_seed=identity_seed,
            flow_scope=DecisionFlowScope.UAEP_STEP,
        )
        flow_result = await evaluate_agent_execution_flow(
            gate,
            flow_request,
        )
        if self._critic_parity_shadow is not None:
            from intergrax.runtime.migration.critic_shadow_adapter import observe_uaep_step_parity

            critic_context: dict[str, object] = {}
            guardrail_scan = exec_ctx.metadata.get("guardrail_scan")
            if isinstance(guardrail_scan, dict):
                critic_context["guardrail_scan"] = guardrail_scan
            await observe_uaep_step_parity(
                shadow=self._critic_parity_shadow,
                decision_result=flow_result,
                execution=execution,
                contract=contract,
                task_id=task_id,
                run_id=active_run_id,
                attempt_id=active_attempt_id,
                tenant_id=tenant_id,
                step_id=step.step_id,
                observer=self._parity_observer,
                extra_context=critic_context or None,
            )
        if flow_result.host_action is DecisionFlowHostAction.CONTINUE:
            return None
        if flow_result.host_action is DecisionFlowHostAction.PENDING_HUMAN:
            decision = AgentDecision(
                type=AgentDecisionType.REQUEST_HUMAN,
                reason="decision_uaep_human_review_pending",
                payload={
                    "blocking": True,
                    "human_review_request_id": str(
                        flow_result.human_review_pending.request.request_id,
                    )
                    if flow_result.human_review_pending is not None
                    else "",
                },
            )
        else:
            decision = AgentDecision(
                type=AgentDecisionType.FAIL,
                reason=flow_result.authority_reason or "decision_uaep_verification_failed",
                payload={
                    "resolution": (
                        flow_result.resolution_record.resolution.value
                        if flow_result.resolution_record is not None
                        else "rejected"
                    ),
                },
            )
        return self._interrupt_handler.resolve_decision(
            decision,
            task_id=task_id,
            run_id=run_id,
            agent_id=contract.id,
            step_id=step.step_id,
            decision_policy_context=AgentDecisionPolicyContext(
                require_human_on_critical=task_options.governance.require_human_on_critical,
            ),
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
        step_cursor: Optional[dict[str, bool]] = None,
    ) -> RuntimeCheckpoint:
        run_id, attempt_id = require_active_execution_identity()
        root_execution_id = require_active_execution_id()
        step_completed = last_output is not None and step_cursor is None
        pending_decisions: list[PendingDecision] = []
        if resolution.human_request is not None:
            pending_decisions.append(
                PendingDecision(
                    type="human_request",
                    agent_id=contract_id,
                    payload=resolution.human_request.model_dump(mode="json"),
                )
            )
        existing = request.runtime_checkpoint
        if existing is not None:
            execution_tree = existing.execution_tree
        else:
            execution_tree = ExecutionTreeRecorder.start_root(
                task_id=validate_task_id(request.task_id),
                run_id=run_id,
                attempt_id=attempt_id,
                root_execution_id=root_execution_id,
            ).snapshot
        return RuntimeCheckpoint(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_tree=execution_tree,
            plan_id=str(request.metadata.get("plan_id") or "") or None,
            graph_id=str(request.metadata.get("graph_id") or "") or None,
            graph_node_id=str(request.metadata.get("graph_node_id") or "") or None,
            agent_id=contract_id,
            uaep_step_index=step_index,
            uaep_step_id=step.step_id,
            uaep_step_completed=step_completed,
            uaep_step_cursor=(
                UaepStepCursor(values=dict(step_cursor)) if step_cursor is not None else None
            ),
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
            last_step_output=(
                UaepStepOutput(
                    step_id=last_output.step_id,
                    summary=last_output.summary,
                )
                if last_output is not None
                else None
            ),
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
        kernel_ctx = ctx.metadata.get(UaepBridgeMetadataKey.KERNEL_SESSION)
        if kernel_ctx is not None:
            return await execute_uaep_step_via_kernel(agent, step, ctx, kernel_ctx)
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
        from intergrax.agents.authoring.base import IntergraxAgent
        from intergrax.agents.authoring.uaep_linear_bridge import linear_agent_decide_after_step

        if isinstance(agent, IntergraxAgent):
            return linear_agent_decide_after_step(agent, step, output, ctx)
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
        from intergrax.runtime.workspace.exec_ctx_isolation import attach_shadow_workspace_to_exec_ctx

        attach_shadow_workspace_to_exec_ctx(
            exec_ctx,
            request,
            shadow_manager=self._shadow_manager,
            task_id=task_id,
        )

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
        from intergrax.runtime.workspace.exec_ctx_isolation import attach_sandbox_session_to_exec_ctx

        attach_sandbox_session_to_exec_ctx(
            exec_ctx,
            request,
            sandbox_manager=self._sandbox_manager,
            task_id=task_id,
        )

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

    async def _emit_context_assembled(
        self,
        ctx: RuntimeExecutionContext,
        *,
        node_id: str,
        agent_id: str,
        context_original_chars: int,
        context_final_chars: int,
        engine_id: str = "default",
    ) -> None:
        if self._event_bus is None:
            return
        typed = ContextAssemblyPayloadV1(
            node_id=node_id,
            context_original_chars=context_original_chars,
            context_final_chars=context_final_chars,
            trimmed=False,
            engine_id=engine_id,
        )
        event = runtime_event_with_payload(
            RuntimeEvent(
                task_id=ctx.task_id,
                run_id=ctx.run_id,
                attempt_id=ctx.attempt_id,
                execution_id=ctx.execution_id,
                node_id=ctx.node_id,
                agent_id=agent_id,
                tenant_id=_tenant_id_from_ctx(ctx),
                event_type=RuntimeEventType.CONTEXT_ASSEMBLED,
                phase=ExecutionPhase.CONTEXT_BUILDING,
                correlation_id=ctx.correlation_id or ctx.task_id,
            ),
            typed,
            promote_fields={
                "node_id": node_id,
                "engine_id": engine_id,
                "context_original_chars": context_original_chars,
                "context_final_chars": context_final_chars,
            },
        )
        await self._event_bus.publish(event)

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
            attempt_id=ctx.attempt_id,
            execution_id=ctx.execution_id,
            node_id=ctx.node_id,
            agent_id=ctx.agent_id,
            tenant_id=_tenant_id_from_ctx(ctx),
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
