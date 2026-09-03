# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution graph phase extracted from NexusLoop (Phase Q-N.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.request_contract import human_request_event_payload
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNodeStatus,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.orchestration.graph_trace_callbacks import GraphTraceCallbacks
from intergrax.runtime.nexus.orchestration.hitl_runner import NexusHitlRunner
from intergrax.runtime.nexus.orchestration.task_events import NexusRuntimeEventPublisher
from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
from intergrax.runtime.nexus.retry.retry_engine import (
    RetryPolicy,
    RetryRecord,
    _resilience_policy_from_task,
)
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.decision_flow import DecisionFlowHostAction, DecisionFlowScope
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import PersistingTaskTraceEmitter, TaskTraceEmitter
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.critic.critic_wiring import CriticGraphHooks
    from intergrax.runtime.critic.trace import CriticTraceEmitter
    from intergrax.runtime.decision_flow import DecisionFlowGate
    from intergrax.runtime.migration.critic_shadow_adapter import CriticShadowAdapter
    from intergrax.runtime.migration.decision_critic_parity import DecisionCriticParityObserver

FinishFn = Callable[..., Awaitable[TaskResult]]
FinalizeFn = Callable[..., Awaitable[None]]
CheckpointFn = Callable[..., Awaitable[None]]


@dataclass(slots=True)
class GraphPhaseOutcome:
    """Result of the graph execution phase; ``early_result`` short-circuits NexusLoop."""

    early_result: Optional[TaskResult] = None
    executions: List[AgentExecutionResult] | None = None
    retry_records: List[RetryRecord] | None = None
    graph: Optional[ExecutionGraph] = None
    plan: Optional[NexusPlan] = None
    final_validation: Optional[ValidationResult] = None


@dataclass
class NexusGraphRunner:
    registry: AgentRegistry
    graph_executor: GraphExecutor
    validation_engine: NexusValidationEngine
    composer: FinalResponseComposer
    hitl: NexusHitlRunner
    events: NexusRuntimeEventPublisher
    finish_task: FinishFn
    finalize_trace: FinalizeFn
    maybe_checkpoint: CheckpointFn
    max_run_retries: int = 0
    critic_graph_hooks: CriticGraphHooks | None = None
    decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None = None
    critic_parity_shadow: CriticShadowAdapter | None = None
    parity_observer: DecisionCriticParityObserver | None = None

    async def run(
        self,
        task: Task,
        *,
        plan: NexusPlan,
        graph: ExecutionGraph,
        lifecycle: TaskLifecycle,
        trace_emitter: TaskTraceEmitter,
    ) -> GraphPhaseOutcome:
        callbacks = GraphTraceCallbacks(task=task, trace_emitter=trace_emitter)
        critic_trace_emitter = _build_critic_trace_emitter(
            task=task,
            trace_emitter=trace_emitter,
            hooks=self.critic_graph_hooks,
        )
        retry_codes = (
            frozenset({RuntimeErrorCode.VALIDATION_ERROR})
            if self.max_run_retries > 0
            else frozenset()
        )
        coordinator = RetryCoordinator(
            max_run_retries=self.max_run_retries,
            retry_run_on=retry_codes,
        )

        if plan.graph_retry_on_error is not None:
            self.graph_executor.set_retry_policy(
                RetryPolicy(max_retries=plan.graph_retry_on_error),
            )

        async def on_retry(record: RetryRecord) -> None:
            callbacks.on_retry(record)
            run_id, attempt_id = self.graph_executor.execution_identity.require()
            await self.events.publish(
                coordinator.scheduled_event_for_agent_retry(
                    task,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    record=record,
                ),
                task=task,
            )
            new_attempt_id = self.graph_executor.execution_identity.transition_retry()
            await self.events.publish(
                RetryCoordinator.build_started_event(
                    task,
                    run_id=run_id,
                    attempt_id=new_attempt_id,
                    scope="agent",
                    retry_ordinal=record.attempt,
                    reason=record.reason,
                ),
                task=task,
            )

        run_attempt = 0
        executions: List[AgentExecutionResult] = []
        retry_records: List[RetryRecord] = []
        graph_cancelled = False
        while True:
            executions, attempt_retries, graph, graph_cancelled = await self.graph_executor.execute(
                graph,
                task,
                plan_criteria=plan.validation_criteria,
                on_retry=on_retry,
                on_node_start=callbacks.on_node_start,
                on_node_complete=callbacks.on_node_complete,
                critic_trace_emitter=critic_trace_emitter,
            )
            retry_records.extend(attempt_retries)
            failed_nodes = [
                n.node_id for n in graph.nodes if n.status == ExecutionNodeStatus.FAILED
            ]
            if not failed_nodes or graph_cancelled:
                break
            if not coordinator.should_retry_run(
                attempt=run_attempt,
                error_code=RuntimeErrorCode.VALIDATION_ERROR,
            ):
                break
            run_id, attempt_id = self.graph_executor.execution_identity.require()
            await self.events.publish(
                coordinator.scheduled_event_for_run_retry(
                    task,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    attempt=run_attempt + 1,
                    error_code=RuntimeErrorCode.VALIDATION_ERROR,
                ),
                task=task,
            )
            new_attempt_id = self.graph_executor.execution_identity.transition_retry()
            await self.events.publish(
                RetryCoordinator.build_started_event(
                    task,
                    run_id=run_id,
                    attempt_id=new_attempt_id,
                    scope="run",
                    retry_ordinal=run_attempt + 1,
                    reason=RuntimeErrorCode.VALIDATION_ERROR.value,
                ),
                task=task,
            )
            run_attempt += 1
            for node in graph.nodes:
                if node.status is ExecutionNodeStatus.FAILED:
                    node.status = ExecutionNodeStatus.PENDING
                    node.execution_result = None

        if graph_cancelled or CancellationCoordinator.is_requested(task.metadata):
            return await self._handle_cancellation(
                task,
                plan=plan,
                graph=graph,
                executions=executions,
                retry_records=retry_records,
                lifecycle=lifecycle,
                trace_emitter=trace_emitter,
            )

        if executions and executions[-1].status == AgentExecutionStatus.NEEDS_INPUT:
            return await self._handle_needs_input(
                task,
                plan=plan,
                graph=graph,
                executions=executions,
                retry_records=retry_records,
                lifecycle=lifecycle,
                trace_emitter=trace_emitter,
            )

        failed_nodes = [
            n.node_id for n in graph.nodes if n.status == ExecutionNodeStatus.FAILED
        ]
        if failed_nodes:
            return await self._handle_graph_failure(
                task,
                plan=plan,
                graph=graph,
                executions=executions,
                retry_records=retry_records,
                failed_nodes=failed_nodes,
                lifecycle=lifecycle,
                trace_emitter=trace_emitter,
            )

        lifecycle.transition(task, TaskState.VALIDATING)
        final_validation = ValidationResult(valid=True)
        if executions:
            final_agent = self.registry.get(executions[-1].agent_id)
            final_contract = final_agent.get_contract()
            active_run_id, active_attempt_id = require_active_execution_identity()
            if (
                self.decision_flow_gate is not None
                and self.decision_flow_gate.supports_scope(DecisionFlowScope.GRAPH_FINAL)
            ):
                from intergrax.runtime.decision_flow_host import (
                    agent_execution_decision_context,
                    agent_execution_identity_seed,
                    build_agent_execution_flow_request,
                    decision_flow_result_to_validation_result,
                    evaluate_agent_execution_flow,
                )

                decision_context = agent_execution_decision_context(
                    task_id=task.task_id,
                    run_id=active_run_id,
                    attempt_id=active_attempt_id,
                    tenant_id=task.tenant_id,
                )
                identity_seed = agent_execution_identity_seed(
                    context=decision_context,
                    namespace="graph.final",
                    subject=graph.graph_id,
                )
                flow_request = build_agent_execution_flow_request(
                    execution=executions[-1],
                    identity_seed=identity_seed,
                    flow_scope=DecisionFlowScope.GRAPH_FINAL,
                )
                flow_result = await evaluate_agent_execution_flow(
                    self.decision_flow_gate,
                    flow_request,
                )
                if self.critic_parity_shadow is not None:
                    from intergrax.runtime.migration.critic_shadow_adapter import (
                        observe_graph_final_parity,
                    )

                    await observe_graph_final_parity(
                        shadow=self.critic_parity_shadow,
                        decision_result=flow_result,
                        execution=executions[-1],
                        contract=final_contract,
                        task_id=task.task_id,
                        run_id=active_run_id,
                        attempt_id=active_attempt_id,
                        tenant_id=task.tenant_id,
                        graph_id=graph.graph_id,
                        capability=task.context.capability,
                        plan_criteria=tuple(plan.validation_criteria or ()),
                        observer=self.parity_observer,
                    )
                if flow_result.host_action is DecisionFlowHostAction.PENDING_HUMAN:
                    executions[-1] = executions[-1].model_copy(
                        update={"status": AgentExecutionStatus.NEEDS_INPUT},
                    )
                    return await self._handle_needs_input(
                        task,
                        plan=plan,
                        graph=graph,
                        executions=executions,
                        retry_records=retry_records,
                        lifecycle=lifecycle,
                        trace_emitter=trace_emitter,
                    )
                final_validation = decision_flow_result_to_validation_result(flow_result)
            elif (
                self.critic_graph_hooks is not None
                and self.critic_graph_hooks.verify_graph_final
            ):
                from intergrax.runtime.critic.critic_wiring import validate_final_with_critic

                active_run_id, _ = require_active_execution_identity()
                final_validation = validate_final_with_critic(
                    executions[-1],
                    contract=final_contract,
                    hooks=self.critic_graph_hooks,
                    task_id=task.task_id,
                    run_id=active_run_id,
                    tenant_id=task.tenant_id,
                    capability=task.context.capability,
                    plan_criteria=plan.validation_criteria,
                    trace_emitter=critic_trace_emitter,
                )
            else:
                final_validation = self.validation_engine.validate(
                    executions[-1],
                    contract=final_contract,
                    capability=task.context.capability,
                    plan_criteria=plan.validation_criteria,
                )

        if not final_validation.valid:
            lifecycle.transition(task, TaskState.FAILED)
        elif len(executions) > 1 and not all(
            e.status == AgentExecutionStatus.COMPLETED for e in executions
        ):
            policy = _resilience_policy_from_task(task)
            if policy is not None and not policy.allow_partial_result:
                lifecycle.transition(task, TaskState.FAILED)
            else:
                lifecycle.transition(task, TaskState.PARTIALLY_COMPLETED)
        elif task.runtime.orchestration.needs_more_information:
            lifecycle.transition(task, TaskState.NEEDS_MORE_INFORMATION)
        else:
            lifecycle.transition(task, TaskState.COMPLETED)

        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self.finalize_trace(trace_emitter, executions, task_id=task.task_id)

        return GraphPhaseOutcome(
            executions=executions,
            retry_records=retry_records,
            graph=graph,
            plan=plan,
            final_validation=final_validation,
        )

    async def _handle_cancellation(
        self,
        task: Task,
        *,
        plan: NexusPlan,
        graph: ExecutionGraph,
        executions: List[AgentExecutionResult],
        retry_records: List[RetryRecord],
        lifecycle: TaskLifecycle,
        trace_emitter: TaskTraceEmitter,
    ) -> GraphPhaseOutcome:
        await self.events.publish_from_task_state(
            task,
            message="task cancellation propagated",
            event_type=RuntimeEventType.CANCELLED,
            phase=ExecutionPhase.COMPLETION,
            payload={"reason": task.metadata.get("cancellation_reason", "")},
        )
        lifecycle.transition(task, TaskState.VALIDATING)
        lifecycle.transition(task, TaskState.CANCELLED)
        CancellationCoordinator.clear_checkpoint_state(task)
        CancellationCoordinator.clear(task)
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self.finalize_trace(trace_emitter, executions, task_id=task.task_id)
        early = await self.finish_task(
            task,
            trace_emitter,
            answer=self.composer.compose_summary(executions),
            executions=executions,
            validation=ValidationResult(valid=False, errors=["task_cancelled"]),
            plan=plan,
            retry_records=retry_records,
            graph_id=graph.graph_id,
        )
        return GraphPhaseOutcome(early_result=early)

    async def _handle_needs_input(
        self,
        task: Task,
        *,
        plan: NexusPlan,
        graph: ExecutionGraph,
        executions: List[AgentExecutionResult],
        retry_records: List[RetryRecord],
        lifecycle: TaskLifecycle,
        trace_emitter: TaskTraceEmitter,
    ) -> GraphPhaseOutcome:
        paused = executions[-1]
        created_at_utc = SystemTimeProvider.utc_now().isoformat()
        human_payload = (
            human_request_event_payload(
                paused.human_request,
                created_at_utc=created_at_utc,
            )
            if paused.human_request
            else {}
        )
        await self.events.publish_from_task_state(
            task,
            message="human approval requested",
            event_type=RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
            phase=ExecutionPhase.HUMAN_APPROVAL,
            payload={"human_request": human_payload},
        )
        hook_failure = await self.hitl.run_before_human_pause(
            task,
            trace_emitter,
            lifecycle,
            agent_id=paused.agent_id,
            execution=paused,
        )
        if hook_failure is not None:
            if isinstance(trace_emitter, PersistingTaskTraceEmitter):
                await self.finalize_trace(trace_emitter, executions, task_id=task.task_id)
            return GraphPhaseOutcome(early_result=hook_failure)
        HumanPauseCoordinator.apply_pause(task, paused)
        lifecycle.transition(task, TaskState.WAITING_FOR_HUMAN)
        await self.maybe_checkpoint(
            task,
            progress_message="awaiting human input",
            plan=plan,
            graph=graph,
            last_execution=paused,
        )
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self.finalize_trace(trace_emitter, executions, task_id=task.task_id)
        early = await self.finish_task(
            task,
            trace_emitter,
            answer=paused.summary,
            executions=executions,
            validation=ValidationResult(valid=False, errors=["awaiting human input"]),
            plan=plan,
            retry_records=retry_records,
            graph_id=graph.graph_id,
        )
        return GraphPhaseOutcome(early_result=early)

    async def _handle_graph_failure(
        self,
        task: Task,
        *,
        plan: NexusPlan,
        graph: ExecutionGraph,
        executions: List[AgentExecutionResult],
        retry_records: List[RetryRecord],
        failed_nodes: List[str],
        lifecycle: TaskLifecycle,
        trace_emitter: TaskTraceEmitter,
    ) -> GraphPhaseOutcome:
        lifecycle.transition(task, TaskState.VALIDATING)
        lifecycle.transition(task, TaskState.FAILED)
        if LongRunningCoordinator.is_long_running(task):
            await self.maybe_checkpoint(
                task,
                progress_message=f"graph failed at {failed_nodes}",
                plan=plan,
                graph=graph,
                last_execution=executions[-1] if executions else None,
            )
        if isinstance(trace_emitter, PersistingTaskTraceEmitter):
            await self.finalize_trace(trace_emitter, executions, task_id=task.task_id)
        early = await self.finish_task(
            task,
            trace_emitter,
            answer=self.composer.compose_summary(executions),
            executions=executions,
            validation=ValidationResult(
                valid=False,
                errors=[f"graph node failed: {failed_nodes}"],
            ),
            plan=plan,
            retry_records=retry_records,
            graph_id=graph.graph_id,
        )
        return GraphPhaseOutcome(early_result=early)


def _build_critic_trace_emitter(
    *,
    task: Task,
    trace_emitter: TaskTraceEmitter,
    hooks: CriticGraphHooks | None,
) -> CriticTraceEmitter | None:
    if hooks is None:
        return None
    if not hooks.verify_node_partial and not hooks.verify_graph_final:
        return None
    from intergrax.runtime.critic.trace import build_critic_trace_emitter

    trace_writer = (
        trace_emitter.trace_store
        if isinstance(trace_emitter, PersistingTaskTraceEmitter)
        else None
    )
    active_run_id, _ = require_active_execution_identity()
    return build_critic_trace_emitter(
        run_id=active_run_id,
        trace_writer=trace_writer,
        event_bus=trace_emitter.event_bus,
        seq_offset=len(trace_emitter.events),
    )
