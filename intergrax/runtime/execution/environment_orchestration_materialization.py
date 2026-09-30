# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-owned orchestration backend materialization (EBH-4-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.host_orchestration_application_wiring_applier import (
    apply_host_orchestration_application_wiring_bundle,
)
from intergrax.runtime.execution.host_orchestration_loop_init_spec import (
    HostOrchestrationLoopInitSpec,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.observability.qualification_runtime_trace import (
    DeferredPersistedTraceFinalize,
)
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead


@dataclass(slots=True)
class EnvironmentOrchestrationMaterialization:
    """Private orchestration backend handle — Execution Engine owner zone only."""

    _backend: NexusLoop

    @property
    def event_bus(self) -> RuntimeEventBus:
        return self._backend.event_bus

    def apply_validation_engine(
        self,
        validation_engine: NexusValidationEngine | None,
    ) -> None:
        self._backend.apply_validation_engine(validation_engine)

    def set_hold_persisted_trace_finalize(self, hold: bool) -> None:
        self._backend.set_hold_persisted_trace_finalize(hold)

    def take_deferred_persisted_trace_finalize(
        self,
    ) -> DeferredPersistedTraceFinalize | None:
        return self._backend.take_deferred_persisted_trace_finalize()


def materialize_host_orchestration_backend(
    registry: AgentRegistryRead,
    spec: HostOrchestrationLoopInitSpec,
) -> EnvironmentOrchestrationMaterialization:
    """Construct the private Nexus backend inside the Execution Engine owner zone."""
    merge_strategy = spec.merge_strategy
    if merge_strategy is None:
        from intergrax.contracts.orchestration_enums import MergeStrategy

        merge_strategy = MergeStrategy.CONCAT
    backend = NexusLoop(
        registry,
        classifier=spec.classifier,
        planner=spec.planner,
        max_parallel_nodes=spec.max_parallel_nodes,
        max_inflight_nodes=spec.max_inflight_nodes,
        max_delegation_depth=spec.max_delegation_depth,
        max_run_retries=spec.max_run_retries,
        merge_strategy=merge_strategy,
        context_manager=spec.context_manager,
        event_bus=spec.event_bus,
        trace_store=spec.trace_store,
        retry_policy=spec.retry_policy,
        shadow_manager=spec.shadow_manager,
        sandbox_manager=spec.sandbox_manager,
        checkpoint_store=spec.checkpoint_store,
        agent_checkpoint_store=spec.agent_checkpoint_store,
        compensation_queue_store=spec.compensation_queue_store,
        idempotency_store=spec.idempotency_store,
        declarative_tool_invoker=spec.declarative_tool_invoker,
        notification_adapter=spec.notification_adapter,
        runtime_events_db_path=spec.runtime_events_db_path,
        task_memory_store=spec.task_memory_store,
        task_memory_db_path=spec.task_memory_db_path,
        production_mode=spec.production_mode,
        signal_collector=spec.signal_collector,
        run_budget=spec.run_budget,
        decision_flow_gate=spec.decision_flow_gate,
        emit_coordination_advisory=spec.emit_coordination_advisory,
        allow_dynamic_replan=spec.allow_dynamic_replan,
        denied_planner_model_ids=spec.denied_planner_model_ids,
        planner_model_id=spec.planner_model_id,
        validation_engine=spec.validation_engine,
        authority_policy=spec.authority_policy,
        budget_allocation_policy=spec.budget_allocation_policy,
        execution_budget_ledger_factory=spec.execution_budget_ledger_factory,
        attempt_lifecycle=spec.attempt_lifecycle,
        execution_terminal=spec.execution_terminal,
        execution_lineage_persistence=spec.execution_lineage_persistence,
        execution_continuation_state_store=spec.execution_continuation_state_store,
        governance_evidence_recorder=spec.governance_evidence_recorder,
    )
    if spec.application_wiring is not None:
        apply_host_orchestration_application_wiring_bundle(backend, spec.application_wiring)
    return EnvironmentOrchestrationMaterialization(_backend=backend)


__all__ = [
    "EnvironmentOrchestrationMaterialization",
    "materialize_host_orchestration_backend",
]
