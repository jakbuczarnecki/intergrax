# © Artur Czarnecki. All rights reserved.

"""Harness for DIAG R3 multi-agent failure localization qualification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import get_catalog_entry
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt

_TENANT = "tenant-r3-localization"


@dataclass(slots=True)
class MultiAgentFailureLocalizationScenario:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    lineage: InMemoryExecutionLineagePersistence | None
    runtime_store: InMemoryRuntimeEventStore
    reconstructor: ExecutionReconstructor
    assessment_builder: DiagnosticAssessmentBuilder
    lifecycle_analyzer: LifecycleAnomalyAnalyzer

    @property
    def scope(self) -> ExecutionLineageAttemptScope:
        return build_execution_lineage_attempt_scope(
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
        )

    def assess(self):
        reconstruction = self.reconstructor.reconstruct_execution(
            self.tenant_id,
            self.task_id,
            self.run_id,
        )
        lifecycle = self.lifecycle_analyzer.analyze(reconstruction)
        return self.assessment_builder.assess(reconstruction, lifecycle)


def build_multi_agent_failure_localization_scenario(
    *,
    with_lineage: bool = True,
) -> MultiAgentFailureLocalizationScenario:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    lineage = InMemoryExecutionLineagePersistence() if with_lineage else None
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT,
            source=MessageBusTaskRef(provider="celery", task_id="t-r3", tenant_id=_TENANT),
            target=RuntimeExecutionRef(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                tenant_id=_TENANT,
            ),
            recorded_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
        ),
    )
    runtime_store.append(
        sample_runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        tenant_id=_TENANT,
    )
    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
        execution_lineage=lineage,
        max_lineage_records=10_000,
    )
    return MultiAgentFailureLocalizationScenario(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        lineage=lineage,
        runtime_store=runtime_store,
        reconstructor=reconstructor,
        assessment_builder=DiagnosticAssessmentBuilder(),
        lifecycle_analyzer=LifecycleAnomalyAnalyzer(),
    )


def open_lineage_chain(
    scenario: MultiAgentFailureLocalizationScenario,
    execution_ids: tuple[ExecutionId, ...],
) -> None:
    """Register a linear parent chain E1 -> E2 -> ... under one segment."""
    if not execution_ids:
        raise ValueError("execution_ids must not be empty")
    register_v1_attempt(scenario.lineage, scenario.scope)
    root = execution_ids[0]
    scenario.lineage.open_segment(scenario.scope, root)
    scenario.lineage.admit_root(scenario.scope, root, root)
    for index in range(1, len(execution_ids)):
        child = execution_ids[index]
        parent = execution_ids[index - 1]
        scenario.lineage.admit_child(scenario.scope, root, child, parent)


def open_lineage_fan_out(
    scenario: MultiAgentFailureLocalizationScenario,
    root: ExecutionId,
    children: tuple[tuple[ExecutionId, ExecutionId], ...],
) -> None:
    """Register root and child admissions (child_id, parent_id)."""
    register_v1_attempt(scenario.lineage, scenario.scope)
    scenario.lineage.open_segment(scenario.scope, root)
    scenario.lineage.admit_root(scenario.scope, root, root)
    for child_id, parent_id in children:
        scenario.lineage.admit_child(scenario.scope, root, child_id, parent_id)


def record_execution_failed(
    scenario: MultiAgentFailureLocalizationScenario,
    execution_id: ExecutionId,
) -> None:
    catalog_entry = get_catalog_entry(RuntimeEventType.EXECUTION_FAILED)
    if catalog_entry is None:
        raise RuntimeError("EXECUTION_FAILED missing event catalog entry")
    from intergrax.contracts.event_severity import EventSeverity

    event = RuntimeEvent(
        tenant_id=scenario.tenant_id,
        task_id=scenario.task_id,
        run_id=scenario.run_id,
        attempt_id=scenario.attempt_id,
        execution_id=execution_id,
        event_type=RuntimeEventType.EXECUTION_FAILED,
        phase=catalog_entry.phase,
        severity=EventSeverity.ERROR,
        payload={},
    )
    payload = ExecutionFailurePayloadV1(
        failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
        safe_summary="Execution delegate failed",
    )
    event = runtime_event_with_payload(event, payload)
    scenario.runtime_store.append(event, tenant_id=scenario.tenant_id)


__all__ = [
    "MultiAgentFailureLocalizationScenario",
    "build_multi_agent_failure_localization_scenario",
    "open_lineage_chain",
    "open_lineage_fan_out",
    "record_execution_failed",
]
