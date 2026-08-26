# © Artur Czarnecki. All rights reserved.

"""Product observability dashboard central diagnostic read wiring (ONE-SPINE-1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.applications._shared.product_observability_dashboard_wiring import (
    _build_diagnostic_operations_pane,
    resolve_product_observability_dashboard_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import ObservabilityProfile
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.deterministic_problem_grouping import STRATEGY_ID
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingEngine
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingStrategyRegistry
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine, ProblemStatus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TENANT = "product-host"
_OBSERVED_AT = datetime(2026, 8, 26, 9, 0, tzinfo=UTC)


def _product_env(*, diagnostics_pane_enabled: bool = True) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id=_TENANT)
    return env.model_copy(
        update={
            "observability_profile": env.observability_profile.model_copy(
                update={"diagnostics_pane_enabled": diagnostics_pane_enabled},
            ),
        },
    )


def _grouping_engine() -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(registry)


def _assess_retry_pair(
    *,
    tenant_id: str,
    runtime_store: InMemoryRuntimeEventStore,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
) -> tuple[object, object]:
    from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
    from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
    from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
    from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingAssessmentInput

    sequence = [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
        violating_event_type,
    ]

    def assess_once() -> ProblemGroupingAssessmentInput:
        task_id = mint_task_id()
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        for event_type in sequence:
            event = sample_runtime_event(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ).model_copy(update={"event_type": event_type})
            runtime_store.append(event, tenant_id=tenant_id)
        reconstruction = ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ).reconstruct_execution(tenant_id, task_id, run_id)
        lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
        assessment = DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)
        return ProblemGroupingAssessmentInput(assessment=assessment)

    return assess_once(), assess_once()


def _read_service_with_problems(
    *,
    tenant_id: str,
    open_count: int,
    resolved_count: int,
) -> DiagnosticReadService:
    persistence = InMemoryProblemPersistence()
    runtime_store = InMemoryRuntimeEventStore()
    lifecycle = ProblemLifecycleEngine(persistence)
    grouping_engine = _grouping_engine()

    open_violations = [
        RuntimeEventType.RETRY_SCHEDULED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.PAUSE_REQUESTED,
    ]
    for index in range(open_count):
        grouping = grouping_engine.group(
            _assess_retry_pair(
                tenant_id=tenant_id,
                runtime_store=runtime_store,
                violating_event_type=open_violations[index % len(open_violations)],
            ),
            strategy_id=STRATEGY_ID,
        )
        lifecycle.reconcile(
            grouping,
            observed_at=_OBSERVED_AT + timedelta(minutes=index),
        )

    resolved_violations = [
        RuntimeEventType.PAUSE_REQUESTED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.RETRY_SCHEDULED,
    ]
    for index in range(resolved_count):
        grouping = grouping_engine.group(
            _assess_retry_pair(
                tenant_id=tenant_id,
                runtime_store=runtime_store,
                violating_event_type=resolved_violations[index % len(resolved_violations)],
            ),
            strategy_id=STRATEGY_ID,
        )
        result = lifecycle.reconcile(
            grouping,
            observed_at=_OBSERVED_AT + timedelta(hours=1, minutes=index),
        )
        problem = result.created[0] if result.created else result.updated[0]
        lifecycle.resolve(
            tenant_id=tenant_id,
            problem_id=problem.problem_id,
            resolved_at=_OBSERVED_AT + timedelta(hours=2, minutes=index),
        )

    return DiagnosticReadService(
        problem_persistence=persistence,
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=runtime_store,
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )


def test_diagnostic_pane_unavailable_when_read_service_not_injected() -> None:
    env = _product_env()
    pane = _build_diagnostic_operations_pane(env, None)
    assert pane.ready is False
    assert pane.problem_count == 0
    assert pane.open_problem_count == 0


def test_diagnostic_pane_disabled_when_flag_off() -> None:
    env = _product_env(diagnostics_pane_enabled=False)
    service = _read_service_with_problems(tenant_id=_TENANT, open_count=1, resolved_count=0)
    pane = _build_diagnostic_operations_pane(env, service)
    assert pane.ready is False
    assert pane.problem_count == 0
    assert pane.open_problem_count == 0


def test_diagnostic_pane_empty_ready_when_service_wired() -> None:
    env = _product_env()
    service = DiagnosticReadService(
        problem_persistence=InMemoryProblemPersistence(),
        execution_reconstructor=ExecutionReconstructor(
            runtime_events=InMemoryRuntimeEventStore(),
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )
    pane = _build_diagnostic_operations_pane(env, service)
    assert pane.ready is True
    assert pane.problem_count == 0
    assert pane.open_problem_count == 0


def test_diagnostic_pane_counts_from_central_read_service() -> None:
    env = _product_env()
    service = _read_service_with_problems(tenant_id=_TENANT, open_count=2, resolved_count=1)
    pane = _build_diagnostic_operations_pane(env, service)
    assert pane.ready is True
    assert pane.problem_count == 3
    assert pane.open_problem_count == 2


def test_diagnostic_pane_uses_host_tenant_scope() -> None:
    env = _product_env()
    other_tenant_service = _read_service_with_problems(
        tenant_id="other-tenant",
        open_count=3,
        resolved_count=0,
    )
    pane = _build_diagnostic_operations_pane(env, other_tenant_service)
    assert pane.ready is True
    assert pane.problem_count == 0
    assert pane.open_problem_count == 0


def test_dashboard_wiring_exposes_diagnostics_pane_not_legacy_causal() -> None:
    env = _product_env()
    service = _read_service_with_problems(tenant_id=_TENANT, open_count=1, resolved_count=0)
    wiring = resolve_product_observability_dashboard_wiring(
        env,
        repo_root=_REPO_ROOT,
        diagnostic_read_service=service,
    )
    assert wiring.enabled is True
    assert wiring.dashboard is not None
    dashboard = wiring.dashboard
    assert hasattr(dashboard, "diagnostics")
    assert not hasattr(dashboard, "causal")
    assert dashboard.diagnostics.ready is True
    assert dashboard.diagnostics.problem_count == 1
    assert dashboard.diagnostics.open_problem_count == 1
