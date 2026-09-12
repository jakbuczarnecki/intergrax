# © Artur Czarnecki. All rights reserved.

"""SELF-HEALING R5.1 strategy performance memory foundation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext
from intergrax.contracts.self_healing.performance_memory import (
    SelfHealingStrategyExecutionOutcome,
    SelfHealingStrategyPerformanceExperience,
    StrategyPerformanceMemoryQuery,
    StrategyPerformanceMemoryRepository,
    mint_self_healing_strategy_performance_experience_id,
)
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.contracts.self_healing.workflow.validation import ValidationResult, ValidationStatus
from intergrax.runtime.self_healing.performance_memory import (
    InMemoryStrategyPerformanceMemoryRepository,
    StrategyPerformanceMemoryRecorder,
    derive_strategy_execution_outcome,
)
from tests.unit.runtime.self_healing.test_autonomous_enterprise_self_healing_orchestration_r3_q import (
    _decision,
    _lifecycle_stack,
)
from tests.unit.runtime.self_healing.test_autonomous_enterprise_self_healing_strategy_r1_q import (
    _approved_context,
    _sample_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sample_execution_context() -> SelfHealingExecutionContext:
    return SelfHealingExecutionContext(
        workflow_id="sh_wf_testworkflow0001",
        plan_id="sh_plan_testplan00001",
        strategy_id="platform.default.restart",
        tenant_id="tenant-a",
        execution_ids=("exec-1",),
        operation_attempt_ids=("attempt-1",),
        evidence_refs=("evidence://run/1",),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _sample_outcome(**overrides: object) -> SelfHealingWorkflowOutcome:
    base = {
        "workflow_id": "sh_wf_testworkflow0001",
        "strategy_id": "platform.default.restart",
        "tenant_id": "tenant-a",
        "successful_steps": ("step-1",),
        "failed_steps": (),
        "rollback_executed": False,
        "validation_result": ValidationResult(
            status=ValidationStatus.PASSED,
            confidence=0.9,
            evidence_refs=("evidence://run/1",),
            explanation="ok",
        ),
        "recovery_time": timedelta(seconds=12.5),
        "evidence_refs": ("evidence://run/1",),
    }
    base.update(overrides)
    return SelfHealingWorkflowOutcome(**base)  # type: ignore[arg-type]


def test_repository_protocol_is_runtime_checkable() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    assert isinstance(repo, StrategyPerformanceMemoryRepository)


def test_empty_history_query_returns_empty_tuple() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    rows = repo.query(StrategyPerformanceMemoryQuery(tenant_id="tenant-a"))
    assert rows == ()


def test_append_and_query_by_strategy() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    recorded = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    experience = SelfHealingStrategyPerformanceExperience(
        experience_id=mint_self_healing_strategy_performance_experience_id(),
        tenant_id="tenant-a",
        strategy_id="platform.default.restart",
        workflow_id="sh_wf_abc",
        plan_id="sh_plan_xyz",
        execution_ids=("exec-1",),
        diagnostic_investigation_id="inv-1",
        execution_outcome=SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED,
        rollback_executed=False,
        recovery_time_seconds=3.0,
        evidence_refs=("evidence://1",),
        recorded_at=recorded,
    )
    repo.append(experience)
    by_strategy = repo.query(
        StrategyPerformanceMemoryQuery(tenant_id="tenant-a", strategy_id="platform.default.restart"),
    )
    assert by_strategy == (experience,)
    assert repo.query(StrategyPerformanceMemoryQuery(tenant_id="tenant-b")) == ()


def test_recorder_persists_workflow_facts() -> None:
    repo = InMemoryStrategyPerformanceMemoryRepository()
    recorder = StrategyPerformanceMemoryRecorder(repository=repo)
    outcome = _sample_outcome()
    exec_ctx = _sample_execution_context()
    stored = recorder.observe_workflow_completion(
        outcome,
        exec_ctx,
        diagnostic_investigation_id="inv-diag-1",
        recorded_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    assert stored.execution_outcome is SelfHealingStrategyExecutionOutcome.REPAIR_SUCCEEDED
    history = repo.query(
        StrategyPerformanceMemoryQuery(
            tenant_id="tenant-a",
            workflow_id="sh_wf_testworkflow0001",
        ),
    )
    assert history == (stored,)


def test_derive_outcome_rollback_takes_precedence() -> None:
    outcome = _sample_outcome(
        rollback_executed=True,
        validation_result=ValidationResult(
            status=ValidationStatus.PASSED,
            confidence=0.9,
            evidence_refs=("evidence://run/1",),
            explanation="ok",
        ),
    )
    assert derive_strategy_execution_outcome(outcome) is SelfHealingStrategyExecutionOutcome.ROLLED_BACK


def test_lifecycle_completion_observed_without_engine_change() -> None:
    lifecycle = _lifecycle_stack()
    repo = InMemoryStrategyPerformanceMemoryRepository()
    recorder = StrategyPerformanceMemoryRecorder(repository=repo)
    decision = _decision()
    context = _sample_context()
    exec_ctx = lifecycle.start_from_decision(decision, context)
    final_ctx, outcome = lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    investigation_id = context.diagnostic_investigation.investigation_id
    recorder.observe_workflow_completion(
        outcome,
        final_ctx,
        diagnostic_investigation_id=investigation_id,
    )
    history = repo.query(
        StrategyPerformanceMemoryQuery(
            tenant_id=context.tenant_id,
            strategy_id=decision.strategy_id,
        ),
    )
    assert len(history) == 1
    assert history[0].diagnostic_investigation_id == investigation_id
    assert history[0].execution_ids == final_ctx.execution_ids
