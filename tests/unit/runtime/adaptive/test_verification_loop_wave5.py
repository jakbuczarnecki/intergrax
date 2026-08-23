# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-5: VerificationLoop, auto-rollback, L4 evidence, and scheduler tests."""

from __future__ import annotations

import time

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.adaptation_scheduler import AdaptationScheduler
from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptive_runtime_events import (
    build_adaptive_loop_blocked_event,
    build_adaptive_verification_failed_event,
)
from intergrax.runtime.adaptive.bandit_state_store import InMemoryBanditStateStore
from intergrax.runtime.adaptive.contracts import (
    HarnessOutcomeSignal,
    OutcomeEvalMode,
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionStatus,
)
from intergrax.runtime.adaptive.governance_pipeline import AdaptationGovernancePipeline
from intergrax.runtime.adaptive.l4_runtime_evidence import (
    build_harness_baseline_l4_evidence,
    build_harness_baseline_signals,
    build_l4_runtime_evidence_from_signals,
)
from intergrax.runtime.adaptive.loop_apply_block_store import InMemoryLoopApplyBlockStore
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
from intergrax.runtime.adaptive.profile_mutation_store import InMemoryAdaptiveProfileMutationStore
from intergrax.runtime.adaptive.profile_pointer_store import InMemoryProfileActivePointerStore
from intergrax.runtime.adaptive.profile_version_store import InMemoryProfileVersionStore
from intergrax.runtime.adaptive.proposal_builder import ProposalBuilder
from intergrax.runtime.adaptive.proposal_cooldown_store import InMemoryProposalCooldownStore
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.adaptive.verification_checks import (
    check_eval_registry_trend,
    check_regression_rate,
    check_utility_trend,
)
from intergrax.runtime.adaptive.verification_loop import VerificationLoop
from intergrax.runtime.adaptive.verification_models import VerificationContext, VerificationTarget
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge
from intergrax.runtime.architecture.cost_budget import BudgetEnvelope, BudgetScope
from intergrax.runtime.architecture.evaluation_automation import (
    AutomatedEvaluationRecord,
    AutomatedEvaluationReport,
    AutomatedEvaluatorResult,
    EvaluatorType,
)
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationReleaseSnapshot,
    build_evaluation_registry_trend_report,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _signal(
    *,
    run_id: str,
    task_class: str,
    utility: float,
    eval_mode: OutcomeEvalMode,
    regression_flags: list[str] | None = None,
) -> HarnessOutcomeSignal:
    return HarnessOutcomeSignal(
        run_id=run_id,
        tenant_id="tenant-a",
        application_id="lab_application",
        agent_id="agent:echo",
        task_class=task_class,
        eval_mode=eval_mode,
        quality_score=utility,
        utility=utility,
        regression_flags=regression_flags or [],
    )


def _automated_report(*, passed: int, failed: int) -> AutomatedEvaluationReport:
    records: list[AutomatedEvaluationRecord] = []
    for index in range(passed):
        records.append(
            AutomatedEvaluationRecord(
                run_id=f"pass-{index}",
                target_id="agent:echo",
                mode="offline",
                rule_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.RULE_BASED,
                    passed=True,
                    score=1.0,
                ),
                llm_judge_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.LLM_JUDGE,
                    passed=True,
                    score=0.9,
                ),
                final_passed=True,
            )
        )
    for index in range(failed):
        records.append(
            AutomatedEvaluationRecord(
                run_id=f"fail-{index}",
                target_id="agent:echo",
                mode="offline",
                rule_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.RULE_BASED,
                    passed=False,
                    score=0.0,
                ),
                llm_judge_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.LLM_JUDGE,
                    passed=False,
                    score=0.0,
                ),
                final_passed=False,
            )
        )
    return AutomatedEvaluationReport(records=records)


def _build_scheduler_engine() -> AdaptationEngine:
    bandit_store = InMemoryBanditStateStore()
    governance = AdaptationGovernancePipeline()
    builder = ProposalBuilder(governance)
    return AdaptationEngine(
        sub_engines=[],
        proposal_builder=builder,
        bandit_store=bandit_store,
        cooldown_store=InMemoryProposalCooldownStore(),
    )


class _AlwaysPassSecurityChecker:
    def evaluate(self) -> bool:
        return True


class _AllowControlPlaneEvaluator:
    def evaluate(self, request) -> PolicyDecision:
        return PolicyDecision(action=PolicyAction.ALLOW, reason="ok")


def test_utility_trend_passes_when_candidate_beats_baseline() -> None:
    context = VerificationContext(min_improvement_delta=0.05, min_run_count=3)
    candidate = [_signal(run_id=f"c-{i}", task_class="golden-echo", utility=0.85, eval_mode=OutcomeEvalMode.SHADOW) for i in range(4)]
    baseline = [_signal(run_id=f"b-{i}", task_class="golden-echo", utility=0.70, eval_mode=OutcomeEvalMode.OFFLINE) for i in range(4)]
    result = check_utility_trend(
        candidate_signals=candidate,
        baseline_signals=baseline,
        context=context,
    )
    assert result.passed is True


def test_eval_registry_trend_uses_release_comparison() -> None:
    trend = build_evaluation_registry_trend_report(
        snapshots=[
            EvaluationReleaseSnapshot(
                release_id="2026.05",
                automated_report=_automated_report(passed=8, failed=2),
            ),
            EvaluationReleaseSnapshot(
                release_id="2026.06",
                automated_report=_automated_report(passed=9, failed=1),
            ),
        ]
    )
    result = check_eval_registry_trend(evaluation_trend=trend, context=VerificationContext())
    assert result.passed is True
    assert result.metric_value == pytest.approx(0.9)


def test_regression_rate_fails_on_spike() -> None:
    context = VerificationContext(max_regression_rate_delta=0.05)
    candidate = [
        _signal(
            run_id=f"c-{i}",
            task_class="golden-echo",
            utility=0.5,
            eval_mode=OutcomeEvalMode.SHADOW,
            regression_flags=["step_explosion"],
        )
        for i in range(4)
    ]
    baseline = [
        _signal(run_id=f"b-{i}", task_class="golden-echo", utility=0.7, eval_mode=OutcomeEvalMode.OFFLINE)
        for i in range(4)
    ]
    result = check_regression_rate(
        candidate_signals=candidate,
        baseline_signals=baseline,
        context=context,
    )
    assert result.passed is False


def test_verification_loop_auto_rollback_and_block_on_failure() -> None:
    signal_store = InMemorySignalStore()
    for signal in [
        _signal(run_id="c-1", task_class="echo", utility=0.40, eval_mode=OutcomeEvalMode.SHADOW),
        _signal(run_id="c-2", task_class="echo", utility=0.35, eval_mode=OutcomeEvalMode.SHADOW),
        _signal(run_id="c-3", task_class="echo", utility=0.30, eval_mode=OutcomeEvalMode.SHADOW),
        _signal(run_id="b-1", task_class="echo", utility=0.80, eval_mode=OutcomeEvalMode.OFFLINE),
        _signal(run_id="b-2", task_class="echo", utility=0.78, eval_mode=OutcomeEvalMode.OFFLINE),
        _signal(run_id="b-3", task_class="echo", utility=0.79, eval_mode=OutcomeEvalMode.OFFLINE),
    ]:
        signal_store.append(signal)

    profile_store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store=profile_store)
    profile_store.create_from_draft(
        ProfileVersionDraft(
            version_id="baseline-v1",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "standard"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id="tenant-a",
        task_class="echo",
    )
    profile_store.create_from_draft(
        ProfileVersionDraft(
            version_id="candidate-v2",
            artifact_type=ProfileArtifactType.RAG,
            artifact_payload={"tier": "deep"},
            parent_version_id="baseline-v1",
            status=ProfileVersionStatus.CANARY,
        ),
        tenant_id="tenant-a",
        task_class="echo",
    )
    pointer_store.swap_active(
        tenant_id="tenant-a",
        task_class="echo",
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="baseline-v1",
        expected_active_version_id=None,
    )
    pointer_store.swap_active(
        tenant_id="tenant-a",
        task_class="echo",
        artifact_type=ProfileArtifactType.RAG,
        new_active_version_id="candidate-v2",
        expected_active_version_id="baseline-v1",
    )
    lifecycle.transition("baseline-v1", target=ProfileVersionStatus.RETIRED)
    lifecycle.transition("candidate-v2", target=ProfileVersionStatus.ACTIVE)
    block_store = InMemoryLoopApplyBlockStore()
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=_AllowControlPlaneEvaluator(),
    )
    governance_bridge = RuntimeArchitectureGovernanceBridge(
        mutation_authorization_boundary=boundary,
    )
    mutation_store = InMemoryAdaptiveProfileMutationStore(
        version_store=profile_store,
        pointer_store=pointer_store,
    )
    executor = AdaptationExecutor(
        profile_store=profile_store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
        mutation_store=mutation_store,
    )
    loop = VerificationLoop(
        signal_store=signal_store,
        profile_store=profile_store,
        executor=executor,
        governance_bridge=governance_bridge,
        pointer_store=pointer_store,
        block_store=block_store,
        security_checker=_AlwaysPassSecurityChecker(),
    )
    target = VerificationTarget(
        tenant_id="tenant-a",
        task_class="echo",
        artifact_type=ProfileArtifactType.RAG,
        candidate_version_id="candidate-v2",
        loop_kind=AdaptiveLoopKind.ROUTING_TUNING,
    )
    result = loop.verify_target(
        target,
        context=VerificationContext(
            min_run_count=3,
            auto_rollback_enabled=True,
            auto_rollback_service_principal=RequestIdentity(
                tenant_id="tenant-a",
                user_id="verification-scheduler",
                principal_type=PrincipalType.SERVICE,
                auth_subject="verification-scheduler",
            ),
            auto_rollback_mutation_id="mut-auto-rollback-1",
        ),
    )
    assert result.passed is False
    assert result.rolled_back is True
    pointer = pointer_store.get_pointer(
        tenant_id="tenant-a",
        task_class="echo",
        artifact_type=ProfileArtifactType.RAG,
    )
    assert pointer is not None
    assert pointer.active_version_id == "baseline-v1"
    assert block_store.is_blocked(AdaptiveLoopKind.ROUTING_TUNING, tenant_id="tenant-a")


def test_l4_runtime_evidence_from_signals_requires_three_golden_scenarios() -> None:
    store = InMemorySignalStore()
    for signal in build_harness_baseline_signals():
        store.append(signal)
    evidence = build_l4_runtime_evidence_from_signals(store)
    assert evidence.scenarios_passed_count == 3
    assert evidence.runtime_l4_closed_loop_passed is True


def test_harness_baseline_l4_evidence_passes_closeout_thresholds() -> None:
    evidence = build_harness_baseline_l4_evidence()
    assert evidence.runtime_l4_closed_loop_passed is True
    assert evidence.apply_rollback_rate < 0.10


def test_scheduler_run_verification_loop_requires_configuration() -> None:
    engine = _build_scheduler_engine()
    scheduler = AdaptationScheduler(engine=engine, signal_store=InMemorySignalStore())
    with pytest.raises(ValueError, match="VerificationLoop"):
        scheduler.run_verification_loop(context=VerificationContext())


def test_scheduler_run_verification_loop_delegates_to_verification_loop() -> None:
    signal_store = InMemorySignalStore()
    profile_store = InMemoryProfileVersionStore()
    loop = VerificationLoop(signal_store=signal_store, profile_store=profile_store)
    engine = _build_scheduler_engine()
    scheduler = AdaptationScheduler(
        engine=engine,
        signal_store=signal_store,
        verification_loop=loop,
    )
    report = scheduler.run_verification_loop(context=VerificationContext())
    assert report.passed is True


def test_adaptive_verification_runtime_events() -> None:
    token = bind_active_execution_identity(
        run_id="run_0123456789abcdef0123456789abcdef",
        attempt_id="attempt_0123456789abcdef0123456789abcdef",
    )
    try:
        failed = build_adaptive_verification_failed_event(
            task_id="task_0123456789abcdef0123456789abcdef",
            run_id="run_0123456789abcdef0123456789abcdef",
            tenant_id="tenant-a",
            candidate_version_id="candidate-v2",
            failure_reasons=["utility_trend"],
        )
        blocked = build_adaptive_loop_blocked_event(
            task_id="task_0123456789abcdef0123456789abcdef",
            run_id="run_0123456789abcdef0123456789abcdef",
            tenant_id="tenant-a",
            loop_kind="routing_tuning",
            reason="verification_failed",
        )
        assert failed.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert failed.event_kind == "platform.adaptive.adaptive_verification_failed"
        assert blocked.event_type == RuntimeEventType.DOMAIN_SIGNAL
        assert blocked.event_kind == "platform.adaptive.adaptive_loop_blocked"
    finally:
        reset_active_execution_identity(token)


def test_cost_budget_check_uses_budget_envelopes() -> None:
    from intergrax.runtime.adaptive.verification_checks import check_cost_budget

    result = check_cost_budget(
        candidate_signals=[],
        budget_envelopes=[
            BudgetEnvelope(
                scope=BudgetScope.TENANT,
                scope_id="tenant-a",
                limit_amount=100.0,
                spent_amount=150.0,
            )
        ],
        context=VerificationContext(),
    )
    assert result.passed is False


def test_rollback_drill_completes_under_five_minutes() -> None:
    """W-ADAPT-5.10: rollback drill smoke — apply then verify-fail rollback."""
    started = time.monotonic()
    signal_store = InMemorySignalStore()
    profile_store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store=profile_store)
    profile_store.create_from_draft(
        ProfileVersionDraft(
            version_id="baseline-drill",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
            artifact_payload={"mode": "baseline"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id="tenant-drill",
        task_class="drill",
    )
    profile_store.create_from_draft(
        ProfileVersionDraft(
            version_id="candidate-drill",
            artifact_type=ProfileArtifactType.ORCHESTRATION,
            artifact_payload={"mode": "candidate"},
            status=ProfileVersionStatus.ACTIVE,
        ),
        tenant_id="tenant-drill",
        task_class="drill",
    )
    pointer_store.swap_active(
        tenant_id="tenant-drill",
        task_class="drill",
        artifact_type=ProfileArtifactType.ORCHESTRATION,
        new_active_version_id="baseline-drill",
        expected_active_version_id=None,
    )
    pointer_store.swap_active(
        tenant_id="tenant-drill",
        task_class="drill",
        artifact_type=ProfileArtifactType.ORCHESTRATION,
        new_active_version_id="candidate-drill",
        expected_active_version_id="baseline-drill",
    )
    lifecycle.transition("baseline-drill", target=ProfileVersionStatus.RETIRED)
    mutation_store = InMemoryAdaptiveProfileMutationStore(
        version_store=profile_store,
        pointer_store=pointer_store,
    )
    executor = AdaptationExecutor(
        profile_store=profile_store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
        mutation_store=mutation_store,
    )
    rollback = executor.rollback(
        tenant_id="tenant-drill",
        task_class="drill",
        artifact_type=ProfileArtifactType.ORCHESTRATION,
        expected_active_version_id="candidate-drill",
    )
    elapsed_seconds = time.monotonic() - started
    assert rollback.restored_version_id == "baseline-drill"
    assert elapsed_seconds < 300.0
