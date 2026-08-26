# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import random

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticAssessmentBuilder,
    DiagnosticFindingKind,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    STRATEGY_ID,
    STRATEGY_VERSION,
    DeterministicProblemGroupingStrategy,
    build_deterministic_problem_signature,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnomalyAnalyzer,
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
    LifecycleViolationTransition,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemGroupingBasis,
    ProblemGroupingAssessmentInput,
    ProblemGroupingBasisKind,
    ProblemGroupingEngine,
    ProblemGroupingInput,
    ProblemGroupingMethod,
    ProblemGroupingStrategyRegistry,
    ProblemGroupingSubject,
    ProblemGroupingSubjectFinding,
    ProblemGroupingSubjectLimitation,
    ProblemGroupingSubjectRef,
)
from intergrax.runtime.diagnostics.problem_grouping_features import (
    REPRESENTATION_VERSION_V1,
    ProblemGroupingRepresentationVersion,
    project_assessment_features,
)
from intergrax.runtime.events.asof_projection import (
    RunExecutionLifecycleStatus,
    RunLifecycleViolationKind,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"


def _assessment_input(assessment: DiagnosticAssessment) -> ProblemGroupingAssessmentInput:
    return ProblemGroupingAssessmentInput(assessment=assessment)


def _subject_ref(
    *,
    tenant_id: str = _TENANT_A,
    task_id=None,
    run_id=None,
) -> ProblemGroupingSubjectRef:
    return ProblemGroupingSubjectRef(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
    )


def _lifecycle_transition(
    *,
    prior_status: RunExecutionLifecycleStatus,
    violating_event_type: RuntimeEventType,
    violation_kind: RunLifecycleViolationKind = RunLifecycleViolationKind.EVENT_AFTER_TERMINAL,
) -> LifecycleViolationTransition:
    return LifecycleViolationTransition(
        violation_kind=violation_kind,
        prior_status=prior_status,
        violating_event_type=violating_event_type,
    )


def _event_after_terminal_finding(
    *,
    scope: LifecycleAnomalyScope = LifecycleAnomalyScope.EXECUTION,
    prior_status: RunExecutionLifecycleStatus = RunExecutionLifecycleStatus.COMPLETED,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
) -> ProblemGroupingSubjectFinding:
    return ProblemGroupingSubjectFinding(
        kind=DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
        scope=scope,
        source_anomaly_kind=LifecycleAnomalyKind.EVENT_AFTER_TERMINAL,
        lifecycle_transition=_lifecycle_transition(
            prior_status=prior_status,
            violating_event_type=violating_event_type,
        ),
    )


def _subject(
    *,
    ref: ProblemGroupingSubjectRef | None = None,
    findings: tuple[ProblemGroupingSubjectFinding, ...] = (),
    limitations: tuple[ProblemGroupingSubjectLimitation, ...] = (),
) -> ProblemGroupingSubject:
    resolved_ref = ref or _subject_ref()
    return ProblemGroupingSubject(
        tenant_id=resolved_ref.tenant_id,
        task_id=resolved_ref.task_id,
        run_id=resolved_ref.run_id,
        findings=findings,
        limitations=limitations,
    )


def _truncation_limitation() -> ProblemGroupingSubjectLimitation:
    return ProblemGroupingSubjectLimitation(
        kind=DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED,
        source_anomaly_kind=LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED,
    )


def _input(subject: ProblemGroupingSubject) -> ProblemGroupingInput:
    return ProblemGroupingInput(subject=subject)


class _AssessmentFeatureProjector:
    @property
    def representation_version(self) -> ProblemGroupingRepresentationVersion:
        return REPRESENTATION_VERSION_V1

    def project(
        self,
        assessment,
        subject: ProblemGroupingSubject,
        *,
        source_facts=None,
    ):
        return project_assessment_features(assessment, subject=subject)


def _engine_with_deterministic_strategy(
    *,
    feature_projector: _AssessmentFeatureProjector | None = None,
) -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(DeterministicProblemGroupingStrategy())
    return ProblemGroupingEngine(registry, feature_projector=feature_projector)


def _assess_attempt_sequence(event_types: list[RuntimeEventType]) -> DiagnosticAssessment:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for event_type in event_types:
        event = sample_runtime_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ).model_copy(update={"event_type": event_type})
        runtime_store.append(event, tenant_id=_TENANT_A)

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(_TENANT_A, task_id, run_id)
    lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
    return DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)


def _assessment_with_truncation() -> DiagnosticAssessment:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for _ in range(5):
        event = sample_runtime_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        runtime_store.append(event, tenant_id=_TENANT_A)

    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        initial_limit=2,
        max_limit=2,
    )
    lifecycle = LifecycleAnomalyAnalyzer().analyze(reconstruction)
    return DiagnosticAssessmentBuilder().assess(reconstruction, lifecycle)


def test_strategy_metadata() -> None:
    strategy = DeterministicProblemGroupingStrategy()

    assert strategy.strategy_id == STRATEGY_ID
    assert strategy.strategy_version == STRATEGY_VERSION
    assert strategy.characteristics.method is ProblemGroupingMethod.DETERMINISTIC
    assert strategy.characteristics.deterministic is True
    assert strategy.characteristics.requires_features is False


def test_engine_integration_groups_same_structure() -> None:
    assessment_a = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    assessment_b = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    engine = _engine_with_deterministic_strategy()

    result = engine.group((_assessment_input(assessment_a), _assessment_input(assessment_b)), strategy_id=STRATEGY_ID)

    assert len(result.candidates) == 1
    assert len(result.candidates[0].members) == 2
    assert result.ungrouped_subjects == ()


def test_primary_collision_not_grouped() -> None:
    assessment_a = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    assessment_b = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.TASK_FAILED,
        ]
    )
    engine = _engine_with_deterministic_strategy()

    result = engine.group((_assessment_input(assessment_a), _assessment_input(assessment_b)), strategy_id=STRATEGY_ID)

    assert result.candidates == ()
    assert len(result.ungrouped_subjects) == 2


def test_prior_status_difference_not_grouped() -> None:
    ref_a = _subject_ref()
    ref_b = _subject_ref()
    subject_a = _subject(
        ref=ref_a,
        findings=(
            _event_after_terminal_finding(
                prior_status=RunExecutionLifecycleStatus.COMPLETED,
                violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
            ),
        ),
    )
    subject_b = _subject(
        ref=ref_b,
        findings=(
            _event_after_terminal_finding(
                prior_status=RunExecutionLifecycleStatus.CANCELLED,
                violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
            ),
        ),
    )
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert result.candidates == ()


def test_finding_order_independent() -> None:
    finding_x = _event_after_terminal_finding(
        prior_status=RunExecutionLifecycleStatus.COMPLETED,
        violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
    )
    finding_y = ProblemGroupingSubjectFinding(
        kind=DiagnosticFindingKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
        scope=LifecycleAnomalyScope.ATTEMPT,
        source_anomaly_kind=LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
        lifecycle_transition=None,
    )
    subject_a = _subject(findings=(finding_x, finding_y))
    subject_b = _subject(findings=(finding_y, finding_x))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert len(result.candidates) == 1
    assert result.candidates[0].provenance.basis is not None
    basis = result.candidates[0].provenance.basis
    assert isinstance(basis, DeterministicProblemGroupingBasis)
    assert basis.signature == build_deterministic_problem_signature(subject_a)


def test_multiplicity_preserved_not_grouped() -> None:
    finding = _event_after_terminal_finding()
    subject_a = _subject(findings=(finding,))
    subject_b = _subject(findings=(finding, finding))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert result.candidates == ()


def test_scope_difference_not_grouped() -> None:
    subject_a = _subject(
        findings=(
            _event_after_terminal_finding(scope=LifecycleAnomalyScope.ATTEMPT),
        ),
    )
    subject_b = _subject(
        findings=(
            _event_after_terminal_finding(scope=LifecycleAnomalyScope.EXECUTION),
        ),
    )
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert result.candidates == ()


def test_limitation_difference_not_grouped() -> None:
    finding = _event_after_terminal_finding()
    subject_a = _subject(findings=(finding,))
    subject_b = _subject(findings=(finding,), limitations=(_truncation_limitation(),))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert result.candidates == ()


def test_empty_subjects_not_grouped() -> None:
    subject_a = _subject()
    subject_b = _subject()
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert result.candidates == ()


def test_limitation_only_subjects_not_grouped() -> None:
    limitation = _truncation_limitation()
    subject_a = _subject(limitations=(limitation,))
    subject_b = _subject(limitations=(limitation,))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert result.candidates == ()


def test_three_member_group() -> None:
    finding = _event_after_terminal_finding()
    subjects = tuple(_subject(findings=(finding,)) for _ in range(3))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group(tuple(_input(subject) for subject in subjects))

    assert len(result.candidates) == 1
    assert len(result.candidates[0].members) == 3


def test_two_groups_plus_singleton() -> None:
    finding_x = _event_after_terminal_finding(
        prior_status=RunExecutionLifecycleStatus.COMPLETED,
        violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
    )
    finding_y = _event_after_terminal_finding(
        prior_status=RunExecutionLifecycleStatus.COMPLETED,
        violating_event_type=RuntimeEventType.TASK_FAILED,
    )
    finding_z = ProblemGroupingSubjectFinding(
        kind=DiagnosticFindingKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
        scope=LifecycleAnomalyScope.ATTEMPT,
        source_anomaly_kind=LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
        lifecycle_transition=None,
    )
    subject_a = _subject(findings=(finding_x,))
    subject_b = _subject(findings=(finding_x,))
    subject_c = _subject(findings=(finding_y,))
    subject_d = _subject(findings=(finding_y,))
    subject_e = _subject(findings=(finding_z,))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group(
        (
            _input(subject_a),
            _input(subject_b),
            _input(subject_c),
            _input(subject_d),
            _input(subject_e),
        )
    )

    assert len(result.candidates) == 2
    member_sets = {frozenset(candidate.members) for candidate in result.candidates}
    assert member_sets == {
        frozenset({subject_a.ref, subject_b.ref}),
        frozenset({subject_c.ref, subject_d.ref}),
    }
    grouped_refs = {ref for candidate in result.candidates for ref in candidate.members}
    assert subject_e.ref not in grouped_refs


def test_basis_and_supporting_refs() -> None:
    finding = _event_after_terminal_finding()
    subject_a = _subject(findings=(finding,))
    subject_b = _subject(findings=(finding,))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))
    candidate = result.candidates[0]
    basis = candidate.provenance.basis

    assert basis is not None
    assert isinstance(basis, DeterministicProblemGroupingBasis)
    assert basis.kind is ProblemGroupingBasisKind.DETERMINISTIC
    assert basis.signature == build_deterministic_problem_signature(subject_a)
    assert candidate.provenance.supporting_subject_refs == candidate.members


def test_determinism_same_input() -> None:
    assessment_a = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    assessment_b = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    engine = _engine_with_deterministic_strategy()

    first = engine.group((_assessment_input(assessment_a), _assessment_input(assessment_b)), strategy_id=STRATEGY_ID)
    second = engine.group((_assessment_input(assessment_a), _assessment_input(assessment_b)), strategy_id=STRATEGY_ID)

    assert first == second


def test_input_shuffle_preserves_group_membership() -> None:
    finding = _event_after_terminal_finding()
    subjects = tuple(_subject(findings=(finding,)) for _ in range(4))
    strategy = DeterministicProblemGroupingStrategy()
    shuffled = list(subjects)
    random.Random(42).shuffle(shuffled)

    baseline = strategy.group(tuple(_input(subject) for subject in subjects))
    shuffled_result = strategy.group(tuple(_input(subject) for subject in shuffled))

    baseline_sets = {frozenset(candidate.members) for candidate in baseline.candidates}
    shuffled_sets = {frozenset(candidate.members) for candidate in shuffled_result.candidates}
    assert baseline_sets == shuffled_sets


def test_truncation_only_assessments_not_grouped_via_engine() -> None:
    assessment_a = _assessment_with_truncation()
    assessment_b = _assessment_with_truncation()
    engine = _engine_with_deterministic_strategy()

    result = engine.group((_assessment_input(assessment_a), _assessment_input(assessment_b)), strategy_id=STRATEGY_ID)

    assert result.candidates == ()
    assert len(result.ungrouped_subjects) == 2


def test_empty_assessments_not_grouped_via_engine() -> None:
    assessment_a = DiagnosticAssessment(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=(),
        limitations=(),
    )
    assessment_b = DiagnosticAssessment(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=(),
        limitations=(),
    )
    engine = _engine_with_deterministic_strategy()

    result = engine.group((_assessment_input(assessment_a), _assessment_input(assessment_b)), strategy_id=STRATEGY_ID)

    assert result.candidates == ()
    assert len(result.ungrouped_subjects) == 2


def test_non_lifecycle_finding_groups_without_instance_ids() -> None:
    finding = ProblemGroupingSubjectFinding(
        kind=DiagnosticFindingKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
        scope=LifecycleAnomalyScope.ATTEMPT,
        source_anomaly_kind=LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY,
        lifecycle_transition=None,
    )
    subject_a = _subject(findings=(finding,))
    subject_b = _subject(findings=(finding,))
    strategy = DeterministicProblemGroupingStrategy()

    result = strategy.group((_input(subject_a), _input(subject_b)))

    assert len(result.candidates) == 1
    assert subject_a.ref.task_id != subject_b.ref.task_id
    assert subject_a.ref.run_id != subject_b.ref.run_id


def test_deterministic_result_unchanged_with_feature_projector() -> None:
    assessment_a = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    assessment_b = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    without_projector = _engine_with_deterministic_strategy()
    with_projector = _engine_with_deterministic_strategy(
        feature_projector=_AssessmentFeatureProjector(),
    )

    baseline = without_projector.group(
        (_assessment_input(assessment_a), _assessment_input(assessment_b)),
        strategy_id=STRATEGY_ID,
    )
    featured = with_projector.group(
        (_assessment_input(assessment_a), _assessment_input(assessment_b)),
        strategy_id=STRATEGY_ID,
    )

    assert featured == baseline
