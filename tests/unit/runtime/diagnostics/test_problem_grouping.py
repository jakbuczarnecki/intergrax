# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticAssessmentBuilder,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    DeterministicProblemGroupingBasis,
    DuplicateProblemGroupingStrategyError,
    MissingProblemGroupingStrategyError,
    ProblemGroupingBasis,
    ProblemGroupingBasisKind,
    ProblemGroupingCandidate,
    ProblemGroupingEngine,
    ProblemGroupingIntegrityError,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingStrategy,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyRegistry,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubject,
    ProblemGroupingSubjectRef,
    normalize_assessment,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_FAKE_STRATEGY_ID = ProblemGroupingStrategyId("test.fake_pair")
_FAKE_STRATEGY_VERSION = ProblemGroupingStrategyVersion("1.0.0")


def _assessment(
    *,
    tenant_id: str = _TENANT_A,
    task_id=None,
    run_id=None,
) -> DiagnosticAssessment:
    return DiagnosticAssessment(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        findings=(),
        limitations=(),
    )


def _subject_ref(assessment: DiagnosticAssessment) -> ProblemGroupingSubjectRef:
    return ProblemGroupingSubjectRef(
        tenant_id=assessment.tenant_id,
        task_id=assessment.task_id,
        run_id=assessment.run_id,
    )


def _provenance(
    *,
    members: tuple[ProblemGroupingSubjectRef, ...],
    method: ProblemGroupingMethod = ProblemGroupingMethod.DETERMINISTIC,
    basis: ProblemGroupingBasis | None = None,
) -> ProblemGroupingProvenance:
    return ProblemGroupingProvenance(
        strategy_id=_FAKE_STRATEGY_ID,
        strategy_version=_FAKE_STRATEGY_VERSION,
        method=method,
        supporting_subject_refs=members,
        basis=basis if basis is not None else DeterministicProblemGroupingBasis(),
    )


class _ConfigurableFakeStrategy:
    """Test-only strategy — not a production grouping algorithm."""

    def __init__(
        self,
        *,
        candidates: tuple[ProblemGroupingCandidate, ...] = (),
    ) -> None:
        self._candidates = candidates

    @property
    def strategy_id(self) -> ProblemGroupingStrategyId:
        return _FAKE_STRATEGY_ID

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion:
        return _FAKE_STRATEGY_VERSION

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.DETERMINISTIC,
            deterministic=True,
        )

    def group(
        self,
        subjects: tuple[ProblemGroupingSubject, ...],
    ) -> ProblemGroupingStrategyResult:
        if not self._candidates and len(subjects) >= 2:
            members = (subjects[0].ref, subjects[1].ref)
            self._candidates = (
                ProblemGroupingCandidate(
                    members=members,
                    provenance=_provenance(members=members),
                ),
            )
        return ProblemGroupingStrategyResult(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            candidates=self._candidates,
        )


def _engine_with_strategy(strategy: ProblemGroupingStrategy) -> ProblemGroupingEngine:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(strategy)
    return ProblemGroupingEngine(registry)


def test_registry_register_resolve_and_list() -> None:
    registry = ProblemGroupingStrategyRegistry()
    strategy = _ConfigurableFakeStrategy()

    registry.register(strategy)
    resolved = registry.resolve(_FAKE_STRATEGY_ID)

    assert resolved.strategy_id == _FAKE_STRATEGY_ID
    assert registry.registered_strategy_ids() == (_FAKE_STRATEGY_ID,)


def test_registry_duplicate_id_fails() -> None:
    registry = ProblemGroupingStrategyRegistry()
    registry.register(_ConfigurableFakeStrategy())

    with pytest.raises(DuplicateProblemGroupingStrategyError):
        registry.register(_ConfigurableFakeStrategy())


def test_registry_unknown_id_fails() -> None:
    registry = ProblemGroupingStrategyRegistry()

    with pytest.raises(MissingProblemGroupingStrategyError):
        registry.resolve(ProblemGroupingStrategyId("missing.strategy"))


def test_engine_groups_first_pair_and_leaves_third_ungrouped() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    assessment_c = _assessment()
    engine = _engine_with_strategy(_ConfigurableFakeStrategy())

    result = engine.group(
        (assessment_a, assessment_b, assessment_c),
        strategy_id=_FAKE_STRATEGY_ID,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].members == (_subject_ref(assessment_a), _subject_ref(assessment_b))
    assert result.ungrouped_subjects == (_subject_ref(assessment_c),)
    assert result.tenant_id == _TENANT_A
    assert result.strategy_id == _FAKE_STRATEGY_ID
    assert result.strategy_version == _FAKE_STRATEGY_VERSION
    assert result.method is ProblemGroupingMethod.DETERMINISTIC


def test_foreign_member_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    foreign = _assessment(tenant_id=_TENANT_A)
    members = (_subject_ref(assessment_a), _subject_ref(foreign))
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=members,
                provenance=_provenance(members=members),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="not present"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_mixed_tenants_fail_before_strategy() -> None:
    assessment_a = _assessment(tenant_id=_TENANT_A)
    assessment_b = _assessment(tenant_id=_TENANT_B)
    engine = _engine_with_strategy(_ConfigurableFakeStrategy())

    with pytest.raises(ProblemGroupingIntegrityError, match="mixed tenant_id"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_duplicate_input_rejected() -> None:
    assessment = _assessment()
    engine = _engine_with_strategy(_ConfigurableFakeStrategy())

    with pytest.raises(ProblemGroupingIntegrityError, match="duplicate subject"):
        engine.group((assessment, assessment), strategy_id=_FAKE_STRATEGY_ID)


def test_overlapping_candidates_allowed() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    assessment_c = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    ref_c = _subject_ref(assessment_c)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=_provenance(members=(ref_a, ref_b)),
            ),
            ProblemGroupingCandidate(
                members=(ref_b, ref_c),
                provenance=_provenance(members=(ref_b, ref_c)),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    result = engine.group(
        (assessment_a, assessment_b, assessment_c),
        strategy_id=_FAKE_STRATEGY_ID,
    )

    assert len(result.candidates) == 2
    assert result.ungrouped_subjects == ()


def test_singleton_candidate_rejected() -> None:
    assessment_a = _assessment()
    ref_a = _subject_ref(assessment_a)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a,),
                provenance=_provenance(members=(ref_a,)),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="at least two members"):
        engine.group((assessment_a,), strategy_id=_FAKE_STRATEGY_ID)


def test_strategy_identity_mismatch_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)

    class _SpoofedStrategy(_ConfigurableFakeStrategy):
        def group(
            self,
            subjects: tuple[ProblemGroupingSubject, ...],
        ) -> ProblemGroupingStrategyResult:
            return ProblemGroupingStrategyResult(
                strategy_id=ProblemGroupingStrategyId("spoofed.strategy"),
                strategy_version=_FAKE_STRATEGY_VERSION,
                candidates=(
                    ProblemGroupingCandidate(
                        members=(ref_a, ref_b),
                        provenance=_provenance(members=(ref_a, ref_b)),
                    ),
                ),
            )

    engine = _engine_with_strategy(_SpoofedStrategy())

    with pytest.raises(ProblemGroupingIntegrityError, match="strategy_id"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_engine_determinism() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    assessment_c = _assessment()
    engine = _engine_with_strategy(_ConfigurableFakeStrategy())

    first = engine.group(
        (assessment_a, assessment_b, assessment_c),
        strategy_id=_FAKE_STRATEGY_ID,
    )
    second = engine.group(
        (assessment_a, assessment_b, assessment_c),
        strategy_id=_FAKE_STRATEGY_ID,
    )

    assert first == second


@dataclass(frozen=True, slots=True)
class _TestSemanticProblemGroupingBasis:
    """Test-only basis proving plugin extensibility without engine changes."""

    similarity_threshold: float = 0.9

    @property
    def kind(self) -> ProblemGroupingBasisKind:
        return ProblemGroupingBasisKind.SEMANTIC


def test_extensible_basis_accepted_by_provenance() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    basis = _TestSemanticProblemGroupingBasis()

    provenance = ProblemGroupingProvenance(
        strategy_id=_FAKE_STRATEGY_ID,
        strategy_version=_FAKE_STRATEGY_VERSION,
        method=ProblemGroupingMethod.SEMANTIC,
        supporting_subject_refs=(ref_a, ref_b),
        basis=basis,
    )

    assert isinstance(provenance.basis, ProblemGroupingBasis)
    assert provenance.basis is not None
    assert provenance.basis.kind is ProblemGroupingBasisKind.SEMANTIC


def test_deterministic_basis_valid_with_deterministic_strategy() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    basis = DeterministicProblemGroupingBasis()
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=_provenance(members=(ref_a, ref_b), basis=basis),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    result = engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)

    assert result.candidates[0].provenance.basis == basis
    assert result.candidates[0].provenance.basis.kind is ProblemGroupingBasisKind.DETERMINISTIC


def test_method_spoof_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=_provenance(
                    members=(ref_a, ref_b),
                    method=ProblemGroupingMethod.LLM,
                ),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="method"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_basis_method_mismatch_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=_provenance(
                    members=(ref_a, ref_b),
                    basis=_TestSemanticProblemGroupingBasis(),
                ),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="basis kind"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_foreign_supporting_ref_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    foreign = _subject_ref(_assessment())
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=ProblemGroupingProvenance(
                    strategy_id=_FAKE_STRATEGY_ID,
                    strategy_version=_FAKE_STRATEGY_VERSION,
                    method=ProblemGroupingMethod.DETERMINISTIC,
                    supporting_subject_refs=(ref_a, foreign),
                    basis=DeterministicProblemGroupingBasis(),
                ),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="supporting_subject_ref"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_supporting_refs_must_equal_members() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    assessment_c = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    ref_c = _subject_ref(assessment_c)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=ProblemGroupingProvenance(
                    strategy_id=_FAKE_STRATEGY_ID,
                    strategy_version=_FAKE_STRATEGY_VERSION,
                    method=ProblemGroupingMethod.DETERMINISTIC,
                    supporting_subject_refs=(ref_a, ref_c),
                    basis=DeterministicProblemGroupingBasis(),
                ),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="must equal candidate members"):
        engine.group(
            (assessment_a, assessment_b, assessment_c),
            strategy_id=_FAKE_STRATEGY_ID,
        )


def test_duplicate_supporting_ref_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_a, ref_b),
                provenance=ProblemGroupingProvenance(
                    strategy_id=_FAKE_STRATEGY_ID,
                    strategy_version=_FAKE_STRATEGY_VERSION,
                    method=ProblemGroupingMethod.DETERMINISTIC,
                    supporting_subject_refs=(ref_a, ref_a),
                    basis=DeterministicProblemGroupingBasis(),
                ),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    with pytest.raises(ProblemGroupingIntegrityError, match="duplicate supporting_subject_ref"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


def test_member_order_normalized_to_input_order() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()
    ref_a = _subject_ref(assessment_a)
    ref_b = _subject_ref(assessment_b)
    strategy = _ConfigurableFakeStrategy(
        candidates=(
            ProblemGroupingCandidate(
                members=(ref_b, ref_a),
                provenance=_provenance(members=(ref_b, ref_a)),
            ),
        ),
    )
    engine = _engine_with_strategy(strategy)

    result = engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)

    assert result.candidates[0].members == (ref_a, ref_b)
    assert result.candidates[0].provenance.supporting_subject_refs == (ref_a, ref_b)


def test_registered_strategy_mutation_rejected() -> None:
    assessment_a = _assessment()
    assessment_b = _assessment()

    class _MutableStrategy(_ConfigurableFakeStrategy):
        def __init__(self) -> None:
            super().__init__()
            self._version = _FAKE_STRATEGY_VERSION

        @property
        def strategy_version(self) -> ProblemGroupingStrategyVersion:
            return self._version

    strategy = _MutableStrategy()
    registry = ProblemGroupingStrategyRegistry()
    registry.register(strategy)
    engine = ProblemGroupingEngine(registry)
    strategy._version = ProblemGroupingStrategyVersion("9.9.9")

    with pytest.raises(ProblemGroupingIntegrityError, match="mutated after registration"):
        engine.group((assessment_a, assessment_b), strategy_id=_FAKE_STRATEGY_ID)


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


def test_normalize_assessment_passes_lifecycle_transition() -> None:
    assessment = _assess_attempt_sequence(
        [
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.TASK_COMPLETED,
            RuntimeEventType.RETRY_SCHEDULED,
        ]
    )
    finding = next(
        item
        for item in assessment.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )
    subject = normalize_assessment(assessment)
    subject_finding = next(
        item
        for item in subject.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )

    assert subject_finding.lifecycle_transition is finding.lifecycle_transition
    assert subject_finding.lifecycle_transition is not None


def test_normalized_subject_findings_differ_for_structural_collision() -> None:
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

    subject_a = normalize_assessment(assessment_a)
    subject_b = normalize_assessment(assessment_b)
    finding_a = next(
        item
        for item in subject_a.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )
    finding_b = next(
        item
        for item in subject_b.findings
        if item.kind is DiagnosticFindingKind.EVENT_AFTER_TERMINAL
    )

    assert finding_a.lifecycle_transition != finding_b.lifecycle_transition
    assert finding_a.kind == finding_b.kind
    assert finding_a.scope == finding_b.scope
    assert finding_a.source_anomaly_kind == finding_b.source_anomaly_kind


def test_normalized_non_lifecycle_finding_lifecycle_transition_is_none() -> None:
    assessment = DiagnosticAssessment(
        tenant_id=_TENANT_A,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        findings=(),
        limitations=(),
    )
    subject = normalize_assessment(assessment)
    assert subject.findings == ()
