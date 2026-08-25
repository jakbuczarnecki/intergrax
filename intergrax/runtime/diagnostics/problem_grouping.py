# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin-capable problem grouping over diagnostic assessments (DIAG-5A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol, runtime_checkable

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticFinding,
    DiagnosticFindingKind,
    DiagnosticLimitation,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyKind, LifecycleAnomalyScope

ProblemGroupingStrategyId = NewType("ProblemGroupingStrategyId", str)
ProblemGroupingStrategyVersion = NewType("ProblemGroupingStrategyVersion", str)


class ProblemGroupingMethod(StrEnum):
    """High-level grouping approach declared by a strategy."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"


class ProblemGroupingStrategyError(Exception):
    """Raised when a grouping strategy algorithm fails."""


class ProblemGroupingIntegrityError(Exception):
    """Raised when input or strategy output violates grouping contracts."""


class DuplicateProblemGroupingStrategyError(ProblemGroupingIntegrityError):
    """Raised when a strategy_id is registered more than once."""


class MissingProblemGroupingStrategyError(ProblemGroupingIntegrityError):
    """Raised when a strategy_id cannot be resolved."""


def _validate_strategy_id(value: object) -> ProblemGroupingStrategyId:
    if type(value) is not str:
        raise TypeError("ProblemGroupingStrategyId must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError("ProblemGroupingStrategyId must be non-empty")
    return ProblemGroupingStrategyId(normalized)


def _validate_strategy_version(value: object) -> ProblemGroupingStrategyVersion:
    if type(value) is not str:
        raise TypeError("ProblemGroupingStrategyVersion must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError("ProblemGroupingStrategyVersion must be non-empty")
    return ProblemGroupingStrategyVersion(normalized)


@dataclass(frozen=True, slots=True)
class ProblemGroupingSubjectRef:
    """Stable identity for one execution assessment in a grouping invocation."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId


@dataclass(frozen=True, slots=True)
class ProblemGroupingSubjectFinding:
    """Normalized finding characteristics exposed to grouping strategies."""

    kind: DiagnosticFindingKind
    scope: LifecycleAnomalyScope
    source_anomaly_kind: LifecycleAnomalyKind


@dataclass(frozen=True, slots=True)
class ProblemGroupingSubjectLimitation:
    """Normalized limitation characteristics exposed to grouping strategies."""

    kind: DiagnosticLimitationKind
    source_anomaly_kind: LifecycleAnomalyKind


@dataclass(frozen=True, slots=True)
class ProblemGroupingSubject:
    """
    Normalized, immutable view of one DiagnosticAssessment for grouping.

    Does not expose arbitrary assessment internals; retains provenance fields
    needed for later validation and deterministic basis evidence.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    findings: tuple[ProblemGroupingSubjectFinding, ...]
    limitations: tuple[ProblemGroupingSubjectLimitation, ...]

    @property
    def ref(self) -> ProblemGroupingSubjectRef:
        return ProblemGroupingSubjectRef(
            tenant_id=self.tenant_id,
            task_id=self.task_id,
            run_id=self.run_id,
        )


@dataclass(frozen=True, slots=True)
class DeterministicProblemGroupingBasis:
    """Deterministic strategy evidence (DIAG-5B-ready)."""

    matched_finding_kinds: tuple[DiagnosticFindingKind, ...] = ()


@dataclass(frozen=True, slots=True)
class ProblemGroupingProvenance:
    """Audit trail for who grouped subjects and on what basis."""

    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod
    supporting_subject_refs: tuple[ProblemGroupingSubjectRef, ...]
    basis: DeterministicProblemGroupingBasis | None = None


@dataclass(frozen=True, slots=True)
class ProblemGroupingCandidate:
    """
    Strategy proposal that related executions share a recurring problem pattern.

    NOT canonical problem identity and NOT persisted.
    """

    members: tuple[ProblemGroupingSubjectRef, ...]
    provenance: ProblemGroupingProvenance


@dataclass(frozen=True, slots=True)
class ProblemGroupingStrategyCharacteristics:
    """Declarative metadata for reproducibility and audit."""

    method: ProblemGroupingMethod
    deterministic: bool


@dataclass(frozen=True, slots=True)
class ProblemGroupingStrategyResult:
    """Raw typed output from a grouping strategy before platform validation."""

    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    candidates: tuple[ProblemGroupingCandidate, ...]


@dataclass(frozen=True, slots=True)
class ProblemGroupingResult:
    """
    Platform-validated grouping hypothesis for one tenant invocation.

    Derived analytical output — NOT canonical execution truth and NOT persisted.
    """

    tenant_id: str
    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod
    candidates: tuple[ProblemGroupingCandidate, ...]
    ungrouped_subjects: tuple[ProblemGroupingSubjectRef, ...]


@runtime_checkable
class ProblemGroupingStrategy(Protocol):
    """
    Replaceable grouping algorithm contract.

    Synchronous by design: the diagnostics read-model pipeline (DIAG-2..4) is
    synchronous. Future async ML/LLM strategies should be wrapped by an adapter
    that blocks or by a higher orchestration layer — not by making this Protocol
    async speculatively.
    """

    @property
    def strategy_id(self) -> ProblemGroupingStrategyId: ...

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion: ...

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics: ...

    def group(
        self,
        subjects: tuple[ProblemGroupingSubject, ...],
    ) -> ProblemGroupingStrategyResult: ...


class ProblemGroupingStrategyRegistry:
    """Explicit, typed strategy registry — no reflection or entry-point discovery."""

    def __init__(self) -> None:
        self._strategies: dict[ProblemGroupingStrategyId, ProblemGroupingStrategy] = {}

    def register(self, strategy: ProblemGroupingStrategy) -> None:
        strategy_id = _validate_strategy_id(strategy.strategy_id)
        if strategy_id in self._strategies:
            raise DuplicateProblemGroupingStrategyError(
                f"strategy already registered: {strategy_id!r}"
            )
        self._strategies[strategy_id] = strategy

    def resolve(self, strategy_id: ProblemGroupingStrategyId) -> ProblemGroupingStrategy:
        normalized = _validate_strategy_id(strategy_id)
        strategy = self._strategies.get(normalized)
        if strategy is None:
            raise MissingProblemGroupingStrategyError(
                f"no strategy registered for id {normalized!r}"
            )
        return strategy

    def registered_strategy_ids(self) -> tuple[ProblemGroupingStrategyId, ...]:
        return tuple(sorted(self._strategies.keys()))


class ProblemGroupingEngine:
    """
    Owns subject normalization, strategy resolution, invocation, and validation.

    Strategies propose candidates; the engine enforces tenant isolation and
    contract integrity. Stable ProblemId lifecycle is out of scope (DIAG-5D).
    """

    def __init__(self, registry: ProblemGroupingStrategyRegistry) -> None:
        self._registry = registry

    def group(
        self,
        assessments: tuple[DiagnosticAssessment, ...],
        *,
        strategy_id: ProblemGroupingStrategyId,
    ) -> ProblemGroupingResult:
        subjects = _normalize_and_validate_subjects(assessments)
        strategy = self._registry.resolve(strategy_id)

        try:
            strategy_result = strategy.group(subjects)
        except ProblemGroupingIntegrityError:
            raise
        except ProblemGroupingStrategyError:
            raise
        except Exception as exc:
            raise ProblemGroupingStrategyError(
                f"grouping strategy {strategy.strategy_id!r} failed"
            ) from exc

        tenant_id = subjects[0].tenant_id
        validated_candidates = _validate_strategy_result(
            strategy_result=strategy_result,
            expected_strategy=strategy,
            input_subjects=subjects,
            tenant_id=tenant_id,
        )
        ungrouped = _compute_ungrouped_subjects(subjects, validated_candidates)

        return ProblemGroupingResult(
            tenant_id=tenant_id,
            strategy_id=strategy.strategy_id,
            strategy_version=strategy.strategy_version,
            method=strategy.characteristics.method,
            candidates=validated_candidates,
            ungrouped_subjects=ungrouped,
        )


def normalize_assessment(assessment: DiagnosticAssessment) -> ProblemGroupingSubject:
    """Map one DiagnosticAssessment to a grouping subject."""
    return ProblemGroupingSubject(
        tenant_id=assessment.tenant_id,
        task_id=assessment.task_id,
        run_id=assessment.run_id,
        findings=tuple(_normalize_finding(finding) for finding in assessment.findings),
        limitations=tuple(
            _normalize_limitation(limitation) for limitation in assessment.limitations
        ),
    )


def _normalize_finding(finding: DiagnosticFinding) -> ProblemGroupingSubjectFinding:
    return ProblemGroupingSubjectFinding(
        kind=finding.kind,
        scope=finding.scope,
        source_anomaly_kind=finding.source_anomaly_kind,
    )


def _normalize_limitation(
    limitation: DiagnosticLimitation,
) -> ProblemGroupingSubjectLimitation:
    return ProblemGroupingSubjectLimitation(
        kind=limitation.kind,
        source_anomaly_kind=limitation.source_anomaly_kind,
    )


def _normalize_and_validate_subjects(
    assessments: tuple[DiagnosticAssessment, ...],
) -> tuple[ProblemGroupingSubject, ...]:
    if not assessments:
        raise ProblemGroupingIntegrityError("grouping requires at least one assessment")

    subjects: list[ProblemGroupingSubject] = []
    seen_refs: set[ProblemGroupingSubjectRef] = set()
    tenant_id: str | None = None

    for assessment in assessments:
        subject = normalize_assessment(assessment)
        if tenant_id is None:
            tenant_id = subject.tenant_id
        elif subject.tenant_id != tenant_id:
            raise ProblemGroupingIntegrityError(
                "mixed tenant_id in one grouping invocation is not allowed"
            )

        ref = subject.ref
        if ref in seen_refs:
            raise ProblemGroupingIntegrityError(
                f"duplicate subject in grouping input: {ref.task_id!r}/{ref.run_id!r}"
            )
        seen_refs.add(ref)
        subjects.append(subject)

    return tuple(subjects)


def _validate_strategy_result(
    *,
    strategy_result: ProblemGroupingStrategyResult,
    expected_strategy: ProblemGroupingStrategy,
    input_subjects: tuple[ProblemGroupingSubject, ...],
    tenant_id: str,
) -> tuple[ProblemGroupingCandidate, ...]:
    if strategy_result.strategy_id != expected_strategy.strategy_id:
        raise ProblemGroupingIntegrityError(
            "strategy result strategy_id does not match invoked strategy"
        )
    if strategy_result.strategy_version != expected_strategy.strategy_version:
        raise ProblemGroupingIntegrityError(
            "strategy result strategy_version does not match invoked strategy"
        )

    allowed_refs = {subject.ref for subject in input_subjects}
    validated: list[ProblemGroupingCandidate] = []

    for candidate in strategy_result.candidates:
        validated.append(
            _validate_candidate(
                candidate=candidate,
                allowed_refs=allowed_refs,
                tenant_id=tenant_id,
                expected_strategy=expected_strategy,
            )
        )

    return tuple(validated)


def _validate_candidate(
    *,
    candidate: ProblemGroupingCandidate,
    allowed_refs: set[ProblemGroupingSubjectRef],
    tenant_id: str,
    expected_strategy: ProblemGroupingStrategy,
) -> ProblemGroupingCandidate:
    if len(candidate.members) < 2:
        raise ProblemGroupingIntegrityError(
            "grouping candidate must contain at least two members"
        )

    provenance = candidate.provenance
    if provenance.strategy_id != expected_strategy.strategy_id:
        raise ProblemGroupingIntegrityError(
            "candidate provenance strategy_id does not match invoked strategy"
        )
    if provenance.strategy_version != expected_strategy.strategy_version:
        raise ProblemGroupingIntegrityError(
            "candidate provenance strategy_version does not match invoked strategy"
        )

    seen_members: set[ProblemGroupingSubjectRef] = set()
    for member in candidate.members:
        if member.tenant_id != tenant_id:
            raise ProblemGroupingIntegrityError(
                "candidate member tenant_id does not match grouping invocation tenant"
            )
        if member not in allowed_refs:
            raise ProblemGroupingIntegrityError(
                "candidate member is not present in grouping input subjects"
            )
        if member in seen_members:
            raise ProblemGroupingIntegrityError(
                "duplicate member within one grouping candidate"
            )
        seen_members.add(member)

    return candidate


def _compute_ungrouped_subjects(
    subjects: tuple[ProblemGroupingSubject, ...],
    candidates: tuple[ProblemGroupingCandidate, ...],
) -> tuple[ProblemGroupingSubjectRef, ...]:
    grouped: set[ProblemGroupingSubjectRef] = set()
    for candidate in candidates:
        grouped.update(candidate.members)

    ungrouped = [subject.ref for subject in subjects if subject.ref not in grouped]
    return tuple(ungrouped)
