# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin-capable problem grouping over diagnostic assessments (DIAG-5A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NewType, Protocol, runtime_checkable

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticFinding,
    DiagnosticFindingKind,
    DiagnosticLimitation,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyKind, LifecycleAnomalyScope, LifecycleViolationTransition

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.problem_grouping_features import (
        ProblemGroupingFeatureProjector,
        ProblemGroupingFeatureSet,
    )

ProblemGroupingStrategyId = NewType("ProblemGroupingStrategyId", str)
ProblemGroupingStrategyVersion = NewType("ProblemGroupingStrategyVersion", str)


class ProblemGroupingMethod(StrEnum):
    """High-level grouping approach declared by a strategy."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"


class ProblemGroupingBasisKind(StrEnum):
    """Typed discriminator for strategy-specific grouping evidence."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"


_METHOD_TO_BASIS_KIND: dict[ProblemGroupingMethod, ProblemGroupingBasisKind] = {
    ProblemGroupingMethod.DETERMINISTIC: ProblemGroupingBasisKind.DETERMINISTIC,
    ProblemGroupingMethod.SEMANTIC: ProblemGroupingBasisKind.SEMANTIC,
    ProblemGroupingMethod.ML: ProblemGroupingBasisKind.ML,
    ProblemGroupingMethod.LLM: ProblemGroupingBasisKind.LLM,
    ProblemGroupingMethod.HYBRID: ProblemGroupingBasisKind.HYBRID,
}


@runtime_checkable
class ProblemGroupingBasis(Protocol):
    """Strategy-specific immutable grouping evidence contract."""

    @property
    def kind(self) -> ProblemGroupingBasisKind: ...


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
    lifecycle_transition: LifecycleViolationTransition | None = None


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
class DeterministicFindingSignature:
    """Typed structural descriptor for one normalized grouping finding."""

    kind: DiagnosticFindingKind
    scope: LifecycleAnomalyScope
    source_anomaly_kind: LifecycleAnomalyKind
    lifecycle_transition: LifecycleViolationTransition | None = None


@dataclass(frozen=True, slots=True)
class DeterministicLimitationSignature:
    """Typed structural descriptor for one normalized grouping limitation."""

    kind: DiagnosticLimitationKind
    source_anomaly_kind: LifecycleAnomalyKind


@dataclass(frozen=True, slots=True)
class DeterministicProblemSignature:
    """
    Exact structural identity for deterministic problem grouping (DIAG-5B).

    Equality is typed field equality — not an opaque fingerprint string.
    """

    findings: tuple[DeterministicFindingSignature, ...]
    limitations: tuple[DeterministicLimitationSignature, ...]


@dataclass(frozen=True, slots=True)
class DeterministicProblemGroupingBasis:
    """Deterministic strategy evidence: the exact signature that grouped members."""

    signature: DeterministicProblemSignature

    @property
    def kind(self) -> ProblemGroupingBasisKind:
        return ProblemGroupingBasisKind.DETERMINISTIC


@dataclass(frozen=True, slots=True)
class ProblemGroupingProvenance:
    """Audit trail for who grouped subjects and on what basis."""

    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod
    supporting_subject_refs: tuple[ProblemGroupingSubjectRef, ...]
    basis: ProblemGroupingBasis | None = None


@dataclass(frozen=True, slots=True)
class ProblemGroupingCandidate:
    """
    Strategy proposal that related executions share a recurring problem pattern.

    NOT canonical problem identity and NOT persisted.
    """

    members: tuple[ProblemGroupingSubjectRef, ...]
    provenance: ProblemGroupingProvenance


@dataclass(frozen=True, slots=True)
class ProblemGroupingInput:
    """
    Central strategy invocation input: normalized subject plus optional features.

    The engine constructs inputs; strategies must not rebuild subjects from
  persistence or duplicate normalization pipelines.
    """

    subject: ProblemGroupingSubject
    features: ProblemGroupingFeatureSet | None = None


@dataclass(frozen=True, slots=True)
class ProblemGroupingStrategyCharacteristics:
    """Declarative metadata for reproducibility and audit."""

    method: ProblemGroupingMethod
    deterministic: bool
    requires_features: bool = False


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
        inputs: tuple[ProblemGroupingInput, ...],
    ) -> ProblemGroupingStrategyResult: ...


@dataclass(frozen=True, slots=True)
class _RegisteredProblemGroupingStrategy:
    """Validated strategy identity captured at registration time."""

    strategy: ProblemGroupingStrategy
    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    characteristics: ProblemGroupingStrategyCharacteristics


class ProblemGroupingStrategyRegistry:
    """Explicit, typed strategy registry — no reflection or entry-point discovery."""

    def __init__(self) -> None:
        self._strategies: dict[
            ProblemGroupingStrategyId, _RegisteredProblemGroupingStrategy
        ] = {}

    def register(self, strategy: ProblemGroupingStrategy) -> None:
        registration = _validate_strategy_registration(strategy)
        if registration.strategy_id in self._strategies:
            raise DuplicateProblemGroupingStrategyError(
                f"strategy already registered: {registration.strategy_id!r}"
            )
        self._strategies[registration.strategy_id] = registration

    def resolve(self, strategy_id: ProblemGroupingStrategyId) -> ProblemGroupingStrategy:
        return self._resolve_registration(strategy_id).strategy

    def _resolve_registration(
        self, strategy_id: ProblemGroupingStrategyId
    ) -> _RegisteredProblemGroupingStrategy:
        normalized = _validate_strategy_id(strategy_id)
        registration = self._strategies.get(normalized)
        if registration is None:
            raise MissingProblemGroupingStrategyError(
                f"no strategy registered for id {normalized!r}"
            )
        _assert_registration_coherent(registration)
        return registration

    def registered_strategy_ids(self) -> tuple[ProblemGroupingStrategyId, ...]:
        return tuple(sorted(self._strategies.keys()))


class ProblemGroupingEngine:
    """
    Owns subject normalization, strategy resolution, invocation, and validation.

    Strategies propose candidates; the engine enforces tenant isolation and
    contract integrity. Stable ProblemId lifecycle is out of scope (DIAG-5D).
    """

    def __init__(
        self,
        registry: ProblemGroupingStrategyRegistry,
        *,
        feature_projector: ProblemGroupingFeatureProjector | None = None,
    ) -> None:
        self._registry = registry
        self._feature_projector = feature_projector

    def group(
        self,
        assessments: tuple[DiagnosticAssessment, ...],
        *,
        strategy_id: ProblemGroupingStrategyId,
        feature_projector: ProblemGroupingFeatureProjector | None = None,
    ) -> ProblemGroupingResult:
        projector = (
            feature_projector
            if feature_projector is not None
            else self._feature_projector
        )
        inputs = _normalize_and_validate_inputs(
            assessments,
            feature_projector=projector,
        )
        registration = self._registry._resolve_registration(strategy_id)
        strategy = registration.strategy

        _validate_inputs_for_strategy(
            inputs=inputs,
            registration=registration,
            feature_projector=projector,
        )

        try:
            strategy_result = strategy.group(inputs)
        except ProblemGroupingIntegrityError:
            raise
        except ProblemGroupingStrategyError:
            raise
        except Exception as exc:
            raise ProblemGroupingStrategyError(
                f"grouping strategy {registration.strategy_id!r} failed"
            ) from exc

        subjects = tuple(input_item.subject for input_item in inputs)
        tenant_id = subjects[0].tenant_id
        input_order = tuple(subject.ref for subject in subjects)
        validated_candidates = _validate_strategy_result(
            strategy_result=strategy_result,
            registration=registration,
            input_order=input_order,
            tenant_id=tenant_id,
        )
        ungrouped = _compute_ungrouped_subjects(subjects, validated_candidates)

        return ProblemGroupingResult(
            tenant_id=tenant_id,
            strategy_id=registration.strategy_id,
            strategy_version=registration.strategy_version,
            method=registration.characteristics.method,
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
        lifecycle_transition=finding.lifecycle_transition,
    )


def _normalize_limitation(
    limitation: DiagnosticLimitation,
) -> ProblemGroupingSubjectLimitation:
    return ProblemGroupingSubjectLimitation(
        kind=limitation.kind,
        source_anomaly_kind=limitation.source_anomaly_kind,
    )


def _normalize_and_validate_inputs(
    assessments: tuple[DiagnosticAssessment, ...],
    *,
    feature_projector: ProblemGroupingFeatureProjector | None,
) -> tuple[ProblemGroupingInput, ...]:
    if not assessments:
        raise ProblemGroupingIntegrityError("grouping requires at least one assessment")

    inputs: list[ProblemGroupingInput] = []
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

        features = None
        if feature_projector is not None:
            features = feature_projector.project(assessment, subject)
        inputs.append(ProblemGroupingInput(subject=subject, features=features))

    return tuple(inputs)


def _validate_inputs_for_strategy(
    *,
    inputs: tuple[ProblemGroupingInput, ...],
    registration: _RegisteredProblemGroupingStrategy,
    feature_projector: ProblemGroupingFeatureProjector | None,
) -> None:
    if registration.characteristics.requires_features:
        if feature_projector is None or any(
            input_item.features is None for input_item in inputs
        ):
            raise ProblemGroupingIntegrityError(
                "strategy requires features but features are not available"
            )

    for input_item in inputs:
        _validate_input_feature_coherence(input_item)


def _validate_input_feature_coherence(input_item: ProblemGroupingInput) -> None:
    features = input_item.features
    if features is None:
        return

    subject = input_item.subject
    if features.subject_ref != subject.ref:
        raise ProblemGroupingIntegrityError(
            "feature subject_ref does not match input subject"
        )
    if features.subject_ref.tenant_id != subject.tenant_id:
        raise ProblemGroupingIntegrityError(
            "feature subject_ref tenant_id does not match grouping invocation tenant"
        )

    from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
        build_deterministic_problem_signature,
    )

    expected_signature = build_deterministic_problem_signature(subject)
    if features.structural_signature != expected_signature:
        raise ProblemGroupingIntegrityError(
            "feature structural_signature does not match input subject"
        )

    if not str(features.representation_version).strip():
        raise ProblemGroupingIntegrityError(
            "feature representation_version must be non-empty"
        )


def _validate_strategy_registration(
    strategy: ProblemGroupingStrategy,
) -> _RegisteredProblemGroupingStrategy:
    strategy_id = _validate_strategy_id(strategy.strategy_id)
    strategy_version = _validate_strategy_version(strategy.strategy_version)
    characteristics = _validate_strategy_characteristics(strategy.characteristics)
    return _RegisteredProblemGroupingStrategy(
        strategy=strategy,
        strategy_id=strategy_id,
        strategy_version=strategy_version,
        characteristics=characteristics,
    )


def _validate_strategy_characteristics(
    value: object,
) -> ProblemGroupingStrategyCharacteristics:
    if type(value) is not ProblemGroupingStrategyCharacteristics:
        raise TypeError("strategy characteristics must be ProblemGroupingStrategyCharacteristics")
    if not isinstance(value.method, ProblemGroupingMethod):
        raise TypeError("strategy characteristics method must be ProblemGroupingMethod")
    if type(value.deterministic) is not bool:
        raise TypeError("strategy characteristics deterministic must be bool")
    if type(value.requires_features) is not bool:
        raise TypeError("strategy characteristics requires_features must be bool")
    return value


def _assert_registration_coherent(
    registration: _RegisteredProblemGroupingStrategy,
) -> None:
    strategy = registration.strategy
    if strategy.strategy_id != registration.strategy_id:
        raise ProblemGroupingIntegrityError(
            "registered strategy_id mutated after registration"
        )
    if strategy.strategy_version != registration.strategy_version:
        raise ProblemGroupingIntegrityError(
            "registered strategy_version mutated after registration"
        )
    try:
        live_characteristics = _validate_strategy_characteristics(strategy.characteristics)
    except TypeError as exc:
        raise ProblemGroupingIntegrityError(
            "registered strategy characteristics became invalid after registration"
        ) from exc
    if live_characteristics != registration.characteristics:
        raise ProblemGroupingIntegrityError(
            "registered strategy characteristics mutated after registration"
        )


def _validate_strategy_result(
    *,
    strategy_result: ProblemGroupingStrategyResult,
    registration: _RegisteredProblemGroupingStrategy,
    input_order: tuple[ProblemGroupingSubjectRef, ...],
    tenant_id: str,
) -> tuple[ProblemGroupingCandidate, ...]:
    if strategy_result.strategy_id != registration.strategy_id:
        raise ProblemGroupingIntegrityError(
            "strategy result strategy_id does not match invoked strategy"
        )
    if strategy_result.strategy_version != registration.strategy_version:
        raise ProblemGroupingIntegrityError(
            "strategy result strategy_version does not match invoked strategy"
        )

    allowed_refs = set(input_order)
    validated: list[ProblemGroupingCandidate] = []

    for candidate in strategy_result.candidates:
        validated.append(
            _validate_candidate(
                candidate=candidate,
                allowed_refs=allowed_refs,
                input_order=input_order,
                tenant_id=tenant_id,
                registration=registration,
            )
        )

    return tuple(validated)


def _validate_candidate(
    *,
    candidate: ProblemGroupingCandidate,
    allowed_refs: set[ProblemGroupingSubjectRef],
    input_order: tuple[ProblemGroupingSubjectRef, ...],
    tenant_id: str,
    registration: _RegisteredProblemGroupingStrategy,
) -> ProblemGroupingCandidate:
    if len(candidate.members) < 2:
        raise ProblemGroupingIntegrityError(
            "grouping candidate must contain at least two members"
        )

    provenance = candidate.provenance
    if provenance.strategy_id != registration.strategy_id:
        raise ProblemGroupingIntegrityError(
            "candidate provenance strategy_id does not match invoked strategy"
        )
    if provenance.strategy_version != registration.strategy_version:
        raise ProblemGroupingIntegrityError(
            "candidate provenance strategy_version does not match invoked strategy"
        )
    if provenance.method != registration.characteristics.method:
        raise ProblemGroupingIntegrityError(
            "candidate provenance method does not match invoked strategy method"
        )

    _validate_member_refs(
        members=candidate.members,
        allowed_refs=allowed_refs,
        tenant_id=tenant_id,
        duplicate_error="duplicate member within one grouping candidate",
        foreign_error="candidate member is not present in grouping input subjects",
        tenant_error="candidate member tenant_id does not match grouping invocation tenant",
    )
    normalized_members = _normalize_member_order(candidate.members, input_order)

    normalized_supporting = _validate_supporting_subject_refs(
        supporting_subject_refs=provenance.supporting_subject_refs,
        members=normalized_members,
        allowed_refs=allowed_refs,
        tenant_id=tenant_id,
        input_order=input_order,
    )
    _validate_basis_coherence(
        basis=provenance.basis,
        method=registration.characteristics.method,
    )

    if (
        normalized_members != candidate.members
        or normalized_supporting != provenance.supporting_subject_refs
    ):
        provenance = ProblemGroupingProvenance(
            strategy_id=provenance.strategy_id,
            strategy_version=provenance.strategy_version,
            method=provenance.method,
            supporting_subject_refs=normalized_supporting,
            basis=provenance.basis,
        )
        return ProblemGroupingCandidate(
            members=normalized_members,
            provenance=provenance,
        )

    return candidate


def _normalize_member_order(
    members: tuple[ProblemGroupingSubjectRef, ...],
    input_order: tuple[ProblemGroupingSubjectRef, ...],
) -> tuple[ProblemGroupingSubjectRef, ...]:
    member_set = set(members)
    return tuple(ref for ref in input_order if ref in member_set)


def _validate_member_refs(
    *,
    members: tuple[ProblemGroupingSubjectRef, ...],
    allowed_refs: set[ProblemGroupingSubjectRef],
    tenant_id: str,
    duplicate_error: str,
    foreign_error: str,
    tenant_error: str,
) -> None:
    seen_members: set[ProblemGroupingSubjectRef] = set()
    for member in members:
        if member.tenant_id != tenant_id:
            raise ProblemGroupingIntegrityError(tenant_error)
        if member not in allowed_refs:
            raise ProblemGroupingIntegrityError(foreign_error)
        if member in seen_members:
            raise ProblemGroupingIntegrityError(duplicate_error)
        seen_members.add(member)


def _validate_supporting_subject_refs(
    *,
    supporting_subject_refs: tuple[ProblemGroupingSubjectRef, ...],
    members: tuple[ProblemGroupingSubjectRef, ...],
    allowed_refs: set[ProblemGroupingSubjectRef],
    tenant_id: str,
    input_order: tuple[ProblemGroupingSubjectRef, ...],
) -> tuple[ProblemGroupingSubjectRef, ...]:
    seen_supporting: set[ProblemGroupingSubjectRef] = set()
    for ref in supporting_subject_refs:
        if ref.tenant_id != tenant_id:
            raise ProblemGroupingIntegrityError(
                "candidate provenance supporting_subject_ref tenant_id "
                "does not match grouping invocation tenant"
            )
        if ref not in allowed_refs:
            raise ProblemGroupingIntegrityError(
                "candidate provenance supporting_subject_ref "
                "is not present in grouping input subjects"
            )
        if ref in seen_supporting:
            raise ProblemGroupingIntegrityError(
                "duplicate supporting_subject_ref within candidate provenance"
            )
        seen_supporting.add(ref)

    if set(supporting_subject_refs) != set(members):
        raise ProblemGroupingIntegrityError(
            "candidate provenance supporting_subject_refs must equal candidate members"
        )

    return _normalize_member_order(supporting_subject_refs, input_order)


def _validate_basis_coherence(
    *,
    basis: ProblemGroupingBasis | None,
    method: ProblemGroupingMethod,
) -> None:
    if basis is None:
        return
    if not isinstance(basis, ProblemGroupingBasis):
        raise ProblemGroupingIntegrityError(
            "candidate provenance basis must implement ProblemGroupingBasis"
        )
    expected_kind = _METHOD_TO_BASIS_KIND[method]
    if basis.kind != expected_kind:
        raise ProblemGroupingIntegrityError(
            "candidate provenance basis kind does not match strategy method"
        )


def _compute_ungrouped_subjects(
    subjects: tuple[ProblemGroupingSubject, ...],
    candidates: tuple[ProblemGroupingCandidate, ...],
) -> tuple[ProblemGroupingSubjectRef, ...]:
    grouped: set[ProblemGroupingSubjectRef] = set()
    for candidate in candidates:
        grouped.update(candidate.members)

    ungrouped = [subject.ref for subject in subjects if subject.ref not in grouped]
    return tuple(ungrouped)
