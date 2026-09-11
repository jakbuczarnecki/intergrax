# © Artur Czarnecki. All rights reserved.

"""Declarative AI Incident evaluator failure → qualification signal mapping (DS-E2E-15J-T1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

from intergrax.decision_system.qualification.signals import (
    EvaluatorQualificationSignal,
    ModelBehaviorQualificationSignal,
    ObservabilityQualificationSignal,
    PlatformContractQualificationSignal,
)
from intergrax.decision_system.qualification.taxonomy import DecisionFailureBoundary
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator_failure_vocabulary import (
    AI_INCIDENT_EVALUATOR_FAILURE_ID_PREFIXES,
    AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS,
    BOUNDED_RECOVERY_MISSING,
    CHALLENGE_DEFECT_CODE_MISMATCH,
    CHALLENGE_DEFECT_FAMILY_MISMATCH,
    CHALLENGE_NOT_SATISFIED_AFTER_RESOLUTION,
    CHALLENGE_TARGET_CLAIM_MISMATCH,
    COMPARISON_DOES_NOT_WEAKEN_H1_FROM_RUNTIME,
    COMPARISON_EVIDENCE_GATHERED_MISSING,
    COMPARISON_EVIDENCE_NOT_GATHERED,
    COMPARISON_IN_INITIAL_OBSERVABLE_SET,
    CRITIC_CONTENT_VALIDATION_FAILED,
    CRITIC_FALSIFICATION_MISSING,
    EVALUATOR_LOOP_BUDGET_EXCEEDED,
    EVIDENCE_CHALLENGE_MISSING,
    FAILED_CRITIC_VERDICT_MISSING,
    FAILED_CRITIC_VERDICT_REASON_MISMATCH,
    FINAL_CRITIC_VERDICT_NOT_PASSED,
    FINAL_SUMMARY_NOT_BOUNDED,
    FINAL_SUPPORTED_CLAIM_NOT_H3,
    FIXTURE_TRUTH_INTEGRITY,
    FOLLOW_UP_NOT_VIA_TOOLS,
    GROUND_TRUTH_LEAK_PREFIX,
    H1_FINAL_SUPPORTED_OR_PENDING,
    H1_NOT_PLAUSIBLE_FROM_RUNTIME_EVIDENCE,
    H1_NOT_WEAKENED,
    H1_NOT_WEAKENED_FROM_RUNTIME,
    H1_STILL_PENDING_AT_END,
    H2_CLAIM_MISSING,
    H2_DISPOSITION_NOT_DERIVED_FROM_EVIDENCE,
    H2_FINAL_SUPPORTED,
    H2_NOT_REJECTED,
    H2_REJECTION_MISSING_ATTENDANCE_REF,
    H3_CLAIM_MISSING,
    H3_CLAIM_NOT_INSUFFICIENT,
    H3_DIAGNOSIS_TELEMETRY_NOT_OBSERVABLE,
    MISSING_CLAIM_SET,
    H3_NOT_DERIVED_FROM_RUNTIME_EVIDENCE,
    H3_NOT_INSUFFICIENT_EVIDENCE,
    MISSING_DIAGNOSIS_CLAIM,
    DIAGNOSIS_CLAIM_NOT_ACCEPTABLE,
    NO_SUPPORTED_DIAGNOSIS_CLAIM,
    OPEN_CHALLENGE_MUST_NOT_INCLUDE_RESOLVING_EVIDENCE,
    SATISFIED_CHALLENGE_MISSING_COMPARISON_EVIDENCE,
    SATISFIED_CHALLENGE_MISSING_INITIAL_EVIDENCE,
    SATISFIED_CHALLENGE_MISSING_RESOLVING_EVIDENCE,
    STAFFING_ATTENDANCE_GATHERED_MISSING,
    STAFFING_ATTENDANCE_NOT_GATHERED,
    STAFFING_PRELIMINARY_NOT_GATHERED,
    STAFFING_PRELIMINARY_SHOULD_BE_STALE_FOR_INCIDENT,
    STALE_STAFFING_USED_AS_FINAL_SUPPORT,
    SUPPORTED_CLAIM_MISSING_COMPARISON_REF,
    SUPPORTED_CLAIM_MISSING_TELEMETRY_REF,
    SUPPORTED_DIAGNOSIS_EVIDENCE_NOT_OBSERVABLE,
    SUPPORTED_DIAGNOSIS_PRESENT,
    TELEMETRY_DOES_NOT_SUPPORT_H3_FROM_RUNTIME,
    TELEMETRY_EVIDENCE_GATHERED_MISSING,
    TELEMETRY_EVIDENCE_NOT_IN_GRAPH,
    TELEMETRY_IN_INITIAL_OBSERVABLE_SET,
    TELEMETRY_NOT_OBSERVABLE,
    TELEMETRY_NOT_UNAVAILABLE,
    TELEMETRY_VISIBLE_BEFORE_REVISION,
    TOOL_RUNTIME_NOT_EXERCISED,
    UNAVAILABLE_TELEMETRY_FABRICATED_MEASUREMENTS,
    UNEXPECTED_OUTCOME_PREFIX,
    UNRESOLVED_CHALLENGE_SHOULD_REMAIN_OPEN,
    is_legal_ai_incident_failure_id,
    normalize_ai_incident_failure_ids,
)
from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment import (
    COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
    SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
    UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    COMPARISON_CONTENT_ERROR,
    H1_FALLBACK_ERROR,
    H1_NOT_WEAKENED_ERROR,
    H1_ONLY_DIAGNOSIS_ERROR,
    H2_DISPOSITION_ERROR,
    H2_FALLBACK_ERROR,
    H3_FORGED_WITHOUT_TELEMETRY_ERROR,
    MISSING_COMPARISON_ERROR,
    MODEL_SELF_APPROVED_ERROR,
    STALE_STAFFING_ERROR,
    TELEMETRY_CONTENT_ERROR,
    UNRESOLVED_H3_NOT_INSUFFICIENT_ERROR,
    UNRESOLVED_MISSING_TELEMETRY_UNAVAILABLE_ERROR,
    UNSUPPORTED_INFERENCE_ERROR,
)

AI_INCIDENT_EPISTEMIC_FAILURE_ID = UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR
AI_INCIDENT_DIAGNOSTIC_TOOL_TRACE_FAILURE_ID = TOOL_RUNTIME_NOT_EXERCISED


class AiIncidentFailureMappingError(ValueError):
    """Raised when qualification receives an out-of-vocabulary evaluator failure identifier."""


class AiIncidentQualificationInputError(ValueError):
    """Raised when evaluator facts contradict qualification observation semantics."""


class AiIncidentFailureQualificationFamily(IntEnum):
    """Deterministic multi-failure precedence (lower value wins)."""

    INSUFFICIENT_EVIDENCE = 0
    PREMATURE_COMPLETION = 1
    EPISTEMIC = 2
    UNSUPPORTED_COMPLETION = 3
    PLATFORM_ORACLE = 4
    EVALUATOR_CONTRACT = 5
    OBSERVABILITY_ONLY = 6


@dataclass(frozen=True, slots=True)
class AiIncidentFailureQualificationSpec:
    family: AiIncidentFailureQualificationFamily
    behavior_boundary: DecisionFailureBoundary


def _spec(
    family: AiIncidentFailureQualificationFamily,
    behavior_boundary: DecisionFailureBoundary,
) -> AiIncidentFailureQualificationSpec:
    return AiIncidentFailureQualificationSpec(
        family=family,
        behavior_boundary=behavior_boundary,
    )


_INSUFFICIENT = _spec(
    AiIncidentFailureQualificationFamily.INSUFFICIENT_EVIDENCE,
    DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
)
_PREMATURE = _spec(
    AiIncidentFailureQualificationFamily.PREMATURE_COMPLETION,
    DecisionFailureBoundary.COMPLETION_RECONCILIATION,
)
_EPISTEMIC = _spec(
    AiIncidentFailureQualificationFamily.EPISTEMIC,
    DecisionFailureBoundary.COMPLETION_RECONCILIATION,
)
_UNSUPPORTED = _spec(
    AiIncidentFailureQualificationFamily.UNSUPPORTED_COMPLETION,
    DecisionFailureBoundary.REASONING,
)
_PLATFORM = _spec(
    AiIncidentFailureQualificationFamily.PLATFORM_ORACLE,
    DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
)
_EVALUATOR = _spec(
    AiIncidentFailureQualificationFamily.EVALUATOR_CONTRACT,
    DecisionFailureBoundary.EVALUATOR,
)
_OBSERVABILITY = _spec(
    AiIncidentFailureQualificationFamily.OBSERVABILITY_ONLY,
    DecisionFailureBoundary.HOST_EXECUTION,
)

AI_INCIDENT_INSUFFICIENT_EVIDENCE_FAILURE_IDS: frozenset[str] = frozenset(
    {
        STAFFING_ATTENDANCE_NOT_GATHERED,
        STAFFING_PRELIMINARY_NOT_GATHERED,
        COMPARISON_EVIDENCE_NOT_GATHERED,
        TELEMETRY_EVIDENCE_NOT_IN_GRAPH,
        COMPARISON_EVIDENCE_GATHERED_MISSING,
        STAFFING_ATTENDANCE_GATHERED_MISSING,
        TELEMETRY_EVIDENCE_GATHERED_MISSING,
        TELEMETRY_NOT_OBSERVABLE,
        SATISFIED_CHALLENGE_MISSING_RESOLVING_EVIDENCE,
        SATISFIED_CHALLENGE_MISSING_COMPARISON_EVIDENCE,
        SATISFIED_CHALLENGE_MISSING_INITIAL_EVIDENCE,
        SUPPORTED_CLAIM_MISSING_TELEMETRY_REF,
        SUPPORTED_CLAIM_MISSING_COMPARISON_REF,
        H2_REJECTION_MISSING_ATTENDANCE_REF,
        SUPPORTED_DIAGNOSIS_EVIDENCE_NOT_OBSERVABLE,
        H3_DIAGNOSIS_TELEMETRY_NOT_OBSERVABLE,
        MISSING_COMPARISON_ERROR,
    }
)

AI_INCIDENT_REVISION_FLOW_FAILURE_IDS: frozenset[str] = frozenset(
    {
        TELEMETRY_VISIBLE_BEFORE_REVISION,
        TELEMETRY_IN_INITIAL_OBSERVABLE_SET,
        COMPARISON_IN_INITIAL_OBSERVABLE_SET,
        CRITIC_FALSIFICATION_MISSING,
        FAILED_CRITIC_VERDICT_MISSING,
        FAILED_CRITIC_VERDICT_REASON_MISMATCH,
        EVIDENCE_CHALLENGE_MISSING,
        CHALLENGE_TARGET_CLAIM_MISMATCH,
        CHALLENGE_DEFECT_FAMILY_MISMATCH,
        CHALLENGE_DEFECT_CODE_MISMATCH,
        CHALLENGE_NOT_SATISFIED_AFTER_RESOLUTION,
        UNRESOLVED_CHALLENGE_SHOULD_REMAIN_OPEN,
        OPEN_CHALLENGE_MUST_NOT_INCLUDE_RESOLVING_EVIDENCE,
        BOUNDED_RECOVERY_MISSING,
        EVALUATOR_LOOP_BUDGET_EXCEEDED,
        FOLLOW_UP_NOT_VIA_TOOLS,
    }
)

AI_INCIDENT_EPISTEMIC_FAILURE_IDS: frozenset[str] = frozenset(
    {
        UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
        COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
        H2_NOT_REJECTED,
        H1_NOT_WEAKENED,
        H1_NOT_WEAKENED_FROM_RUNTIME,
        H1_STILL_PENDING_AT_END,
        SUPPORTED_DIAGNOSIS_PRESENT,
        H1_FINAL_SUPPORTED_OR_PENDING,
        H2_FINAL_SUPPORTED,
        H3_NOT_INSUFFICIENT_EVIDENCE,
        H3_CLAIM_NOT_INSUFFICIENT,
        COMPARISON_DOES_NOT_WEAKEN_H1_FROM_RUNTIME,
        TELEMETRY_DOES_NOT_SUPPORT_H3_FROM_RUNTIME,
        UNAVAILABLE_TELEMETRY_FABRICATED_MEASUREMENTS,
        TELEMETRY_NOT_UNAVAILABLE,
        STALE_STAFFING_USED_AS_FINAL_SUPPORT,
        STAFFING_PRELIMINARY_SHOULD_BE_STALE_FOR_INCIDENT,
        H1_NOT_PLAUSIBLE_FROM_RUNTIME_EVIDENCE,
        H1_NOT_WEAKENED_ERROR,
        UNRESOLVED_H3_NOT_INSUFFICIENT_ERROR,
    }
)

AI_INCIDENT_UNSUPPORTED_COMPLETION_FAILURE_IDS: frozenset[str] = frozenset(
    {
        CRITIC_CONTENT_VALIDATION_FAILED,
        FINAL_CRITIC_VERDICT_NOT_PASSED,
        FINAL_SUMMARY_NOT_BOUNDED,
        H2_CLAIM_MISSING,
        H3_CLAIM_MISSING,
        H2_DISPOSITION_NOT_DERIVED_FROM_EVIDENCE,
        NO_SUPPORTED_DIAGNOSIS_CLAIM,
        FINAL_SUPPORTED_CLAIM_NOT_H3,
        H3_NOT_DERIVED_FROM_RUNTIME_EVIDENCE,
        MISSING_DIAGNOSIS_CLAIM,
        DIAGNOSIS_CLAIM_NOT_ACCEPTABLE,
        UNSUPPORTED_INFERENCE_ERROR,
        H1_ONLY_DIAGNOSIS_ERROR,
        TELEMETRY_CONTENT_ERROR,
        COMPARISON_CONTENT_ERROR,
        H2_DISPOSITION_ERROR,
        H3_FORGED_WITHOUT_TELEMETRY_ERROR,
        H1_FALLBACK_ERROR,
        H2_FALLBACK_ERROR,
        UNRESOLVED_MISSING_TELEMETRY_UNAVAILABLE_ERROR,
        MODEL_SELF_APPROVED_ERROR,
        STALE_STAFFING_ERROR,
    }
)

AI_INCIDENT_PLATFORM_ORACLE_FAILURE_IDS: frozenset[str] = frozenset(
    {
        FIXTURE_TRUTH_INTEGRITY,
    }
)

AI_INCIDENT_EVALUATOR_CONTRACT_FAILURE_IDS: frozenset[str] = frozenset(
    {
        MISSING_CLAIM_SET,
    }
)

AI_INCIDENT_OBSERVABILITY_ONLY_FAILURE_IDS: frozenset[str] = frozenset(
    {
        TOOL_RUNTIME_NOT_EXERCISED,
    }
)

_STATIC_FAILURE_SPECS: dict[str, AiIncidentFailureQualificationSpec] = {}
for failure_id in AI_INCIDENT_INSUFFICIENT_EVIDENCE_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _INSUFFICIENT
for failure_id in AI_INCIDENT_REVISION_FLOW_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _PREMATURE
for failure_id in AI_INCIDENT_EPISTEMIC_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _EPISTEMIC
for failure_id in AI_INCIDENT_UNSUPPORTED_COMPLETION_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _UNSUPPORTED
for failure_id in AI_INCIDENT_PLATFORM_ORACLE_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _PLATFORM
for failure_id in AI_INCIDENT_EVALUATOR_CONTRACT_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _EVALUATOR
for failure_id in AI_INCIDENT_OBSERVABILITY_ONLY_FAILURE_IDS:
    _STATIC_FAILURE_SPECS[failure_id] = _OBSERVABILITY

_PREFIX_FAILURE_SPECS: dict[str, AiIncidentFailureQualificationSpec] = {
    GROUND_TRUTH_LEAK_PREFIX: _PLATFORM,
    UNEXPECTED_OUTCOME_PREFIX: _EVALUATOR,
    "cited_evidence_not_in_graph:": _UNSUPPORTED,
}

AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS: frozenset[str] = frozenset(_STATIC_FAILURE_SPECS)
AI_INCIDENT_MAPPED_FAILURE_ID_PREFIXES: frozenset[str] = frozenset(_PREFIX_FAILURE_SPECS)


def mapped_ai_incident_failure_ids() -> frozenset[str]:
    """Return all statically mapped failure identifiers (excludes prefix families)."""
    return AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS


def assert_ai_incident_failure_vocabulary_complete() -> None:
    unmapped = AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS - AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS
    if unmapped:
        missing = ", ".join(sorted(unmapped))
        raise AssertionError(f"unmapped legal AI Incident failure identifiers: {missing}")
    stale = AI_INCIDENT_MAPPED_STATIC_FAILURE_IDS - AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS
    if stale:
        extra = ", ".join(sorted(stale))
        raise AssertionError(f"stale mapped AI Incident failure identifiers: {extra}")
    uncovered_prefixes = AI_INCIDENT_EVALUATOR_FAILURE_ID_PREFIXES - AI_INCIDENT_MAPPED_FAILURE_ID_PREFIXES
    if uncovered_prefixes:
        missing = ", ".join(sorted(uncovered_prefixes))
        raise AssertionError(f"unmapped AI Incident failure identifier prefixes: {missing}")


def qualification_spec_for_failure_id(failure_id: str) -> AiIncidentFailureQualificationSpec:
    static = _STATIC_FAILURE_SPECS.get(failure_id)
    if static is not None:
        return static
    for prefix, spec in _PREFIX_FAILURE_SPECS.items():
        if failure_id.startswith(prefix):
            return spec
    raise AiIncidentFailureMappingError(
        f"failure_id={failure_id!r} scenario=ai_incident_investigation mapping_status=unknown"
    )


def _assert_legal_failure_ids(failures: tuple[str, ...]) -> None:
    illegal = tuple(
        failure_id
        for failure_id in failures
        if not is_legal_ai_incident_failure_id(failure_id)
    )
    if illegal:
        joined = ", ".join(illegal)
        raise AiIncidentFailureMappingError(
            "failure_id="
            f"{joined} scenario=ai_incident_investigation mapping_status=contract_violation"
        )


def _validate_evaluator_facts(
    *,
    failures: tuple[str, ...],
    evaluator_passed: bool,
) -> None:
    if evaluator_passed and failures:
        raise AiIncidentQualificationInputError(
            "evaluator_passed=True with non-empty failures is contradictory"
        )


def _select_dominant_family(
    failures: tuple[str, ...],
) -> AiIncidentFailureQualificationSpec | None:
    selected: AiIncidentFailureQualificationSpec | None = None
    for failure_id in failures:
        spec = qualification_spec_for_failure_id(failure_id)
        if selected is None or spec.family.value < selected.family.value:
            selected = spec
    return selected


def build_ai_incident_qualification_signals(
    *,
    failures: tuple[str, ...],
    evaluator_passed: bool,
    trace_finalized: bool = True,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> tuple[
    ModelBehaviorQualificationSignal,
    PlatformContractQualificationSignal,
    EvaluatorQualificationSignal,
    ObservabilityQualificationSignal,
    DecisionFailureBoundary,
]:
    normalized_failures = normalize_ai_incident_failure_ids(failures)
    _validate_evaluator_facts(failures=normalized_failures, evaluator_passed=evaluator_passed)

    if evaluator_passed:
        return (
            ModelBehaviorQualificationSignal(),
            PlatformContractQualificationSignal(trace_finalized=trace_finalized),
            EvaluatorQualificationSignal(passed=True),
            ObservabilityQualificationSignal(),
            boundary,
        )

    if not normalized_failures:
        return (
            ModelBehaviorQualificationSignal(),
            PlatformContractQualificationSignal(trace_finalized=trace_finalized),
            EvaluatorQualificationSignal(passed=False, contract_error=True),
            ObservabilityQualificationSignal(missing_required_signal=True),
            boundary,
        )

    _assert_legal_failure_ids(normalized_failures)
    dominant = _select_dominant_family(normalized_failures)
    assert dominant is not None

    model_behavior = ModelBehaviorQualificationSignal()
    platform_contract = PlatformContractQualificationSignal(trace_finalized=trace_finalized)
    evaluator = EvaluatorQualificationSignal(passed=False)
    observability = ObservabilityQualificationSignal()
    effective_boundary = boundary

    if dominant.family is AiIncidentFailureQualificationFamily.INSUFFICIENT_EVIDENCE:
        model_behavior = ModelBehaviorQualificationSignal(
            insufficient_evidence_gathering=True,
            behavior_boundary=dominant.behavior_boundary,
        )
    elif dominant.family is AiIncidentFailureQualificationFamily.PREMATURE_COMPLETION:
        model_behavior = ModelBehaviorQualificationSignal(
            premature_completion=True,
            behavior_boundary=dominant.behavior_boundary,
        )
    elif dominant.family is AiIncidentFailureQualificationFamily.EPISTEMIC:
        model_behavior = ModelBehaviorQualificationSignal(
            epistemic_contradiction=True,
            behavior_boundary=dominant.behavior_boundary,
        )
    elif dominant.family is AiIncidentFailureQualificationFamily.UNSUPPORTED_COMPLETION:
        model_behavior = ModelBehaviorQualificationSignal(
            unsupported_completion=True,
            behavior_boundary=dominant.behavior_boundary,
        )
    elif dominant.family is AiIncidentFailureQualificationFamily.PLATFORM_ORACLE:
        platform_contract = PlatformContractQualificationSignal(
            trace_finalized=trace_finalized,
            terminal_acceptance_contract_violation=True,
            violation_boundary=dominant.behavior_boundary,
        )
        effective_boundary = dominant.behavior_boundary
    elif dominant.family is AiIncidentFailureQualificationFamily.EVALUATOR_CONTRACT:
        evaluator = EvaluatorQualificationSignal(passed=False, contract_error=True)
        effective_boundary = dominant.behavior_boundary
    elif dominant.family is AiIncidentFailureQualificationFamily.OBSERVABILITY_ONLY:
        pass

    return model_behavior, platform_contract, evaluator, observability, effective_boundary


AI_INCIDENT_FAILURE_QUALIFICATION_MAPPING: MappingProxyType[
    str, AiIncidentFailureQualificationSpec
] = MappingProxyType(_STATIC_FAILURE_SPECS)

assert_ai_incident_failure_vocabulary_complete()
