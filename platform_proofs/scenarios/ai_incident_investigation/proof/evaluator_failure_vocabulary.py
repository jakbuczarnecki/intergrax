# © Artur Czarnecki. All rights reserved.

"""Canonical AI Incident evaluator failure identifier vocabulary (DS-E2E-15J-T1)."""

from __future__ import annotations

from types import MappingProxyType

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

# --- Evaluator structural check failures (resolved path) ---
TELEMETRY_VISIBLE_BEFORE_REVISION = "telemetry_visible_before_revision"
TELEMETRY_IN_INITIAL_OBSERVABLE_SET = "telemetry_in_initial_observable_set"
COMPARISON_IN_INITIAL_OBSERVABLE_SET = "comparison_in_initial_observable_set"
H1_NOT_PLAUSIBLE_FROM_RUNTIME_EVIDENCE = "h1_not_plausible_from_runtime_evidence"
CRITIC_FALSIFICATION_MISSING = "critic_falsification_missing"
FAILED_CRITIC_VERDICT_MISSING = "failed_critic_verdict_missing"
FAILED_CRITIC_VERDICT_REASON_MISMATCH = "failed_critic_verdict_reason_mismatch"
EVIDENCE_CHALLENGE_MISSING = "evidence_challenge_missing"
CHALLENGE_TARGET_CLAIM_MISMATCH = "challenge_target_claim_mismatch"
CHALLENGE_DEFECT_FAMILY_MISMATCH = "challenge_defect_family_mismatch"
CHALLENGE_DEFECT_CODE_MISMATCH = "challenge_defect_code_mismatch"
CHALLENGE_NOT_SATISFIED_AFTER_RESOLUTION = "challenge_not_satisfied_after_resolution"
SATISFIED_CHALLENGE_MISSING_RESOLVING_EVIDENCE = "satisfied_challenge_missing_resolving_evidence"
SATISFIED_CHALLENGE_MISSING_COMPARISON_EVIDENCE = "satisfied_challenge_missing_comparison_evidence"
SATISFIED_CHALLENGE_MISSING_INITIAL_EVIDENCE = "satisfied_challenge_missing_initial_evidence"
TELEMETRY_EVIDENCE_NOT_IN_GRAPH = "telemetry_evidence_not_in_graph"
UNRESOLVED_CHALLENGE_SHOULD_REMAIN_OPEN = "unresolved_challenge_should_remain_open"
OPEN_CHALLENGE_MUST_NOT_INCLUDE_RESOLVING_EVIDENCE = (
    "open_challenge_must_not_include_resolving_evidence"
)
BOUNDED_RECOVERY_MISSING = "bounded_recovery_missing"
EVALUATOR_LOOP_BUDGET_EXCEEDED = "evaluator_loop_budget_exceeded"
FOLLOW_UP_NOT_VIA_TOOLS = "follow_up_not_via_tools"
COMPARISON_EVIDENCE_NOT_GATHERED = "comparison_evidence_not_gathered"
STAFFING_PRELIMINARY_NOT_GATHERED = "staffing_preliminary_not_gathered"
STAFFING_ATTENDANCE_NOT_GATHERED = "staffing_attendance_not_gathered"
STAFFING_PRELIMINARY_SHOULD_BE_STALE_FOR_INCIDENT = (
    "staffing_preliminary_should_be_stale_for_incident"
)
H2_CLAIM_MISSING = "h2_claim_missing"
H2_DISPOSITION_NOT_DERIVED_FROM_EVIDENCE = "h2_disposition_not_derived_from_evidence"
H2_NOT_REJECTED = "h2_not_rejected"
H2_REJECTION_MISSING_ATTENDANCE_REF = "h2_rejection_missing_attendance_ref"
STALE_STAFFING_USED_AS_FINAL_SUPPORT = "stale_staffing_used_as_final_support"
COMPARISON_DOES_NOT_WEAKEN_H1_FROM_RUNTIME = "comparison_does_not_weaken_h1_from_runtime"
TELEMETRY_DOES_NOT_SUPPORT_H3_FROM_RUNTIME = "telemetry_does_not_support_h3_from_runtime"
NO_SUPPORTED_DIAGNOSIS_CLAIM = "no_supported_diagnosis_claim"
FINAL_SUPPORTED_CLAIM_NOT_H3 = "final_supported_claim_not_h3"
H3_NOT_DERIVED_FROM_RUNTIME_EVIDENCE = "h3_not_derived_from_runtime_evidence"
SUPPORTED_CLAIM_MISSING_TELEMETRY_REF = "supported_claim_missing_telemetry_ref"
SUPPORTED_CLAIM_MISSING_COMPARISON_REF = "supported_claim_missing_comparison_ref"
H1_STILL_PENDING_AT_END = "h1_still_pending_at_end"
H1_NOT_WEAKENED = "h1_not_weakened"
H1_NOT_WEAKENED_FROM_RUNTIME = "h1_not_weakened_from_runtime"
CRITIC_CONTENT_VALIDATION_FAILED = "critic_content_validation_failed"
FINAL_CRITIC_VERDICT_NOT_PASSED = "final_critic_verdict_not_passed"
FIXTURE_TRUTH_INTEGRITY = "fixture_truth_integrity"
FINAL_SUMMARY_NOT_BOUNDED = "final_summary_not_bounded"

# --- Evaluator structural check failures (unresolved path) ---
COMPARISON_EVIDENCE_GATHERED_MISSING = "comparison_evidence_gathered_missing"
STAFFING_ATTENDANCE_GATHERED_MISSING = "staffing_attendance_gathered_missing"
TELEMETRY_EVIDENCE_GATHERED_MISSING = "telemetry_evidence_gathered_missing"
TELEMETRY_NOT_OBSERVABLE = "telemetry_not_observable"
TELEMETRY_NOT_UNAVAILABLE = "telemetry_not_unavailable"
UNAVAILABLE_TELEMETRY_FABRICATED_MEASUREMENTS = "unavailable_telemetry_fabricated_measurements"
H1_FINAL_SUPPORTED_OR_PENDING = "h1_final_supported_or_pending"
H2_FINAL_SUPPORTED = "h2_final_supported"
H3_NOT_INSUFFICIENT_EVIDENCE = "h3_not_insufficient_evidence"
SUPPORTED_DIAGNOSIS_PRESENT = "supported_diagnosis_present"
H3_CLAIM_MISSING = "h3_claim_missing"
H3_CLAIM_NOT_INSUFFICIENT = "h3_claim_not_insufficient"

# --- Declared parameterized evaluator failure prefixes ---
CITED_EVIDENCE_NOT_IN_GRAPH_PREFIX = "cited_evidence_not_in_graph:"
UNEXPECTED_OUTCOME_PREFIX = "unexpected_outcome:"
GROUND_TRUTH_LEAK_PREFIX = "ground_truth_leak:"

AI_INCIDENT_EVALUATOR_FAILURE_ID_PREFIXES: frozenset[str] = frozenset(
    {
        CITED_EVIDENCE_NOT_IN_GRAPH_PREFIX,
        UNEXPECTED_OUTCOME_PREFIX,
        GROUND_TRUTH_LEAK_PREFIX,
    }
)

# --- Validation-layer failure identifiers surfaced through qualification ---
MISSING_DIAGNOSIS_CLAIM = "missing_diagnosis_claim"
DIAGNOSIS_CLAIM_NOT_ACCEPTABLE = "diagnosis_claim_not_acceptable"
SUPPORTED_DIAGNOSIS_EVIDENCE_NOT_OBSERVABLE = "supported_diagnosis_evidence_not_observable"
H3_DIAGNOSIS_TELEMETRY_NOT_OBSERVABLE = "h3_diagnosis_telemetry_not_observable"
MISSING_CLAIM_SET = "missing_claim_set"

AI_INCIDENT_VALIDATION_FAILURE_IDS: frozenset[str] = frozenset(
    {
        UNSUPPORTED_INFERENCE_ERROR,
        H1_ONLY_DIAGNOSIS_ERROR,
        MISSING_COMPARISON_ERROR,
        STALE_STAFFING_ERROR,
        TELEMETRY_CONTENT_ERROR,
        COMPARISON_CONTENT_ERROR,
        H2_DISPOSITION_ERROR,
        H1_NOT_WEAKENED_ERROR,
        H3_FORGED_WITHOUT_TELEMETRY_ERROR,
        H1_FALLBACK_ERROR,
        H2_FALLBACK_ERROR,
        UNRESOLVED_MISSING_TELEMETRY_UNAVAILABLE_ERROR,
        UNRESOLVED_H3_NOT_INSUFFICIENT_ERROR,
        MODEL_SELF_APPROVED_ERROR,
        UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR,
        SUPPORTED_DIAGNOSIS_WITHOUT_SUPPORTED_STATE_ERROR,
        COMPLETION_ALIGNMENT_MISMATCH_SIGNAL,
        MISSING_DIAGNOSIS_CLAIM,
        DIAGNOSIS_CLAIM_NOT_ACCEPTABLE,
        SUPPORTED_DIAGNOSIS_EVIDENCE_NOT_OBSERVABLE,
        H3_DIAGNOSIS_TELEMETRY_NOT_OBSERVABLE,
        MISSING_CLAIM_SET,
    }
)

# --- Qualification diagnostic failure (non-authoritative tool-count proxy, DS-E2E-15B.1) ---
TOOL_RUNTIME_NOT_EXERCISED = "tool_runtime_not_exercised"

AI_INCIDENT_EVALUATOR_STATIC_FAILURE_IDS: frozenset[str] = frozenset(
    {
        TELEMETRY_VISIBLE_BEFORE_REVISION,
        TELEMETRY_IN_INITIAL_OBSERVABLE_SET,
        COMPARISON_IN_INITIAL_OBSERVABLE_SET,
        H1_NOT_PLAUSIBLE_FROM_RUNTIME_EVIDENCE,
        CRITIC_FALSIFICATION_MISSING,
        FAILED_CRITIC_VERDICT_MISSING,
        FAILED_CRITIC_VERDICT_REASON_MISMATCH,
        EVIDENCE_CHALLENGE_MISSING,
        CHALLENGE_TARGET_CLAIM_MISMATCH,
        CHALLENGE_DEFECT_FAMILY_MISMATCH,
        CHALLENGE_DEFECT_CODE_MISMATCH,
        CHALLENGE_NOT_SATISFIED_AFTER_RESOLUTION,
        SATISFIED_CHALLENGE_MISSING_RESOLVING_EVIDENCE,
        SATISFIED_CHALLENGE_MISSING_COMPARISON_EVIDENCE,
        SATISFIED_CHALLENGE_MISSING_INITIAL_EVIDENCE,
        TELEMETRY_EVIDENCE_NOT_IN_GRAPH,
        UNRESOLVED_CHALLENGE_SHOULD_REMAIN_OPEN,
        OPEN_CHALLENGE_MUST_NOT_INCLUDE_RESOLVING_EVIDENCE,
        BOUNDED_RECOVERY_MISSING,
        EVALUATOR_LOOP_BUDGET_EXCEEDED,
        FOLLOW_UP_NOT_VIA_TOOLS,
        COMPARISON_EVIDENCE_NOT_GATHERED,
        STAFFING_PRELIMINARY_NOT_GATHERED,
        STAFFING_ATTENDANCE_NOT_GATHERED,
        STAFFING_PRELIMINARY_SHOULD_BE_STALE_FOR_INCIDENT,
        H2_CLAIM_MISSING,
        H2_DISPOSITION_NOT_DERIVED_FROM_EVIDENCE,
        H2_NOT_REJECTED,
        H2_REJECTION_MISSING_ATTENDANCE_REF,
        STALE_STAFFING_USED_AS_FINAL_SUPPORT,
        COMPARISON_DOES_NOT_WEAKEN_H1_FROM_RUNTIME,
        TELEMETRY_DOES_NOT_SUPPORT_H3_FROM_RUNTIME,
        NO_SUPPORTED_DIAGNOSIS_CLAIM,
        FINAL_SUPPORTED_CLAIM_NOT_H3,
        H3_NOT_DERIVED_FROM_RUNTIME_EVIDENCE,
        SUPPORTED_CLAIM_MISSING_TELEMETRY_REF,
        SUPPORTED_CLAIM_MISSING_COMPARISON_REF,
        H1_STILL_PENDING_AT_END,
        H1_NOT_WEAKENED,
        H1_NOT_WEAKENED_FROM_RUNTIME,
        CRITIC_CONTENT_VALIDATION_FAILED,
        FINAL_CRITIC_VERDICT_NOT_PASSED,
        FIXTURE_TRUTH_INTEGRITY,
        FINAL_SUMMARY_NOT_BOUNDED,
        COMPARISON_EVIDENCE_GATHERED_MISSING,
        STAFFING_ATTENDANCE_GATHERED_MISSING,
        TELEMETRY_EVIDENCE_GATHERED_MISSING,
        TELEMETRY_NOT_OBSERVABLE,
        TELEMETRY_NOT_UNAVAILABLE,
        UNAVAILABLE_TELEMETRY_FABRICATED_MEASUREMENTS,
        H1_FINAL_SUPPORTED_OR_PENDING,
        H2_FINAL_SUPPORTED,
        H3_NOT_INSUFFICIENT_EVIDENCE,
        SUPPORTED_DIAGNOSIS_PRESENT,
        H3_CLAIM_MISSING,
        H3_CLAIM_NOT_INSUFFICIENT,
        TOOL_RUNTIME_NOT_EXERCISED,
    }
)

AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS: frozenset[str] = frozenset(
    AI_INCIDENT_EVALUATOR_STATIC_FAILURE_IDS | AI_INCIDENT_VALIDATION_FAILURE_IDS
)


def is_legal_ai_incident_failure_id(failure_id: str) -> bool:
    """Return True when failure_id is in the canonical evaluator vocabulary."""
    if failure_id in AI_INCIDENT_LEGAL_EVALUATOR_FAILURE_IDS:
        return True
    return any(failure_id.startswith(prefix) for prefix in AI_INCIDENT_EVALUATOR_FAILURE_ID_PREFIXES)


def normalize_ai_incident_failure_ids(failures: tuple[str, ...]) -> tuple[str, ...]:
    """Deterministically deduplicate failure identifiers preserving first-seen order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for failure_id in failures:
        if failure_id in seen:
            continue
        seen.add(failure_id)
        normalized.append(failure_id)
    return tuple(normalized)


_AI_INCIDENT_EVALUATOR_STATIC_FAILURE_IDS_VIEW = MappingProxyType(
    dict.fromkeys(sorted(AI_INCIDENT_EVALUATOR_STATIC_FAILURE_IDS), None)
)
