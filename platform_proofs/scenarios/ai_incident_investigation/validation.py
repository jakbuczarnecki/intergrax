# © Artur Czarnecki. All rights reserved.

"""L0 deterministic critic validation for incident investigation claims."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
)
from intergrax.contracts.validation import ValidationResult
from platform_proofs.scenarios.ai_incident_investigation.domain_reasoning import (
    IncidentObservations,
    attendance_meets_required,
    comparison_weakens_overload,
    h1_initially_plausible,
    observations_from_evidence_nodes,
    preliminary_suggests_shortage,
    staffing_record_admissible_for_incident,
    staffing_shortage_confirmed,
    telemetry_is_unavailable,
    telemetry_supports_degradation,
)
from platform_proofs.scenarios.ai_incident_investigation.execution_payload import (
    domain_payload_from_execution,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (
    COMPLETION_UNRESOLVED,
    H2_CLAIM_ID,
    H3_CLAIM_ID,
    INCIDENT_EVIDENCE_IDS,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine

DIAGNOSIS_CLAIM_KIND = "incident.root_cause_diagnosis"
TELEMETRY_EVIDENCE_PREFIX = "evidence.telemetry."
COMPARISON_EVIDENCE_PREFIX = "evidence.comparison."
STAFFING_SCHEDULE_EVIDENCE_PREFIX = "evidence.staffing.schedule."
UNSUPPORTED_INFERENCE_ERROR = (
    "unsupported_inference:missing_distinguishing_equipment_evidence"
)
H1_ONLY_DIAGNOSIS_ERROR = "unsupported_inference:h1_only_causal_diagnosis_insufficient"
MISSING_COMPARISON_ERROR = "unsupported_inference:missing_comparison_evidence"
STALE_STAFFING_ERROR = "admissibility_failure:stale_staffing_used_as_current_support"
TELEMETRY_CONTENT_ERROR = "unsupported_inference:telemetry_content_not_degraded"
COMPARISON_CONTENT_ERROR = "unsupported_inference:comparison_does_not_weaken_overload"
H2_DISPOSITION_ERROR = "unsupported_inference:h2_disposition_incompatible_with_staffing"
H1_NOT_WEAKENED_ERROR = "unsupported_inference:h1_not_weakened_by_comparison"
H3_FORGED_WITHOUT_TELEMETRY_ERROR = "unsupported_inference:h3_supported_without_decisive_telemetry"
H1_FALLBACK_ERROR = "unsupported_inference:h1_fallback_without_distinguishing_evidence"
H2_DISPOSITION_ERROR = "unsupported_inference:h2_disposition_incompatible_with_staffing"
H2_FALLBACK_ERROR = "unsupported_inference:h2_fallback_incompatible_with_staffing"
UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR = "unsupported_inference:unresolved_with_supported_diagnosis"
UNRESOLVED_MISSING_TELEMETRY_UNAVAILABLE_ERROR = (
    "unsupported_inference:unresolved_missing_telemetry_unavailability"
)
UNRESOLVED_H3_NOT_INSUFFICIENT_ERROR = "unsupported_inference:unresolved_h3_not_insufficient"
MODEL_SELF_APPROVED_ERROR = "unsupported_inference:model_self_approved_claim_resolution"


def _observable_evidence_ids(payload: dict[str, object]) -> frozenset[str]:
    raw_nodes = payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return frozenset()
    ids: list[str] = []
    for node in raw_nodes:
        if isinstance(node, dict) and "evidence_id" in node:
            ids.append(str(node["evidence_id"]))
    return frozenset(ids)


def _parse_observations(domain_payload: dict[str, object]) -> IncidentObservations | None:
    raw_nodes = domain_payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return None
    try:
        return observations_from_evidence_nodes(
            tuple(raw_nodes),  # type: ignore[arg-type]
            INCIDENT_EVIDENCE_IDS,
        )
    except (KeyError, ValueError):
        return None


def _claim_uses_stale_staffing_as_current_support(
    claim_set: EvidenceClaimSet,
    observations_payload: dict[str, object],
) -> bool:
    observations = _parse_observations(observations_payload)
    if observations is None or observations.staffing_schedule is None:
        return False
    if staffing_record_admissible_for_incident(observations.staffing_schedule):
        return False
    for claim in claim_set.claims:
        if claim.resolution is not ClaimResolution.SUPPORTED:
            continue
        for evidence_id in claim.supporting_evidence_ids:
            if str(evidence_id).startswith(STAFFING_SCHEDULE_EVIDENCE_PREFIX):
                return True
    return False


def _claim_references_telemetry(claim: EvidenceBackedClaim) -> bool:
    return any(str(eid).startswith(TELEMETRY_EVIDENCE_PREFIX) for eid in claim.supporting_evidence_ids)


def _claim_references_comparison(claim: EvidenceBackedClaim) -> bool:
    return any(str(eid).startswith(COMPARISON_EVIDENCE_PREFIX) for eid in claim.supporting_evidence_ids)


def _h2_claim_supported_by_evidence(observations: IncidentObservations) -> bool:
    schedule = observations.staffing_schedule
    attendance = observations.staffing_attendance
    if schedule is None or attendance is None:
        return False
    return staffing_shortage_confirmed(schedule, attendance)


def _h2_claim_rejected_by_evidence(observations: IncidentObservations) -> bool:
    schedule = observations.staffing_schedule
    attendance = observations.staffing_attendance
    if schedule is None or attendance is None:
        return False
    if attendance_meets_required(schedule, attendance):
        return True
    if preliminary_suggests_shortage(schedule) and staffing_record_admissible_for_incident(schedule):
        return True
    return False


def _h3_claim_supported_by_evidence(observations: IncidentObservations) -> bool:
    if observations.telemetry is None or observations.comparison is None:
        return False
    if not telemetry_supports_degradation(observations.telemetry):
        return False
    return comparison_weakens_overload(
        observations.workload,
        observations.throughput,
        observations.comparison,
    )


def _resolve_h1_claim(
    claim: EvidenceBackedClaim,
    observations: IncidentObservations,
) -> ClaimResolution:
    if not h1_initially_plausible(observations.workload, observations.throughput):
        return ClaimResolution.REJECTED
    if observations.comparison is not None and comparison_weakens_overload(
        observations.workload,
        observations.throughput,
        observations.comparison,
    ):
        return ClaimResolution.SUPERSEDED
    return ClaimResolution.PENDING


def _resolve_h2_claim(observations: IncidentObservations) -> ClaimResolution:
    if _h2_claim_supported_by_evidence(observations):
        return ClaimResolution.SUPPORTED
    if _h2_claim_rejected_by_evidence(observations):
        return ClaimResolution.REJECTED
    return ClaimResolution.PENDING


def _resolve_h3_claim(observations: IncidentObservations) -> ClaimResolution:
    if observations.telemetry is None:
        return ClaimResolution.PENDING
    if telemetry_is_unavailable(observations.telemetry):
        return ClaimResolution.INSUFFICIENT_EVIDENCE
    if _h3_claim_supported_by_evidence(observations):
        return ClaimResolution.SUPPORTED
    return ClaimResolution.INSUFFICIENT_EVIDENCE


def _classify_diagnosis_claim(
    claim: EvidenceBackedClaim,
    observations: IncidentObservations,
) -> ClaimResolution:
    if str(claim.claim_id) == str(H3_CLAIM_ID) or str(claim.claim_id) == str(REVISED_CLAIM_ID):
        return _resolve_h3_claim(observations)
    if str(claim.claim_id) == str(H2_CLAIM_ID) or "H2" in claim.statement:
        return _resolve_h2_claim(observations)
    return _resolve_h1_claim(claim, observations)


def apply_critic_claim_resolutions(
    claim_set: EvidenceClaimSet,
    domain_payload: dict[str, object],
) -> EvidenceClaimSet:
    """Transition model PENDING claims to authoritative critic resolutions."""
    for claim in claim_set.claims:
        if claim.resolution is not ClaimResolution.PENDING:
            raise ValueError(MODEL_SELF_APPROVED_ERROR)

    observations = _parse_observations(domain_payload)
    if observations is None:
        return claim_set

    updated: list[EvidenceBackedClaim] = []
    for claim in claim_set.claims:
        if str(claim.claim_kind) != DIAGNOSIS_CLAIM_KIND:
            updated.append(claim)
            continue
        resolution = _classify_diagnosis_claim(claim, observations)
        if str(claim.claim_id) == str(REVISED_CLAIM_ID) and resolution is ClaimResolution.SUPPORTED:
            resolution = ClaimResolution.SUPPORTED
        updated.append(claim.model_copy(update={"resolution": resolution}))
    return EvidenceClaimSet(claims=tuple(updated), challenges=claim_set.challenges)


def validate_h3_supported_claim(
    claim: EvidenceBackedClaim,
    observations: IncidentObservations,
    observable_ids: frozenset[str],
) -> str | None:
    if not _claim_references_telemetry(claim):
        return H3_FORGED_WITHOUT_TELEMETRY_ERROR
    if not _claim_references_comparison(claim):
        return MISSING_COMPARISON_ERROR
    missing = [
        str(eid)
        for eid in claim.supporting_evidence_ids
        if str(eid) not in observable_ids
    ]
    if missing:
        return "supported_diagnosis_evidence_not_observable"
    if observations.telemetry is None or not telemetry_supports_degradation(observations.telemetry):
        return TELEMETRY_CONTENT_ERROR
    if observations.comparison is None or not comparison_weakens_overload(
        observations.workload,
        observations.throughput,
        observations.comparison,
    ):
        return COMPARISON_CONTENT_ERROR
    if not h1_initially_plausible(observations.workload, observations.throughput):
        return H1_ONLY_DIAGNOSIS_ERROR
    return None


def _validate_unresolved_completion(
    claim_set: EvidenceClaimSet,
    domain_payload: dict[str, object],
) -> str | None:
    observations = _parse_observations(domain_payload)
    if observations is None:
        return "supported_diagnosis_evidence_not_observable"

    diagnosis_claims = [
        claim for claim in claim_set.claims if str(claim.claim_kind) == DIAGNOSIS_CLAIM_KIND
    ]
    supported = [c for c in diagnosis_claims if c.resolution is ClaimResolution.SUPPORTED]
    if supported:
        return UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR

    h1_resolution = _resolve_h1_claim(
        next((c for c in diagnosis_claims if str(c.claim_id) == str(INITIAL_CLAIM_ID)), diagnosis_claims[0]),
        observations,
    )
    h2_resolution = _resolve_h2_claim(observations)
    h3_resolution = _resolve_h3_claim(observations)

    if h1_resolution not in {ClaimResolution.SUPERSEDED, ClaimResolution.REJECTED}:
        return H1_NOT_WEAKENED_ERROR
    if h2_resolution is not ClaimResolution.REJECTED:
        return H2_FALLBACK_ERROR
    if h3_resolution is not ClaimResolution.INSUFFICIENT_EVIDENCE:
        return UNRESOLVED_H3_NOT_INSUFFICIENT_ERROR

    if observations.telemetry is None or not telemetry_is_unavailable(observations.telemetry):
        return UNRESOLVED_MISSING_TELEMETRY_UNAVAILABLE_ERROR

    if observations.comparison is None or not comparison_weakens_overload(
        observations.workload,
        observations.throughput,
        observations.comparison,
    ):
        return COMPARISON_CONTENT_ERROR

    h1_claims = [c for c in claim_set.claims if c.claim_id == INITIAL_CLAIM_ID]
    if h1_claims and h1_claims[0].resolution is ClaimResolution.SUPPORTED:
        return H1_FALLBACK_ERROR

    h2_claims = [c for c in claim_set.claims if c.claim_id == H2_CLAIM_ID]
    if h2_claims and h2_claims[0].resolution is ClaimResolution.SUPPORTED:
        return H2_FALLBACK_ERROR

    if str(domain_payload.get("completion_mode", "")) != COMPLETION_UNRESOLVED:
        return "diagnosis_claim_not_acceptable"

    return None


def validate_claim_set_against_observations(
    claim_set: EvidenceClaimSet,
    domain_payload: dict[str, object],
) -> ValidationResult:
    """Scenario-local independent validation of claim semantics vs observed evidence."""
    observable_ids = _observable_evidence_ids(domain_payload)

    if _claim_uses_stale_staffing_as_current_support(claim_set, domain_payload):
        return ValidationResult(valid=False, errors=[STALE_STAFFING_ERROR])

    diagnosis_claims = [
        claim for claim in claim_set.claims if str(claim.claim_kind) == DIAGNOSIS_CLAIM_KIND
    ]
    if not diagnosis_claims:
        return ValidationResult(valid=False, errors=["missing_diagnosis_claim"])

    supported = [c for c in diagnosis_claims if c.resolution is ClaimResolution.SUPPORTED]
    if supported:
        if str(domain_payload.get("completion_mode", "")) == COMPLETION_UNRESOLVED:
            return ValidationResult(
                valid=False,
                errors=[UNRESOLVED_WITH_SUPPORTED_DIAGNOSIS_ERROR],
            )

        latest = supported[-1]
        observations = _parse_observations(domain_payload)
        h1_claims = [c for c in claim_set.claims if c.claim_id == INITIAL_CLAIM_ID]
        if h1_claims and h1_claims[0].resolution is ClaimResolution.SUPPORTED:
            if observations is not None and observations.comparison is not None:
                if comparison_weakens_overload(
                    observations.workload,
                    observations.throughput,
                    observations.comparison,
                ):
                    return ValidationResult(valid=False, errors=[H1_FALLBACK_ERROR])
            return ValidationResult(valid=False, errors=[H1_FALLBACK_ERROR])

        h2_claims = [c for c in claim_set.claims if c.claim_id == H2_CLAIM_ID]
        if h2_claims and observations is not None:
            h2_expected = _resolve_h2_claim(observations)
            if (
                h2_expected is ClaimResolution.SUPPORTED
                and h2_claims[0].resolution is ClaimResolution.REJECTED
            ):
                return ValidationResult(valid=False, errors=[H2_DISPOSITION_ERROR])
        if h2_claims and h2_claims[0].resolution is ClaimResolution.SUPPORTED:
            if observations is not None and not _h2_claim_supported_by_evidence(observations):
                return ValidationResult(valid=False, errors=[H2_FALLBACK_ERROR])
            return ValidationResult(valid=False, errors=[H2_FALLBACK_ERROR])

        if observations is None:
            return ValidationResult(
                valid=False,
                errors=["supported_diagnosis_evidence_not_observable"],
            )
        content_error = validate_h3_supported_claim(latest, observations, observable_ids)
        if content_error:
            return ValidationResult(valid=False, errors=[content_error])
        return ValidationResult(valid=True)

    completion_mode = str(domain_payload.get("completion_mode", ""))
    if completion_mode == COMPLETION_UNRESOLVED:
        unresolved_error = _validate_unresolved_completion(claim_set, domain_payload)
        if unresolved_error:
            return ValidationResult(valid=False, errors=[unresolved_error])
        return ValidationResult(valid=True)

    active_hypothesis = str(domain_payload.get("active_hypothesis", ""))
    latest = diagnosis_claims[-1]
    telemetry_refs = tuple(
        eid for eid in latest.supporting_evidence_ids
        if str(eid).startswith(TELEMETRY_EVIDENCE_PREFIX)
    )

    if active_hypothesis == "H1" and not telemetry_refs:
        return ValidationResult(valid=False, errors=[UNSUPPORTED_INFERENCE_ERROR])

    if active_hypothesis == "H3" and telemetry_refs:
        missing = [str(eid) for eid in telemetry_refs if str(eid) not in observable_ids]
        if missing:
            return ValidationResult(
                valid=False,
                errors=["h3_diagnosis_telemetry_not_observable"],
            )
        observations = _parse_observations(domain_payload)
        if observations is None:
            return ValidationResult(
                valid=False,
                errors=["supported_diagnosis_evidence_not_observable"],
            )
        content_error = validate_h3_supported_claim(latest, observations, observable_ids)
        if content_error:
            return ValidationResult(valid=False, errors=[content_error])
        return ValidationResult(valid=True)

    return ValidationResult(valid=False, errors=["diagnosis_claim_not_acceptable"])


class IncidentInvestigationValidationEngine(NexusValidationEngine):
    """Platform L0 critic — validates model claims via bounded domain predicates."""

    def validate(
        self,
        execution: AgentExecutionResult,
        *,
        contract: AgentContract,
        capability: str | None = None,
        plan_criteria: list[str] | None = None,
    ) -> ValidationResult:
        base = super().validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )
        if not base.valid:
            return base

        domain_payload = domain_payload_from_execution(execution)
        raw_claim_set = domain_payload.get("claim_set")
        if raw_claim_set is None:
            return ValidationResult(
                valid=False,
                errors=["missing_claim_set"],
                warnings=list(base.warnings),
            )

        claim_set = EvidenceClaimSet.model_validate(raw_claim_set)
        content_result = validate_claim_set_against_observations(claim_set, domain_payload)
        if not content_result.valid:
            return ValidationResult(
                valid=False,
                errors=list(content_result.errors),
                warnings=list(base.warnings),
            )
        return ValidationResult(valid=True, warnings=list(base.warnings))
