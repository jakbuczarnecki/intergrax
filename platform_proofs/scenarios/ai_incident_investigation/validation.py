# © Artur Czarnecki. All rights reserved.

"""L0 deterministic critic validation for incident investigation claims."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceClaimSet,
)
from intergrax.contracts.validation import ValidationResult
from platform_proofs.scenarios.ai_incident_investigation.domain_reasoning import (
    comparison_weakens_overload,
    derive_hypothesis_dispositions,
    h1_initially_plausible,
    observations_from_evidence_nodes,
    staffing_record_admissible_for_incident,
    telemetry_supports_degradation,
)
from platform_proofs.scenarios.ai_incident_investigation.execution_payload import (
    domain_payload_from_execution,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (
    H2_CLAIM_ID,
    INCIDENT_EVIDENCE_IDS,
    INITIAL_CLAIM_ID,
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


def _observable_evidence_ids(payload: dict[str, object]) -> frozenset[str]:
    raw_nodes = payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return frozenset()
    ids: list[str] = []
    for node in raw_nodes:
        if isinstance(node, dict) and "evidence_id" in node:
            ids.append(str(node["evidence_id"]))
    return frozenset(ids)


def _claim_uses_stale_staffing_as_current_support(
    claim_set: EvidenceClaimSet,
    observations_payload: dict[str, object],
) -> bool:
    raw_nodes = observations_payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return False
    by_id: dict[str, object] = {}
    for node in raw_nodes:
        if isinstance(node, dict) and "evidence_id" in node:
            by_id[str(node["evidence_id"])] = node.get("payload")
    if INCIDENT_EVIDENCE_IDS.staffing_schedule not in by_id:
        return False
    try:
        observations = observations_from_evidence_nodes(
            tuple(raw_nodes),  # type: ignore[arg-type]
            INCIDENT_EVIDENCE_IDS,
        )
    except KeyError:
        return False
    if observations.staffing_schedule is None:
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


def _validate_supported_h3_content(
    claim_set: EvidenceClaimSet,
    domain_payload: dict[str, object],
) -> str | None:
    raw_nodes = domain_payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return "supported_diagnosis_evidence_not_observable"
    try:
        observations = observations_from_evidence_nodes(
            tuple(raw_nodes),  # type: ignore[arg-type]
            INCIDENT_EVIDENCE_IDS,
        )
    except KeyError:
        return "supported_diagnosis_evidence_not_observable"
    if observations.telemetry is None:
        return "supported_diagnosis_missing_telemetry_evidence"
    if not telemetry_supports_degradation(observations.telemetry):
        return TELEMETRY_CONTENT_ERROR
    if observations.comparison is None:
        return MISSING_COMPARISON_ERROR
    if not comparison_weakens_overload(
        observations.workload,
        observations.throughput,
        observations.comparison,
    ):
        return COMPARISON_CONTENT_ERROR
    if not h1_initially_plausible(observations.workload, observations.throughput):
        return H1_ONLY_DIAGNOSIS_ERROR

    h2_claims = [c for c in claim_set.claims if c.claim_id == H2_CLAIM_ID]
    runtime_h2 = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS).h2
    if h2_claims and h2_claims[0].resolution is not runtime_h2.disposition:
        return H2_DISPOSITION_ERROR

    h1_claims = [c for c in claim_set.claims if c.claim_id == INITIAL_CLAIM_ID]
    if h1_claims and h1_claims[0].resolution not in {
        ClaimResolution.SUPERSEDED,
        ClaimResolution.REJECTED,
    }:
        return H1_NOT_WEAKENED_ERROR

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
        latest = supported[-1]
        telemetry_refs = tuple(
            eid for eid in latest.supporting_evidence_ids
            if str(eid).startswith(TELEMETRY_EVIDENCE_PREFIX)
        )
        comparison_refs = tuple(
            eid for eid in latest.supporting_evidence_ids
            if str(eid).startswith(COMPARISON_EVIDENCE_PREFIX)
        )
        if not telemetry_refs:
            return ValidationResult(
                valid=False,
                errors=["supported_diagnosis_missing_telemetry_evidence"],
            )
        if not comparison_refs:
            return ValidationResult(valid=False, errors=[MISSING_COMPARISON_ERROR])
        missing = [
            str(eid)
            for eid in (*telemetry_refs, *comparison_refs)
            if str(eid) not in observable_ids
        ]
        if missing:
            return ValidationResult(
                valid=False,
                errors=["supported_diagnosis_evidence_not_observable"],
            )
        content_error = _validate_supported_h3_content(claim_set, domain_payload)
        if content_error:
            return ValidationResult(valid=False, errors=[content_error])
        return ValidationResult(valid=True)

    active_hypothesis = str(domain_payload.get("active_hypothesis", ""))
    latest = diagnosis_claims[-1]
    telemetry_refs = tuple(
        eid for eid in latest.supporting_evidence_ids
        if str(eid).startswith(TELEMETRY_EVIDENCE_PREFIX)
    )
    comparison_refs = tuple(
        eid for eid in latest.supporting_evidence_ids
        if str(eid).startswith(COMPARISON_EVIDENCE_PREFIX)
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
        if not comparison_refs:
            return ValidationResult(valid=False, errors=[MISSING_COMPARISON_ERROR])
        content_error = _validate_supported_h3_content(claim_set, domain_payload)
        if content_error:
            return ValidationResult(valid=False, errors=[content_error])
        return ValidationResult(valid=True)

    return ValidationResult(valid=False, errors=["diagnosis_claim_not_acceptable"])


class IncidentInvestigationValidationEngine(NexusValidationEngine):
    """Platform L0 critic — rejects H1-only correlation; accepts bounded H3 with telemetry."""

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
