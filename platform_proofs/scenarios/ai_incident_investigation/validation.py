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
from platform_proofs.scenarios.ai_incident_investigation.execution_payload import (
    domain_payload_from_execution,
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


def _observable_evidence_ids(payload: dict[str, object]) -> frozenset[str]:
    raw_nodes = payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return frozenset()
    ids: list[str] = []
    for node in raw_nodes:
        if isinstance(node, dict) and "evidence_id" in node:
            ids.append(str(node["evidence_id"]))
    return frozenset(ids)


def _staffing_schedule_admissible_for_incident(payload: dict[str, object]) -> bool:
    """Scenario-local rule: roster export valid window must overlap incident window."""
    raw_nodes = payload.get("evidence_nodes")
    if not isinstance(raw_nodes, list):
        return False
    for node in raw_nodes:
        if not isinstance(node, dict):
            continue
        if not str(node.get("evidence_id", "")).startswith(STAFFING_SCHEDULE_EVIDENCE_PREFIX):
            continue
        node_payload = node.get("payload")
        if not isinstance(node_payload, dict):
            return False
        valid_from = node_payload.get("record_valid_from", "")
        valid_to = node_payload.get("record_valid_to", "")
        window_from = node_payload.get("window_observed_from", "")
        window_to = node_payload.get("window_observed_to", "")
        if not valid_from or not valid_to or not window_from or not window_to:
            return False
        if valid_to < window_from:
            return False
        if valid_from > window_to:
            return False
        return True
    return False


def _claim_uses_stale_staffing_as_current_support(
    claim_set: EvidenceClaimSet,
    payload: dict[str, object],
) -> bool:
    if _staffing_schedule_admissible_for_incident(payload):
        return False
    for claim in claim_set.claims:
        if claim.resolution is not ClaimResolution.SUPPORTED:
            continue
        for evidence_id in claim.supporting_evidence_ids:
            if str(evidence_id).startswith(STAFFING_SCHEDULE_EVIDENCE_PREFIX):
                return True
    return False


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
        diagnosis_claims = [
            claim
            for claim in claim_set.claims
            if str(claim.claim_kind) == DIAGNOSIS_CLAIM_KIND
        ]
        if not diagnosis_claims:
            return ValidationResult(
                valid=False,
                errors=["missing_diagnosis_claim"],
                warnings=list(base.warnings),
            )

        observable_ids = _observable_evidence_ids(domain_payload)
        latest = diagnosis_claims[-1]
        telemetry_refs = tuple(
            evidence_id
            for evidence_id in latest.supporting_evidence_ids
            if str(evidence_id).startswith(TELEMETRY_EVIDENCE_PREFIX)
        )
        comparison_refs = tuple(
            evidence_id
            for evidence_id in latest.supporting_evidence_ids
            if str(evidence_id).startswith(COMPARISON_EVIDENCE_PREFIX)
        )

        if _claim_uses_stale_staffing_as_current_support(claim_set, domain_payload):
            return ValidationResult(
                valid=False,
                errors=[STALE_STAFFING_ERROR],
                warnings=list(base.warnings),
            )

        if latest.resolution is ClaimResolution.SUPPORTED:
            if not telemetry_refs:
                return ValidationResult(
                    valid=False,
                    errors=["supported_diagnosis_missing_telemetry_evidence"],
                    warnings=list(base.warnings),
                )
            if not comparison_refs:
                return ValidationResult(
                    valid=False,
                    errors=[MISSING_COMPARISON_ERROR],
                    warnings=list(base.warnings),
                )
            missing_observable = [
                str(evidence_id)
                for evidence_id in (*telemetry_refs, *comparison_refs)
                if str(evidence_id) not in observable_ids
            ]
            if missing_observable:
                return ValidationResult(
                    valid=False,
                    errors=["supported_diagnosis_evidence_not_observable"],
                    warnings=list(base.warnings),
                )
            return ValidationResult(valid=True, warnings=list(base.warnings))

        active_hypothesis = str(domain_payload.get("active_hypothesis", ""))
        if active_hypothesis == "H1" and not telemetry_refs:
            return ValidationResult(
                valid=False,
                errors=[UNSUPPORTED_INFERENCE_ERROR],
                warnings=list(base.warnings),
            )

        if active_hypothesis == "H3" and telemetry_refs:
            missing_observable = [
                str(evidence_id)
                for evidence_id in telemetry_refs
                if str(evidence_id) not in observable_ids
            ]
            if missing_observable:
                return ValidationResult(
                    valid=False,
                    errors=["h3_diagnosis_telemetry_not_observable"],
                    warnings=list(base.warnings),
                )
            if not comparison_refs:
                return ValidationResult(
                    valid=False,
                    errors=[MISSING_COMPARISON_ERROR],
                    warnings=list(base.warnings),
                )
            return ValidationResult(valid=True, warnings=list(base.warnings))

        return ValidationResult(
            valid=False,
            errors=["diagnosis_claim_not_acceptable"],
            warnings=list(base.warnings),
        )
