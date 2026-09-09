# © Artur Czarnecki. All rights reserved.

"""Canonical AI Incident semantic evidence requirements (DS-E2E-15B.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from intergrax.contracts.evidence_claims import EvidenceReferenceId, validate_evidence_reference_id
from intergrax.decision_system.evidence_requirements import (
    EvidenceRequirement,
    EvidenceRequirementAssessment,
    EvidenceRequirementCriticality,
    EvidenceRequirementId,
    EvidenceRequirementOutcome,
    EvidenceRequirementSet,
    EvidenceSufficiencyAssessment,
    validate_evidence_requirement_id,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
)

AI_INCIDENT_COMPARISON_EVIDENCE_REQUIREMENT_ID = validate_evidence_requirement_id(
    "ai_incident.comparison_evidence_established"
)
AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID = validate_evidence_requirement_id(
    "ai_incident.staffing_attendance_established"
)
AI_INCIDENT_STAFFING_PRELIMINARY_REQUIREMENT_ID = validate_evidence_requirement_id(
    "ai_incident.staffing_preliminary_established"
)
AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID = validate_evidence_requirement_id(
    "ai_incident.telemetry_evidence_established"
)

AI_INCIDENT_RESOLVED_MANDATORY_EVIDENCE_REQUIREMENTS: tuple[EvidenceRequirement, ...] = (
    EvidenceRequirement(
        requirement_id=AI_INCIDENT_COMPARISON_EVIDENCE_REQUIREMENT_ID,
        criticality=EvidenceRequirementCriticality.MANDATORY,
    ),
    EvidenceRequirement(
        requirement_id=AI_INCIDENT_STAFFING_PRELIMINARY_REQUIREMENT_ID,
        criticality=EvidenceRequirementCriticality.MANDATORY,
    ),
    EvidenceRequirement(
        requirement_id=AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID,
        criticality=EvidenceRequirementCriticality.MANDATORY,
    ),
    EvidenceRequirement(
        requirement_id=AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID,
        criticality=EvidenceRequirementCriticality.MANDATORY,
    ),
)

_REQUIREMENT_TO_CANONICAL_EVIDENCE: dict[EvidenceRequirementId, EvidenceReferenceId] = {
    AI_INCIDENT_COMPARISON_EVIDENCE_REQUIREMENT_ID: COMPARISON_EVIDENCE_ID,
    AI_INCIDENT_STAFFING_PRELIMINARY_REQUIREMENT_ID: STAFFING_PRELIMINARY_EVIDENCE_ID,
    AI_INCIDENT_STAFFING_ATTENDANCE_REQUIREMENT_ID: STAFFING_ATTENDANCE_EVIDENCE_ID,
    AI_INCIDENT_TELEMETRY_EVIDENCE_REQUIREMENT_ID: TELEMETRY_EVIDENCE_ID,
}


@dataclass(frozen=True, slots=True)
class ResolvedIncidentEvidenceRequirementProvider:
    """Provider for mandatory resolved-path incident evidence requirements."""

    def requirements_for(self) -> EvidenceRequirementSet:
        return EvidenceRequirementSet(
            requirements=AI_INCIDENT_RESOLVED_MANDATORY_EVIDENCE_REQUIREMENTS
        )


def observable_evidence_ids_from_nodes(
    evidence_nodes: Sequence[Mapping[str, object]],
) -> frozenset[str]:
    return frozenset(str(node.get("evidence_id")) for node in evidence_nodes)


def project_resolved_incident_evidence_assessment(
    evidence_nodes: Sequence[Mapping[str, object]],
) -> EvidenceSufficiencyAssessment:
    """Map canonical scenario evidence nodes to requirement assessments."""
    observable_ids = observable_evidence_ids_from_nodes(evidence_nodes)
    assessments: list[EvidenceRequirementAssessment] = []
    for requirement in AI_INCIDENT_RESOLVED_MANDATORY_EVIDENCE_REQUIREMENTS:
        canonical_evidence_id = _REQUIREMENT_TO_CANONICAL_EVIDENCE[requirement.requirement_id]
        if str(canonical_evidence_id) in observable_ids:
            assessments.append(
                EvidenceRequirementAssessment(
                    requirement_id=requirement.requirement_id,
                    outcome=EvidenceRequirementOutcome.SATISFIED,
                    supporting_evidence_refs=(validate_evidence_reference_id(canonical_evidence_id),),
                )
            )
        else:
            assessments.append(
                EvidenceRequirementAssessment(
                    requirement_id=requirement.requirement_id,
                    outcome=EvidenceRequirementOutcome.UNSATISFIED,
                )
            )
    return EvidenceSufficiencyAssessment(assessments=tuple(assessments))


RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER = ResolvedIncidentEvidenceRequirementProvider()
