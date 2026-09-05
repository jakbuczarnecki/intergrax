# © Artur Czarnecki. All rights reserved.

"""Scenario-local evidence and claim identifiers — shared without import cycles."""

from __future__ import annotations

from intergrax.contracts.evidence_claims import (
    validate_claim_kind,
    validate_evidence_claim_id,
    validate_evidence_reference_id,
)
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import IncidentEvidenceIds

DIAGNOSIS_CLAIM_KIND = "incident.root_cause_diagnosis"

INITIAL_CLAIM_ID = validate_evidence_claim_id("eclaim_a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1")
H2_CLAIM_ID = validate_evidence_claim_id("eclaim_c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3")
H3_CLAIM_ID = validate_evidence_claim_id("eclaim_d4d4d4d4d4d4d4d4d4d4d4d4d4d4d4d4")
REVISED_CLAIM_ID = validate_evidence_claim_id("eclaim_b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2")
WORKLOAD_EVIDENCE_ID = validate_evidence_reference_id("evidence.workload.line4.incident_window")
THROUGHPUT_EVIDENCE_ID = validate_evidence_reference_id("evidence.throughput.line4.incident_window")
STAFFING_PRELIMINARY_EVIDENCE_ID = validate_evidence_reference_id(
    "evidence.staffing.schedule.line4.incident_window"
)
STAFFING_ATTENDANCE_EVIDENCE_ID = validate_evidence_reference_id(
    "evidence.staffing.attendance.line4.incident_window"
)
COMPARISON_EVIDENCE_ID = validate_evidence_reference_id(
    "evidence.comparison.line3.high_load_window"
)
TELEMETRY_EVIDENCE_ID = validate_evidence_reference_id(
    "evidence.telemetry.complex_assembly_station.incident_window"
)
DIAGNOSIS_KIND = validate_claim_kind(DIAGNOSIS_CLAIM_KIND)

COMPLETION_SUPPORTED_DIAGNOSIS = "supported_diagnosis"
COMPLETION_UNRESOLVED = "unresolved"
COMPLETION_NEED_MORE_EVIDENCE = "need_more_evidence"

INCIDENT_EVIDENCE_IDS = IncidentEvidenceIds(
    workload=str(WORKLOAD_EVIDENCE_ID),
    throughput=str(THROUGHPUT_EVIDENCE_ID),
    staffing_schedule=str(STAFFING_PRELIMINARY_EVIDENCE_ID),
    staffing_attendance=str(STAFFING_ATTENDANCE_EVIDENCE_ID),
    comparison=str(COMPARISON_EVIDENCE_ID),
    telemetry=str(TELEMETRY_EVIDENCE_ID),
)
