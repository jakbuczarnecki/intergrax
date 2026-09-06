# © Artur Czarnecki. All rights reserved.

"""Scenario-local evidence interpretation — pure domain predicates and disposition derivation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from intergrax.contracts.evidence_claims import ClaimResolution
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    HypothesisId,
    TelemetryAvailability,
)

# --- Scenario-domain thresholds (bounded, not fixture-overfit) ---

HIGH_WORKLOAD_DELTA_PCT = 15.0
MATERIAL_THROUGHPUT_DROP_PCT = 10.0
WORKLOAD_COMPARISON_TOLERANCE_PCT = 5.0
COMPARISON_HEALTHY_MARGIN_PCT = 8.0
MATERIAL_STATION_THROUGHPUT_DROP_PCT = 15.0

DEGRADED_SIGNAL_STATES: frozenset[str] = frozenset(
    {"intermittent_degraded", "degraded", "fault", "intermittent_fault"}
)
HEALTHY_SIGNAL_STATES: frozenset[str] = frozenset({"healthy", "nominal", "ok"})


class RationaleCode(StrEnum):
    H1_PLAUSIBLE_CORRELATION = "h1_plausible_correlation"
    H1_SUPERSEDED_BY_COMPARISON = "h1_superseded_by_comparison"
    H1_PENDING_INSUFFICIENT = "h1_pending_insufficient"
    H1_NOT_PLAUSIBLE = "h1_not_plausible"
    H2_REJECTED_ATTENDANCE_MEETS_REQUIRED = "h2_rejected_attendance_meets_required"
    H2_REJECTED_NO_SHORTAGE = "h2_rejected_no_shortage"
    H2_SUPPORTED_SHORTAGE_CONFIRMED = "h2_supported_shortage_confirmed"
    H2_PENDING_STALE_SCHEDULE = "h2_pending_stale_schedule"
    H2_PENDING_AWAITING_ATTENDANCE = "h2_pending_awaiting_attendance"
    H2_INSUFFICIENT_EVIDENCE = "h2_insufficient_evidence"
    H3_SUPPORTED_DEGRADATION = "h3_supported_degradation"
    H3_PENDING_AWAITING_TELEMETRY = "h3_pending_awaiting_telemetry"
    H3_INSUFFICIENT_NO_DISTINGUISHING = "h3_insufficient_no_distinguishing"
    H3_INSUFFICIENT_NO_DEGRADATION = "h3_insufficient_no_degradation"
    H3_INSUFFICIENT_TELEMETRY_UNAVAILABLE = "h3_insufficient_telemetry_unavailable"


@dataclass(frozen=True, slots=True)
class ObservedWorkload:
    order_volume_delta_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class ObservedThroughput:
    target_attainment_pct: float
    baseline_attainment_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class ObservedStaffingSchedule:
    scheduled_headcount: int
    required_headcount: int
    record_valid_from: str
    record_valid_to: str
    window_observed_from: str
    window_observed_to: str


@dataclass(frozen=True, slots=True)
class ObservedStaffingAttendance:
    confirmed_headcount: int


@dataclass(frozen=True, slots=True)
class ObservedComparison:
    workload_delta_pct: float
    comparison_attainment_pct: float
    reference_attainment_pct: float
    admissible: bool


@dataclass(frozen=True, slots=True)
class ObservedTelemetry:
    availability: TelemetryAvailability
    signal_state: str | None = None
    complex_assembly_throughput_pct: float | None = None
    baseline_throughput_pct: float | None = None
    admissible: bool = True
    unavailability_reason: str | None = None


@dataclass(frozen=True, slots=True)
class IncidentEvidenceIds:
    workload: str
    throughput: str
    staffing_schedule: str
    staffing_attendance: str
    comparison: str
    telemetry: str


@dataclass(frozen=True, slots=True)
class IncidentObservations:
    workload: ObservedWorkload
    throughput: ObservedThroughput
    staffing_schedule: ObservedStaffingSchedule | None = None
    staffing_attendance: ObservedStaffingAttendance | None = None
    comparison: ObservedComparison | None = None
    telemetry: ObservedTelemetry | None = None


@dataclass(frozen=True, slots=True)
class HypothesisAssessment:
    hypothesis_id: HypothesisId
    disposition: ClaimResolution
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]
    rationale_code: RationaleCode


@dataclass(frozen=True, slots=True)
class IncidentAssessment:
    h1: HypothesisAssessment
    h2: HypothesisAssessment
    h3: HypothesisAssessment
    active_hypothesis: HypothesisId
    summary: str


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def is_high_workload(workload: ObservedWorkload) -> bool:
    return workload.admissible and workload.order_volume_delta_pct >= HIGH_WORKLOAD_DELTA_PCT


def is_material_throughput_drop(throughput: ObservedThroughput) -> bool:
    if not throughput.admissible:
        return False
    drop = throughput.baseline_attainment_pct - throughput.target_attainment_pct
    return drop >= MATERIAL_THROUGHPUT_DROP_PCT


def h1_initially_plausible(workload: ObservedWorkload, throughput: ObservedThroughput) -> bool:
    return is_high_workload(workload) and is_material_throughput_drop(throughput)


def staffing_record_admissible_for_incident(schedule: ObservedStaffingSchedule) -> bool:
    valid_from = _parse_iso(schedule.record_valid_from)
    valid_to = _parse_iso(schedule.record_valid_to)
    window_from = _parse_iso(schedule.window_observed_from)
    window_to = _parse_iso(schedule.window_observed_to)
    if valid_from is None or valid_to is None or window_from is None or window_to is None:
        return False
    if valid_to < window_from:
        return False
    if valid_from > window_to:
        return False
    return True


def preliminary_suggests_shortage(schedule: ObservedStaffingSchedule) -> bool:
    return schedule.scheduled_headcount < schedule.required_headcount


def staffing_shortage_confirmed(
    schedule: ObservedStaffingSchedule,
    attendance: ObservedStaffingAttendance,
) -> bool:
    return attendance.confirmed_headcount < schedule.required_headcount


def attendance_meets_required(
    schedule: ObservedStaffingSchedule,
    attendance: ObservedStaffingAttendance,
) -> bool:
    return attendance.confirmed_headcount >= schedule.required_headcount


def comparison_weakens_overload(
    workload: ObservedWorkload,
    throughput: ObservedThroughput,
    comparison: ObservedComparison,
) -> bool:
    if not comparison.admissible or not workload.admissible:
        return False
    similar_load = (
        comparison.workload_delta_pct
        >= workload.order_volume_delta_pct - WORKLOAD_COMPARISON_TOLERANCE_PCT
    )
    healthier_peer = (
        comparison.comparison_attainment_pct
        >= throughput.target_attainment_pct + COMPARISON_HEALTHY_MARGIN_PCT
    )
    return similar_load and healthier_peer


def telemetry_is_unavailable(telemetry: ObservedTelemetry) -> bool:
    return telemetry.availability is TelemetryAvailability.UNAVAILABLE


def telemetry_supports_degradation(telemetry: ObservedTelemetry) -> bool:
    if telemetry.availability is not TelemetryAvailability.AVAILABLE:
        return False
    if not telemetry.admissible:
        return False
    if telemetry.signal_state not in DEGRADED_SIGNAL_STATES:
        return False
    if (
        telemetry.complex_assembly_throughput_pct is None
        or telemetry.baseline_throughput_pct is None
    ):
        return False
    drop = telemetry.baseline_throughput_pct - telemetry.complex_assembly_throughput_pct
    return drop >= MATERIAL_STATION_THROUGHPUT_DROP_PCT


def telemetry_is_healthy(telemetry: ObservedTelemetry) -> bool:
    if telemetry.availability is not TelemetryAvailability.AVAILABLE:
        return False
    if telemetry.signal_state in HEALTHY_SIGNAL_STATES:
        if (
            telemetry.complex_assembly_throughput_pct is None
            or telemetry.baseline_throughput_pct is None
        ):
            return True
        drop = telemetry.baseline_throughput_pct - telemetry.complex_assembly_throughput_pct
        return drop < MATERIAL_STATION_THROUGHPUT_DROP_PCT
    return False


def _assess_h1(
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
) -> HypothesisAssessment:
    workload = observations.workload
    throughput = observations.throughput
    support = (evidence_ids.workload, evidence_ids.throughput)
    if not h1_initially_plausible(workload, throughput):
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H1,
            disposition=ClaimResolution.REJECTED,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=support,
            rationale_code=RationaleCode.H1_NOT_PLAUSIBLE,
        )
    if observations.comparison is not None and comparison_weakens_overload(
        workload, throughput, observations.comparison
    ):
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H1,
            disposition=ClaimResolution.SUPERSEDED,
            supporting_evidence_ids=support,
            contradicting_evidence_ids=(evidence_ids.comparison,),
            rationale_code=RationaleCode.H1_SUPERSEDED_BY_COMPARISON,
        )
    return HypothesisAssessment(
        hypothesis_id=HypothesisId.H1,
        disposition=ClaimResolution.PENDING,
        supporting_evidence_ids=support,
        contradicting_evidence_ids=(),
        rationale_code=RationaleCode.H1_PLAUSIBLE_CORRELATION,
    )


def _assess_h2(
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
) -> HypothesisAssessment:
    schedule = observations.staffing_schedule
    attendance = observations.staffing_attendance
    schedule_id = evidence_ids.staffing_schedule
    attendance_id = evidence_ids.staffing_attendance

    if schedule is None:
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H2,
            disposition=ClaimResolution.INSUFFICIENT_EVIDENCE,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(),
            rationale_code=RationaleCode.H2_INSUFFICIENT_EVIDENCE,
        )

    if attendance is None:
        if preliminary_suggests_shortage(schedule) and staffing_record_admissible_for_incident(
            schedule
        ):
            return HypothesisAssessment(
                hypothesis_id=HypothesisId.H2,
                disposition=ClaimResolution.PENDING,
                supporting_evidence_ids=(schedule_id,),
                contradicting_evidence_ids=(),
                rationale_code=RationaleCode.H2_PENDING_AWAITING_ATTENDANCE,
            )
        if preliminary_suggests_shortage(schedule):
            return HypothesisAssessment(
                hypothesis_id=HypothesisId.H2,
                disposition=ClaimResolution.PENDING,
                supporting_evidence_ids=(schedule_id,),
                contradicting_evidence_ids=(),
                rationale_code=RationaleCode.H2_PENDING_STALE_SCHEDULE,
            )
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H2,
            disposition=ClaimResolution.INSUFFICIENT_EVIDENCE,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(),
            rationale_code=RationaleCode.H2_INSUFFICIENT_EVIDENCE,
        )

    if attendance_meets_required(schedule, attendance):
        support: tuple[str, ...] = ()
        contradict: tuple[str, ...] = ()
        rationale = RationaleCode.H2_REJECTED_NO_SHORTAGE
        if preliminary_suggests_shortage(schedule):
            support = (schedule_id,)
            contradict = (attendance_id,)
            rationale = RationaleCode.H2_REJECTED_ATTENDANCE_MEETS_REQUIRED
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H2,
            disposition=ClaimResolution.REJECTED,
            supporting_evidence_ids=support,
            contradicting_evidence_ids=contradict,
            rationale_code=rationale,
        )

    if staffing_shortage_confirmed(schedule, attendance):
        support_ids: tuple[str, ...] = (attendance_id,)
        if staffing_record_admissible_for_incident(schedule) and preliminary_suggests_shortage(
            schedule
        ):
            support_ids = (schedule_id, attendance_id)
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H2,
            disposition=ClaimResolution.SUPPORTED,
            supporting_evidence_ids=support_ids,
            contradicting_evidence_ids=(),
            rationale_code=RationaleCode.H2_SUPPORTED_SHORTAGE_CONFIRMED,
        )

    return HypothesisAssessment(
        hypothesis_id=HypothesisId.H2,
        disposition=ClaimResolution.INSUFFICIENT_EVIDENCE,
        supporting_evidence_ids=(),
        contradicting_evidence_ids=(),
        rationale_code=RationaleCode.H2_INSUFFICIENT_EVIDENCE,
    )


def _assess_h3(
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
    h1: HypothesisAssessment,
) -> HypothesisAssessment:
    workload = observations.workload
    throughput = observations.throughput
    comparison = observations.comparison
    telemetry = observations.telemetry

    if telemetry is None:
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H3,
            disposition=ClaimResolution.PENDING,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(),
            rationale_code=RationaleCode.H3_PENDING_AWAITING_TELEMETRY,
        )

    if telemetry_is_unavailable(telemetry):
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H3,
            disposition=ClaimResolution.INSUFFICIENT_EVIDENCE,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(),
            rationale_code=RationaleCode.H3_INSUFFICIENT_TELEMETRY_UNAVAILABLE,
        )

    if not telemetry_supports_degradation(telemetry):
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H3,
            disposition=ClaimResolution.INSUFFICIENT_EVIDENCE,
            supporting_evidence_ids=(),
            contradicting_evidence_ids=(evidence_ids.telemetry,),
            rationale_code=RationaleCode.H3_INSUFFICIENT_NO_DEGRADATION,
        )

    if comparison is None or not comparison_weakens_overload(workload, throughput, comparison):
        return HypothesisAssessment(
            hypothesis_id=HypothesisId.H3,
            disposition=ClaimResolution.INSUFFICIENT_EVIDENCE,
            supporting_evidence_ids=(evidence_ids.telemetry,),
            contradicting_evidence_ids=(),
            rationale_code=RationaleCode.H3_INSUFFICIENT_NO_DISTINGUISHING,
        )

    support = (
        evidence_ids.workload,
        evidence_ids.throughput,
        evidence_ids.comparison,
        evidence_ids.telemetry,
    )
    return HypothesisAssessment(
        hypothesis_id=HypothesisId.H3,
        disposition=ClaimResolution.SUPPORTED,
        supporting_evidence_ids=support,
        contradicting_evidence_ids=(),
        rationale_code=RationaleCode.H3_SUPPORTED_DEGRADATION,
    )


def derive_hypothesis_dispositions(
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
) -> IncidentAssessment:
    """Derive H1/H2/H3 dispositions from observed tool-result data only."""
    h1 = _assess_h1(observations, evidence_ids)
    h2 = _assess_h2(observations, evidence_ids)
    h3 = _assess_h3(observations, evidence_ids, h1)

    if h3.disposition is ClaimResolution.SUPPORTED:
        active = HypothesisId.H3
        summary = (
            "bounded equipment-process degradation diagnosis supported "
            "by telemetry and comparison evidence"
        )
    elif (
        h3.disposition is ClaimResolution.INSUFFICIENT_EVIDENCE
        and h3.rationale_code is RationaleCode.H3_INSUFFICIENT_TELEMETRY_UNAVAILABLE
        and h1.disposition in {ClaimResolution.SUPERSEDED, ClaimResolution.REJECTED}
        and h2.disposition is ClaimResolution.REJECTED
    ):
        active = HypothesisId.H3
        summary = (
            "Investigation remains unresolved: workload-only and staffing explanations "
            "are not supported, while the equipment hypothesis cannot be accepted "
            "because decisive telemetry for the incident window is unavailable."
        )
    elif h1.disposition is ClaimResolution.PENDING:
        active = HypothesisId.H1
        summary = "draft: workload overload candidate diagnosis hypothesis H1"
    elif h2.disposition is ClaimResolution.SUPPORTED:
        active = HypothesisId.H2
        summary = "understaffing hypothesis H2 supported by confirmed attendance evidence"
    else:
        active = HypothesisId.H1
        summary = "incident investigation in progress — no supported final diagnosis"

    return IncidentAssessment(h1=h1, h2=h2, h3=h3, active_hypothesis=active, summary=summary)


def hypothesis_evidence_relations(
    hypothesis_id: HypothesisId,
    observations: IncidentObservations,
    evidence_ids: IncidentEvidenceIds,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministic supporting/contradicting evidence ID tuples for one hypothesis."""
    h1 = _assess_h1(observations, evidence_ids)
    if hypothesis_id is HypothesisId.H1:
        return h1.supporting_evidence_ids, h1.contradicting_evidence_ids
    if hypothesis_id is HypothesisId.H2:
        h2 = _assess_h2(observations, evidence_ids)
        return h2.supporting_evidence_ids, h2.contradicting_evidence_ids
    h3 = _assess_h3(observations, evidence_ids, h1)
    return h3.supporting_evidence_ids, h3.contradicting_evidence_ids


def normalize_tool_payload(payload: object) -> dict[str, object]:
    """Convert ToolRuntime preview transport into a mapping for typed parsing."""
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    if isinstance(payload, str):
        text = payload.rstrip("…")
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("invalid tool payload transport")


def parse_workload_payload(payload: object) -> ObservedWorkload:
    data = normalize_tool_payload(payload)
    return ObservedWorkload(
        order_volume_delta_pct=float(data.get("order_volume_delta_pct", 0.0)),
        admissible=bool(data.get("admissible", False)),
    )


def parse_throughput_payload(payload: object) -> ObservedThroughput:
    data = normalize_tool_payload(payload)
    return ObservedThroughput(
        target_attainment_pct=float(data.get("target_attainment_pct", 0.0)),
        baseline_attainment_pct=float(data.get("baseline_attainment_pct", 0.0)),
        admissible=bool(data.get("admissible", False)),
    )


def parse_staffing_schedule_payload(payload: object) -> ObservedStaffingSchedule:
    data = normalize_tool_payload(payload)
    return ObservedStaffingSchedule(
        scheduled_headcount=int(data.get("scheduled_headcount", 0)),
        required_headcount=int(data.get("required_headcount", 0)),
        record_valid_from=str(data.get("record_valid_from", "")),
        record_valid_to=str(data.get("record_valid_to", "")),
        window_observed_from=str(data.get("window_observed_from", "")),
        window_observed_to=str(data.get("window_observed_to", "")),
    )


def parse_staffing_attendance_payload(payload: object) -> ObservedStaffingAttendance:
    data = normalize_tool_payload(payload)
    return ObservedStaffingAttendance(
        confirmed_headcount=int(data.get("confirmed_headcount", 0)),
    )


def parse_comparison_payload(payload: object) -> ObservedComparison:
    data = normalize_tool_payload(payload)
    return ObservedComparison(
        workload_delta_pct=float(data.get("workload_delta_pct", 0.0)),
        comparison_attainment_pct=float(data.get("comparison_attainment_pct", 0.0)),
        reference_attainment_pct=float(data.get("reference_attainment_pct", 0.0)),
        admissible=bool(data.get("admissible", False)),
    )


def parse_telemetry_payload(payload: object) -> ObservedTelemetry:
    data = normalize_tool_payload(payload)
    availability_raw = str(data.get("availability", "")).lower()
    if availability_raw == TelemetryAvailability.UNAVAILABLE.value:
        return ObservedTelemetry(
            availability=TelemetryAvailability.UNAVAILABLE,
            admissible=bool(data.get("admissible", False)),
            unavailability_reason=(
                str(data["unavailability_reason"])
                if data.get("unavailability_reason") is not None
                else None
            ),
        )
    if availability_raw == TelemetryAvailability.AVAILABLE.value:
        signal_state = data.get("signal_state")
        throughput = data.get("complex_assembly_throughput_pct")
        baseline = data.get("baseline_throughput_pct")
        if signal_state is None or throughput is None or baseline is None:
            raise ValueError("available telemetry missing required measurement fields")
        return ObservedTelemetry(
            availability=TelemetryAvailability.AVAILABLE,
            signal_state=str(signal_state),
            complex_assembly_throughput_pct=float(throughput),
            baseline_throughput_pct=float(baseline),
            admissible=bool(data.get("admissible", False)),
        )
    raise ValueError("telemetry payload missing or invalid availability")


def observations_from_evidence_nodes(
    nodes: tuple[dict[str, object], ...] | list[dict[str, object]],
    evidence_ids: IncidentEvidenceIds,
) -> IncidentObservations:
    """Reconstruct typed observations from runtime evidence graph nodes."""
    by_id: dict[str, object] = {}
    for node in nodes:
        if "evidence_id" in node:
            by_id[str(node["evidence_id"])] = node.get("payload")

    workload_raw = by_id.get(evidence_ids.workload)
    throughput_raw = by_id.get(evidence_ids.throughput)
    if workload_raw is None or throughput_raw is None:
        workload = ObservedWorkload(order_volume_delta_pct=0.0, admissible=False)
        throughput = ObservedThroughput(
            target_attainment_pct=0.0,
            baseline_attainment_pct=0.0,
            admissible=False,
        )
    else:
        workload = parse_workload_payload(workload_raw)
        throughput = parse_throughput_payload(throughput_raw)

    schedule: ObservedStaffingSchedule | None = None
    if evidence_ids.staffing_schedule in by_id:
        schedule = parse_staffing_schedule_payload(by_id[evidence_ids.staffing_schedule])

    attendance: ObservedStaffingAttendance | None = None
    if evidence_ids.staffing_attendance in by_id:
        attendance = parse_staffing_attendance_payload(by_id[evidence_ids.staffing_attendance])

    comparison: ObservedComparison | None = None
    if evidence_ids.comparison in by_id:
        comparison = parse_comparison_payload(by_id[evidence_ids.comparison])

    telemetry: ObservedTelemetry | None = None
    if evidence_ids.telemetry in by_id:
        telemetry = parse_telemetry_payload(by_id[evidence_ids.telemetry])

    return IncidentObservations(
        workload=workload,
        throughput=throughput,
        staffing_schedule=schedule,
        staffing_attendance=attendance,
        comparison=comparison,
        telemetry=telemetry,
    )
