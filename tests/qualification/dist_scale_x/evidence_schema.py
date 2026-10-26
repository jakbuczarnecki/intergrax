# © Artur Czarnecki. All rights reserved.

"""Typed DIST-SCALE-X qualification evidence records (metadata validation only in P1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from tests.qualification.dist_scale_x.catalog import (
    DIST_SCALE_X_THRESHOLD_LOCK_STATE,
    ThresholdLockState,
)


class EvidenceDisposition(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class HistoricalEvidenceReconciliation(StrEnum):
    REUSE = "REUSE"
    RE_VALIDATE = "RE-VALIDATE"
    NOT_SUFFICIENT = "NOT SUFFICIENT"


class CorrectnessInvariantViolation(StrEnum):
    DUPLICATE_SIDE_EFFECT = "duplicate_side_effect"
    LOST_STATE = "lost_state"
    LOST_EVIDENCE = "lost_evidence"
    TENANT_LEAKAGE = "tenant_leakage"
    WRONG_RECOVERY = "wrong_recovery"
    STALE_AUTHORITY = "stale_authority"
    CORRUPTION = "corruption"
    GOVERNANCE_BYPASS = "governance_bypass_under_load"


@dataclass(frozen=True, slots=True)
class DistScaleXMetricsSnapshot:
    throughput_per_sec: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    error_rate: float | None = None
    max_queue_depth: int | None = None
    retry_amplification_factor: float | None = None
    extra: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DistScaleXEvidenceRecord:
    record_id: str
    code_sha: str
    topology_description: str
    environment_definition: str
    workload_id: str
    dimension_ids: tuple[str, ...]
    duration_seconds: float
    metrics: DistScaleXMetricsSnapshot
    fault_description: str
    disposition: EvidenceDisposition
    container_service_versions: tuple[str, ...] = ()
    correctness_violations: tuple[CorrectnessInvariantViolation, ...] = ()
    historical_reconciliation: HistoricalEvidenceReconciliation | None = None
    performance_threshold_metric_ids: tuple[str, ...] = ()
    mandatory_fields_complete: bool = True

    def missing_mandatory_field_names(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.code_sha.strip():
            missing.append("code_sha")
        if not self.topology_description.strip():
            missing.append("topology_description")
        if not self.environment_definition.strip():
            missing.append("environment_definition")
        if not self.workload_id.strip():
            missing.append("workload_id")
        if self.duration_seconds <= 0:
            missing.append("duration_seconds")
        if not self.dimension_ids:
            missing.append("dimension_ids")
        if self.disposition in (EvidenceDisposition.PASS, EvidenceDisposition.FAIL):
            if not self.fault_description.strip() and self.disposition == EvidenceDisposition.FAIL:
                missing.append("fault_description")
        return tuple(missing)


def validate_evidence_record(record: DistScaleXEvidenceRecord) -> tuple[str, ...]:
    violations: list[str] = []
    missing = record.missing_mandatory_field_names()
    if missing:
        violations.append(f"missing mandatory fields: {', '.join(missing)}")

    if record.disposition == EvidenceDisposition.PASS:
        if missing:
            violations.append("PASS disposition forbidden with incomplete mandatory evidence")
        if record.correctness_violations:
            violations.append("PASS forbidden with correctness invariant violations")
        if record.performance_threshold_metric_ids:
            if DIST_SCALE_X_THRESHOLD_LOCK_STATE is ThresholdLockState.NOT_LOCKED:
                violations.append(
                    "performance threshold metrics referenced while thresholds NOT_LOCKED",
                )

    if record.historical_reconciliation == HistoricalEvidenceReconciliation.NOT_SUFFICIENT:
        if record.disposition == EvidenceDisposition.PASS:
            violations.append("NOT SUFFICIENT historical evidence cannot yield PASS alone")

    return tuple(violations)


def resolve_effective_disposition(
    record: DistScaleXEvidenceRecord,
    *,
    structural_violations: tuple[str, ...] = (),
) -> EvidenceDisposition:
    """Test outcomes cannot override automatic correctness or structural failures."""
    if structural_violations:
        return EvidenceDisposition.FAIL
    if record.correctness_violations:
        return EvidenceDisposition.FAIL
    validation = validate_evidence_record(record)
    if validation:
        return EvidenceDisposition.FAIL
    return record.disposition


def evidence_record_from_mapping(payload: Mapping[str, Any]) -> DistScaleXEvidenceRecord:
    metrics_raw = payload.get("metrics") or {}
    metrics = DistScaleXMetricsSnapshot(
        throughput_per_sec=metrics_raw.get("throughput_per_sec"),
        p95_latency_ms=metrics_raw.get("p95_latency_ms"),
        p99_latency_ms=metrics_raw.get("p99_latency_ms"),
        error_rate=metrics_raw.get("error_rate"),
        max_queue_depth=metrics_raw.get("max_queue_depth"),
        retry_amplification_factor=metrics_raw.get("retry_amplification_factor"),
        extra=dict(metrics_raw.get("extra") or {}),
    )
    disp_raw = str(payload.get("disposition", EvidenceDisposition.NOT_RUN))
    disposition = EvidenceDisposition(disp_raw)
    hist_raw = payload.get("historical_reconciliation")
    hist = (
        HistoricalEvidenceReconciliation(str(hist_raw)) if hist_raw is not None else None
    )
    violations_raw = payload.get("correctness_violations") or ()
    violations = tuple(CorrectnessInvariantViolation(str(v)) for v in violations_raw)
    return DistScaleXEvidenceRecord(
        record_id=str(payload.get("record_id", "")),
        code_sha=str(payload.get("code_sha", "")),
        topology_description=str(payload.get("topology_description", "")),
        environment_definition=str(payload.get("environment_definition", "")),
        workload_id=str(payload.get("workload_id", "")),
        dimension_ids=tuple(str(d) for d in (payload.get("dimension_ids") or ())),
        duration_seconds=float(payload.get("duration_seconds") or 0),
        metrics=metrics,
        fault_description=str(payload.get("fault_description", "")),
        disposition=disposition,
        container_service_versions=tuple(
            str(v) for v in (payload.get("container_service_versions") or ())
        ),
        correctness_violations=violations,
        historical_reconciliation=hist,
        performance_threshold_metric_ids=tuple(
            str(m) for m in (payload.get("performance_threshold_metric_ids") or ())
        ),
        mandatory_fields_complete=bool(payload.get("mandatory_fields_complete", True)),
    )
