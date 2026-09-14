# © Artur Czarnecki. All rights reserved.

"""Canonical Execution Engine incident taxonomy (EE-B4-C, certification only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IncidentCategoryId(StrEnum):
    INC_01_CAPACITY_SATURATION = "INC-01"
    INC_02_MANDATORY_EVIDENCE_UNAVAILABLE = "INC-02"
    INC_03_BEST_EFFORT_OBSERVABILITY_EXPORT_UNAVAILABLE = "INC-03"
    INC_04_WORKER_FAILURE_OR_POOL_DEGRADATION = "INC-04"
    INC_05_NEXUS_FAN_OUT_DEGRADATION = "INC-05"
    INC_06_TOOL_PROVIDER_DEPENDENCY_OUTAGE = "INC-06"
    INC_07_RETRY_EXHAUSTION = "INC-07"
    INC_08_RECOVERY_FAILURE = "INC-08"
    INC_09_CHECKPOINT_INCOMPATIBILITY_OR_CORRUPTION = "INC-09"
    INC_10_SECURITY_GOVERNANCE_DENIAL_SPIKE = "INC-10"
    INC_11_GRACEFUL_SHUTDOWN_INCOMPLETE = "INC-11"
    INC_12_EXECUTION_RUNTIME_FATAL_OR_UNHEALTHY = "INC-12"
    INC_13_TENANT_SCOPED_DEGRADATION = "INC-13"
    INC_14_UNKNOWN_UNCLASSIFIED_RUNTIME_FAILURE = "INC-14"


class OperationalSeverity(StrEnum):
    """Typed operational severity (single framework for EE production ops)."""

    SEV_1 = "SEV-1"
    SEV_2 = "SEV-2"
    SEV_3 = "SEV-3"
    SEV_4 = "SEV-4"


class DegradedModeDisposition(StrEnum):
    CONTINUE_NORMALLY = "continue_normally"
    CONTINUE_DEGRADED = "continue_degraded"
    STOP_ACCEPTING = "stop_accepting"
    FAIL_CLOSED = "fail_closed"
    TERMINATE = "terminate"


@dataclass(frozen=True, slots=True)
class ExecutionIncidentDescriptor:
    incident_id: IncidentCategoryId
    title: str
    typical_severity: OperationalSeverity
    primary_runbook_id: str
    readiness_impact: str
    canonical_owner: str
    default_disposition: DegradedModeDisposition


EXECUTION_ENGINE_INCIDENT_CATALOG: tuple[ExecutionIncidentDescriptor, ...] = (
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_01_CAPACITY_SATURATION,
        title="Capacity saturation",
        typical_severity=OperationalSeverity.SEV_2,
        primary_runbook_id="RB-02",
        readiness_impact="not_ready_when_reject",
        canonical_owner="EE-B1.2 ExecutionCapacityAdmissionPort",
        default_disposition=DegradedModeDisposition.STOP_ACCEPTING,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_02_MANDATORY_EVIDENCE_UNAVAILABLE,
        title="Mandatory evidence persistence unavailable",
        typical_severity=OperationalSeverity.SEV_1,
        primary_runbook_id="RB-03",
        readiness_impact="not_ready_fail_closed",
        canonical_owner="runtime event store / mandatory persistence",
        default_disposition=DegradedModeDisposition.FAIL_CLOSED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_03_BEST_EFFORT_OBSERVABILITY_EXPORT_UNAVAILABLE,
        title="Best-effort observability exporter unavailable",
        typical_severity=OperationalSeverity.SEV_4,
        primary_runbook_id="RB-04",
        readiness_impact="ready_execution_continues",
        canonical_owner="observability export boundary",
        default_disposition=DegradedModeDisposition.CONTINUE_DEGRADED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_04_WORKER_FAILURE_OR_POOL_DEGRADATION,
        title="Worker failure / worker pool degradation",
        typical_severity=OperationalSeverity.SEV_3,
        primary_runbook_id="RB-06",
        readiness_impact="scoped_degraded_global_may_remain_ready",
        canonical_owner="EE-B1.3 worker containment",
        default_disposition=DegradedModeDisposition.CONTINUE_DEGRADED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_05_NEXUS_FAN_OUT_DEGRADATION,
        title="Nexus / fan-out degradation",
        typical_severity=OperationalSeverity.SEV_3,
        primary_runbook_id="RB-06",
        readiness_impact="orchestration_scoped_not_second_engine",
        canonical_owner="nexus orchestration plane",
        default_disposition=DegradedModeDisposition.CONTINUE_DEGRADED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_06_TOOL_PROVIDER_DEPENDENCY_OUTAGE,
        title="Tool/provider dependency outage",
        typical_severity=OperationalSeverity.SEV_3,
        primary_runbook_id="RB-09",
        readiness_impact="profile_mandatory_dependency_may_block",
        canonical_owner="integration / tool invocation",
        default_disposition=DegradedModeDisposition.CONTINUE_DEGRADED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_07_RETRY_EXHAUSTION,
        title="Retry exhaustion",
        typical_severity=OperationalSeverity.SEV_3,
        primary_runbook_id="RB-10",
        readiness_impact="execution_terminal_not_global_unready",
        canonical_owner="NPSC-5E recovery plane",
        default_disposition=DegradedModeDisposition.CONTINUE_DEGRADED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_08_RECOVERY_FAILURE,
        title="Recovery failure",
        typical_severity=OperationalSeverity.SEV_2,
        primary_runbook_id="RB-07",
        readiness_impact="scoped_execution_blocked",
        canonical_owner="NPSC-5E recovery plane",
        default_disposition=DegradedModeDisposition.FAIL_CLOSED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_09_CHECKPOINT_INCOMPATIBILITY_OR_CORRUPTION,
        title="Checkpoint incompatibility / corruption",
        typical_severity=OperationalSeverity.SEV_2,
        primary_runbook_id="RB-08",
        readiness_impact="recovery_ineligible_until_resolved",
        canonical_owner="checkpoint store + recovery admission",
        default_disposition=DegradedModeDisposition.FAIL_CLOSED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_10_SECURITY_GOVERNANCE_DENIAL_SPIKE,
        title="Security / governance denial spike",
        typical_severity=OperationalSeverity.SEV_3,
        primary_runbook_id="RB-12",
        readiness_impact="not_engine_unhealthy_by_default",
        canonical_owner="governance admission",
        default_disposition=DegradedModeDisposition.CONTINUE_NORMALLY,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_11_GRACEFUL_SHUTDOWN_INCOMPLETE,
        title="Graceful shutdown incomplete",
        typical_severity=OperationalSeverity.SEV_2,
        primary_runbook_id="RB-13",
        readiness_impact="not_ready_during_shutdown",
        canonical_owner="EE-B4-B shutdown contract",
        default_disposition=DegradedModeDisposition.STOP_ACCEPTING,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_12_EXECUTION_RUNTIME_FATAL_OR_UNHEALTHY,
        title="ExecutionRuntime fatal/unhealthy",
        typical_severity=OperationalSeverity.SEV_1,
        primary_runbook_id="RB-01",
        readiness_impact="not_ready",
        canonical_owner="execution runtime host",
        default_disposition=DegradedModeDisposition.FAIL_CLOSED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_13_TENANT_SCOPED_DEGRADATION,
        title="Tenant-scoped degradation",
        typical_severity=OperationalSeverity.SEV_3,
        primary_runbook_id="RB-14",
        readiness_impact="tenant_or_profile_scope_only",
        canonical_owner="tenant-scoped facts + admission",
        default_disposition=DegradedModeDisposition.CONTINUE_DEGRADED,
    ),
    ExecutionIncidentDescriptor(
        incident_id=IncidentCategoryId.INC_14_UNKNOWN_UNCLASSIFIED_RUNTIME_FAILURE,
        title="Unknown / unclassified runtime failure",
        typical_severity=OperationalSeverity.SEV_2,
        primary_runbook_id="RB-01",
        readiness_impact="classify_before_restart",
        canonical_owner="operator classification then canonical owners",
        default_disposition=DegradedModeDisposition.STOP_ACCEPTING,
    ),
)

REQUIRED_RUNBOOK_IDS: tuple[str, ...] = (
    "RB-01",
    "RB-02",
    "RB-03",
    "RB-04",
    "RB-05",
    "RB-06",
    "RB-07",
    "RB-08",
    "RB-09",
    "RB-10",
    "RB-11",
    "RB-12",
    "RB-13",
    "RB-14",
)
