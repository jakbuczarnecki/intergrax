# © Artur Czarnecki. All rights reserved.

"""Typed operational assessment projection for EE-B4-A (read-only, side-effect free).

Runtime emits facts; this module is the certification reference for composing
health / readiness / liveness without becoming execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_capacity import (
    ExecutionCapacityAdmissionDecision,
    ExecutionCapacityAssessmentContext,
    assess_root_execution_capacity,
)
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityOverloadMode,
)
from intergrax.contracts.execution_reliability import ExecutionRuntimeShutdownPhase
from intergrax.runtime.observability.auditability_health import (
    resolve_auditability_ready,
)


class ExecutionOperationalScope(StrEnum):
    GLOBAL = "global"
    TENANT = "tenant"
    EXECUTION_PROFILE = "execution_profile"
    DEPENDENCY = "dependency"
    WORKER_POOL = "worker_pool"


class HealthClassification(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ReadinessClassification(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"
    STARTING = "starting"


class LivenessClassification(StrEnum):
    LIVE = "live"
    NOT_LIVE = "not_live"


class SaturationClassification(StrEnum):
    NORMAL = "normal"
    APPROACHING_SATURATION = "approaching_saturation"
    SATURATED = "saturated"
    NOT_APPLICABLE = "not_applicable"


class ExecutionOperationalReasonCode(StrEnum):
    OK = "ok"
    STARTING = "starting"
    CAPACITY_SATURATED = "capacity_saturated"
    CAPACITY_APPROACHING = "capacity_approaching"
    MANDATORY_EVIDENCE_UNAVAILABLE = "mandatory_evidence_unavailable"
    MANDATORY_DIAGNOSTICS_UNAVAILABLE = "mandatory_diagnostics_unavailable"
    OBSERVABILITY_EXPORT_DEGRADED = "observability_export_degraded"
    SHUTDOWN_STOP_ACCEPTING = "shutdown_stop_accepting"
    SHUTDOWN_DRAINING = "shutdown_draining"
    SHUTDOWN_TERMINATING = "shutdown_terminating"
    WORKER_POOL_DEGRADED = "worker_pool_degraded"
    DEPENDENCY_DEGRADED = "dependency_degraded"
    RECOVERY_SUBSYSTEM_DEGRADED = "recovery_subsystem_degraded"
    PROCESS_NOT_ALIVE = "process_not_alive"


@dataclass(frozen=True, slots=True)
class ExecutionOperationalFacts:
    """Immutable runtime facts consumed by operational assessment (projection input)."""

    scope: ExecutionOperationalScope
    process_alive: bool
    startup_complete: bool
    shutdown_phase: ExecutionRuntimeShutdownPhase | None
    active_root_executions: int | None
    capacity_limit: int | None
    overload_mode: ExecutionCapacityOverloadMode
    diagnostics_required: bool
    diagnostics_attached: bool
    runtime_event_persistence_available: bool
    diagnostic_read_side_required: bool
    diagnostic_read_side_ready: bool
    mandatory_evidence_persistence_available: bool
    best_effort_observability_export_available: bool
    worker_pool_degraded: bool
    dependency_degraded: bool
    recovery_subsystem_degraded: bool


@dataclass(frozen=True, slots=True)
class ExecutionOperationalAssessment:
    """Immutable operator-visible operational classification."""

    scope: ExecutionOperationalScope
    health: HealthClassification
    readiness: ReadinessClassification
    liveness: LivenessClassification
    saturation: SaturationClassification
    reason_codes: tuple[ExecutionOperationalReasonCode, ...]


def _capacity_decision(
    facts: ExecutionOperationalFacts,
) -> ExecutionCapacityAdmissionDecision | None:
    if facts.active_root_executions is None or facts.capacity_limit is None:
        return None
    context = ExecutionCapacityAssessmentContext(
        active_root_executions=facts.active_root_executions,
        capacity_limit=facts.capacity_limit,
        overload_mode=facts.overload_mode,
    )
    return assess_root_execution_capacity(context)


def _classify_saturation(
    facts: ExecutionOperationalFacts,
    capacity_decision: ExecutionCapacityAdmissionDecision | None,
) -> tuple[SaturationClassification, ExecutionOperationalReasonCode | None]:
    if capacity_decision is None:
        return SaturationClassification.NOT_APPLICABLE, None
    if capacity_decision is ExecutionCapacityAdmissionDecision.ALLOW:
        if (
            facts.capacity_limit is not None
            and facts.active_root_executions is not None
            and facts.capacity_limit > 1
            and facts.active_root_executions == facts.capacity_limit - 1
        ):
            return (
                SaturationClassification.APPROACHING_SATURATION,
                ExecutionOperationalReasonCode.CAPACITY_APPROACHING,
            )
        return SaturationClassification.NORMAL, None
    return (
        SaturationClassification.SATURATED,
        ExecutionOperationalReasonCode.CAPACITY_SATURATED,
    )


def assess_execution_operational_state(
    facts: ExecutionOperationalFacts,
) -> ExecutionOperationalAssessment:
    """Deterministic, side-effect-free operational assessment from runtime facts."""
    reasons: list[ExecutionOperationalReasonCode] = []

    if not facts.process_alive:
        reasons.append(ExecutionOperationalReasonCode.PROCESS_NOT_ALIVE)
        return ExecutionOperationalAssessment(
            scope=facts.scope,
            health=HealthClassification.UNHEALTHY,
            readiness=ReadinessClassification.NOT_READY,
            liveness=LivenessClassification.NOT_LIVE,
            saturation=SaturationClassification.NOT_APPLICABLE,
            reason_codes=tuple(reasons),
        )

    liveness = LivenessClassification.LIVE
    if facts.shutdown_phase is ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS:
        reasons.append(ExecutionOperationalReasonCode.SHUTDOWN_TERMINATING)
        liveness = LivenessClassification.NOT_LIVE

    if not facts.startup_complete:
        reasons.append(ExecutionOperationalReasonCode.STARTING)
        return ExecutionOperationalAssessment(
            scope=facts.scope,
            health=HealthClassification.DEGRADED,
            readiness=ReadinessClassification.STARTING,
            liveness=liveness,
            saturation=SaturationClassification.NOT_APPLICABLE,
            reason_codes=tuple(reasons),
        )

    auditability_ready = resolve_auditability_ready(
        diagnostics_required=facts.diagnostics_required,
        diagnostics_attached=facts.diagnostics_attached,
        runtime_event_persistence_available=facts.runtime_event_persistence_available,
        diagnostic_read_side_required=facts.diagnostic_read_side_required,
        diagnostic_read_side_ready=facts.diagnostic_read_side_ready,
    )
    mandatory_evidence_ready = facts.mandatory_evidence_persistence_available
    if facts.diagnostics_required and not auditability_ready:
        reasons.append(ExecutionOperationalReasonCode.MANDATORY_DIAGNOSTICS_UNAVAILABLE)
    if not mandatory_evidence_ready:
        reasons.append(ExecutionOperationalReasonCode.MANDATORY_EVIDENCE_UNAVAILABLE)

    capacity_decision = _capacity_decision(facts)
    saturation, saturation_reason = _classify_saturation(facts, capacity_decision)
    if saturation_reason is not None:
        reasons.append(saturation_reason)

    if not facts.best_effort_observability_export_available:
        reasons.append(ExecutionOperationalReasonCode.OBSERVABILITY_EXPORT_DEGRADED)

    if facts.worker_pool_degraded:
        reasons.append(ExecutionOperationalReasonCode.WORKER_POOL_DEGRADED)
    if facts.dependency_degraded:
        reasons.append(ExecutionOperationalReasonCode.DEPENDENCY_DEGRADED)
    if facts.recovery_subsystem_degraded:
        reasons.append(ExecutionOperationalReasonCode.RECOVERY_SUBSYSTEM_DEGRADED)

    accepting_new_work = facts.shutdown_phase is None
    if facts.shutdown_phase is not None:
        if (
            facts.shutdown_phase
            is ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK
        ):
            reasons.append(ExecutionOperationalReasonCode.SHUTDOWN_STOP_ACCEPTING)
            accepting_new_work = False
        elif (
            facts.shutdown_phase
            is ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS
        ):
            reasons.append(ExecutionOperationalReasonCode.SHUTDOWN_DRAINING)
            accepting_new_work = False
        elif facts.shutdown_phase in (
            ExecutionRuntimeShutdownPhase.FLUSH_REQUIRED_EVIDENCE,
            ExecutionRuntimeShutdownPhase.PERSIST_FINAL_STATE,
        ):
            accepting_new_work = False

    readiness = ReadinessClassification.READY
    if not accepting_new_work:
        readiness = ReadinessClassification.NOT_READY
    elif not mandatory_evidence_ready:
        readiness = ReadinessClassification.NOT_READY
    elif facts.diagnostics_required and not auditability_ready:
        readiness = ReadinessClassification.NOT_READY
    elif capacity_decision in (
        ExecutionCapacityAdmissionDecision.REJECT,
        ExecutionCapacityAdmissionDecision.DEFER,
    ):
        readiness = ReadinessClassification.NOT_READY

    health = HealthClassification.HEALTHY
    if not mandatory_evidence_ready or (
        facts.diagnostics_required and not auditability_ready
    ):
        health = HealthClassification.UNHEALTHY
    elif (
        saturation is SaturationClassification.SATURATED
        or not facts.best_effort_observability_export_available
        or facts.worker_pool_degraded
        or facts.dependency_degraded
        or facts.recovery_subsystem_degraded
        or facts.shutdown_phase is not None
    ):
        health = HealthClassification.DEGRADED

    if not reasons:
        reasons.append(ExecutionOperationalReasonCode.OK)

    return ExecutionOperationalAssessment(
        scope=facts.scope,
        health=health,
        readiness=readiness,
        liveness=liveness,
        saturation=saturation,
        reason_codes=tuple(reasons),
    )


def assessment_unchanged_by_policy_or_security_denial(
    before: ExecutionOperationalAssessment,
    after: ExecutionOperationalAssessment,
) -> bool:
    """Governance/security denials must not flip global operational classification."""
    return before == after
