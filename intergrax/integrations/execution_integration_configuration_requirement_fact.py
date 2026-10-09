# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build requirement facts from durable pin records (TRACE-X-P5-R2-P4-R2)."""

from __future__ import annotations

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    ExecutionIntegrationConfigurationProvenanceRequirementFact,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationPinRecord,
    obligation_requires_recovery_staging,
)


def build_requirement_fact_from_pin_record(
    pin_record: ExecutionIntegrationConfigurationPinRecord,
    *,
    phase: ExecutionPhase = ExecutionPhase.STEP_EXECUTION,
    severity: EventSeverity = EventSeverity.INFO,
) -> ExecutionIntegrationConfigurationProvenanceRequirementFact:
    provenance = pin_record.provenance
    if not obligation_requires_recovery_staging(provenance):
        raise ValueError("requirement fact requires CONFIGURED_ADOPTED provenance")
    staging = pin_record.requirement_recovery_staging
    if staging is None:
        raise ValueError("requirement fact requires recovery staging on pin record")
    return ExecutionIntegrationConfigurationProvenanceRequirementFact(
        tenant_id=provenance.tenant_id,
        execution_id=provenance.execution_id,
        subject=pin_record.subject,
        mode=provenance.mode,
        task_id=staging.task_id,
        run_id=staging.run_id,
        attempt_id=staging.attempt_id,
        factual_timestamp=staging.requirement_boundary_prepared_at,
        phase=phase,
        severity=severity,
        node_id=staging.node_id,
        agent_id=staging.agent_id,
        step_id=staging.step_id,
        correlation_id=staging.correlation_id,
        traceparent=staging.traceparent,
        tracestate=staging.tracestate,
    )


__all__ = ["build_requirement_fact_from_pin_record"]
