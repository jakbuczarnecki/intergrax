# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime spine recorder for integration configuration provenance requirements."""

from __future__ import annotations

from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    ExecutionIntegrationConfigurationProvenanceRequirementCommitResult,
    ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus,
    ExecutionIntegrationConfigurationProvenanceRequirementFact,
    derive_integration_configuration_provenance_requirement_event_id,
)
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import get_catalog_entry
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.spine_families import (
    IntegrationConfigurationProvenanceRequirementPayloadV1,
)
from intergrax.runtime.events.persistence_contract import MandatoryEvidencePersistenceError
from intergrax.runtime.events.runtime_event import RuntimeEvent


class RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder:
    """Maps immutable requirement facts onto the canonical RuntimeEvent bus."""

    __slots__ = ("_bus",)

    def __init__(self, bus: RuntimeEventBus) -> None:
        self._bus = bus

    def commit(
        self,
        fact: ExecutionIntegrationConfigurationProvenanceRequirementFact,
    ) -> ExecutionIntegrationConfigurationProvenanceRequirementCommitResult:
        if self._bus.persistence is None:
            return ExecutionIntegrationConfigurationProvenanceRequirementCommitResult(
                status=ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus.PERSISTENCE_UNAVAILABLE,
            )
        event_type = RuntimeEventType.INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED
        catalog_entry = get_catalog_entry(event_type)
        if catalog_entry is None:
            raise RuntimeError(
                "INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED missing catalog entry",
            )
        subject = fact.subject
        payload = IntegrationConfigurationProvenanceRequirementPayloadV1(
            integration_category=subject.integration_category.value,
            provider_id=subject.provider_id,
            resource_scope=subject.resource_scope,
            configuration_type=subject.configuration_type,
            provenance_mode=fact.mode.value,
        )
        event_id = derive_integration_configuration_provenance_requirement_event_id(
            tenant_id=fact.tenant_id,
            execution_id=fact.execution_id,
            subject=subject,
        )
        correlation_id = fact.correlation_id if fact.correlation_id is not None else ""
        event = RuntimeEvent(
            event_id=event_id,
            tenant_id=fact.tenant_id,
            task_id=fact.task_id,
            run_id=fact.run_id,
            attempt_id=fact.attempt_id,
            execution_id=fact.execution_id,
            event_type=event_type,
            phase=fact.phase,
            severity=fact.severity,
            timestamp=fact.factual_timestamp,
            node_id=fact.node_id,
            agent_id=fact.agent_id,
            step_id=fact.step_id,
            correlation_id=correlation_id,
            parent_event_id=fact.parent_event_id,
            traceparent=fact.traceparent,
            tracestate=fact.tracestate,
            payload={},
        )
        event = runtime_event_with_payload(event, payload)
        try:
            self._bus.record(event, tenant_id=fact.tenant_id)
        except MandatoryEvidencePersistenceError:
            return ExecutionIntegrationConfigurationProvenanceRequirementCommitResult(
                status=ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus.PERSISTENCE_UNAVAILABLE,
            )
        return ExecutionIntegrationConfigurationProvenanceRequirementCommitResult(
            status=ExecutionIntegrationConfigurationProvenanceRequirementCommitStatus.COMMITTED,
            event_id=event_id,
        )


__all__ = ["RuntimeEventIntegrationConfigurationProvenanceRequirementRecorder"]
