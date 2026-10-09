# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Build ConfiguredCapabilityExecutionSubject from adoption + CONFIGURE_EXISTING decision."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidateKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_qualification.configured_capability_execution_subject import (
    ConfiguredCapabilityExecutionSubject,
    derive_configuration_adoption_identity,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)


def build_configured_capability_execution_subject(
    *,
    tenant_id: str,
    worker_need_id: str,
    recovery_decision_id: str,
    decision: WorkerCapabilityAcquisitionDecision,
    adoption: ExecutionIntegrationConfigurationAdoption,
    selected_operations: tuple[str, ...],
) -> ConfiguredCapabilityExecutionSubject | None:
    if decision.disposition is not CapabilityAcquisitionDisposition.CONFIGURE_EXISTING:
        return None
    candidate = decision.selected_candidate
    if candidate is None:
        return None
    if candidate.candidate_kind is not WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION:
        return None
    identity = candidate.capability_identity
    if type(identity) is not CapabilityIdentityKey:
        return None
    binding = adoption.configured_binding
    if binding.tenant_id != tenant_id:
        return None
    if not selected_operations:
        return None
    adoption_identity = derive_configuration_adoption_identity(
        recovery_decision_id=recovery_decision_id,
        decision_id=decision.decision_id,
        configuration_fingerprint=binding.configuration_fingerprint,
    )
    return ConfiguredCapabilityExecutionSubject(
        tenant_id=tenant_id,
        worker_need_id=worker_need_id,
        recovery_decision_id=recovery_decision_id,
        decision_id=decision.decision_id,
        capability_identity=identity,
        configuration_adoption_identity=adoption_identity,
        configuration_fingerprint=binding.configuration_fingerprint,
        selected_operations=selected_operations,
    )


__all__ = ["build_configured_capability_execution_subject"]
