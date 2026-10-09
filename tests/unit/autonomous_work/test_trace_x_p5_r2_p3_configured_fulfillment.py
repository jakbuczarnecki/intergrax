# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3 configured fulfillment and coordinator routing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_configured_capability_fulfillment_service import (
    WorkerConfiguredCapabilityFulfillmentService,
)
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingResolver,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityProfileRef,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    derive_worker_capability_candidate_id,
)
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.autonomous_work.ids import mint_worker_instance_id
from intergrax.contracts.autonomous_work.profile_reference import initial_profile_version
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationResult,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
    ExistingCapabilityConfigurationOpportunity,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLiteRelationalStoreConfigurationPayload,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = "tenant-a"
_CONFIG_REF = ConfigurationOpportunityRef("cfg/opportunity-1")


def _capability_identity() -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.database.relational",
    )


def _payload() -> SQLiteRelationalStoreConfigurationPayload:
    return SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/data/tenant/store"),
        relational_db=Path("/data/tenant/store/custom.db"),
    )


def _opportunity() -> ExistingCapabilityConfigurationOpportunity:
    payload = _payload()
    return ExistingCapabilityConfigurationOpportunity(
        configuration_ref=_CONFIG_REF,
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        current_revision="rev-1",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        risk_classification=ControlPlaneMutationRisk.LOW,
    )


def _binding() -> ConfiguredCapabilityBinding:
    payload = _payload()
    return ConfiguredCapabilityBinding(
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        configuration_type=payload.configuration_type,
        configuration_version=payload.configuration_version,
        configuration_fingerprint=payload.configuration_fingerprint,
        realization_evidence_refs=("evidence-1",),
    )


def test_task_id_without_run_id_omitted_on_realization_request() -> None:
    opportunity = _opportunity()
    read = MagicMock()
    read.read_exact.return_value = opportunity
    realize = MagicMock()
    binding = _binding()
    realize.realize.return_value = ExistingCapabilityConfigurationRealizationResult(
        request_id="req-1",
        configured_binding=binding,
        authorization_evidence=MagicMock(),
    )
    principal = MagicMock()
    principal.tenant_id = _TENANT
    principal.principal_id = "principal-1"
    resolver = MagicMock()
    resolver.resolve.return_value = principal
    service = WorkerConfiguredCapabilityFulfillmentService(
        opportunity_read=read,
        realization=realize,
        principal_binding_resolver=resolver,
    )
    decision = MagicMock()
    decision.disposition = CapabilityAcquisitionDisposition.CONFIGURE_EXISTING
    candidate = MagicMock()
    candidate.candidate_kind = WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION
    candidate.configuration_ref = str(_CONFIG_REF)
    decision.selected_candidate = candidate
    request = MagicMock()
    request.tenant_id = _TENANT
    request.run_id = None
    request.task_id = MagicMock()
    request.acquisition_request.need.recovery_decision_id = "decision-1"
    request.worker_instance_id = MagicMock()
    recovery = WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.CONFIGURE_EXISTING_REQUIRED,
        provenance=WorkerCapabilityRecoveryProvenance(
            worker_need_id="need-1",
            canonical_need_id="canonical-1",
            discovery_correlation_id="corr-1",
            discovery_completion_outcome="MISSING_CAPABILITY",
        ),
    )
    result = service.fulfill_configure_existing(request, recovery, decision)
    assert result.adoption is not None
    realize.realize.assert_called_once()
    call_request = realize.realize.call_args.args[0]
    assert call_request.task_id is None
    assert call_request.run_id is None


def test_coordinator_routes_configure_existing_not_generic_realization() -> None:
    configured = MagicMock()
    configured.fulfill_configure_existing.return_value = MagicMock(
        adoption=MagicMock(),
        failure_reason=None,
    )
    realization = MagicMock()
    worker = mint_worker_instance_id()
    candidate = WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
            capability_ref="integration:sqlite",
            configuration_ref=str(_CONFIG_REF),
        ),
        candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
        capability_ref="integration:sqlite",
        source_domain="integrations",
        operations=("query",),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(),
        discovered_at=_NOW,
        configuration_ref=str(_CONFIG_REF),
        capability_identity=_capability_identity(),
    )
    acquisition_decision = WorkerCapabilityAcquisitionDecision(
        decision_id="decision-1",
        worker_instance_id=worker,
        obstacle_id=f"{worker}:obs-1",
        recovery_decision_id="decision-1",
        need_id="need-1",
        capability_profile_ref=CapabilityProfileRef(
            profile_id="profile/default",
            version=initial_profile_version(),
        ),
        disposition=CapabilityAcquisitionDisposition.CONFIGURE_EXISTING,
        reason_code=CapabilityAcquisitionReasonCode.EXISTING_CONFIGURATION_SELECTED,
        selected_candidate=candidate,
        autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        decided_at=_NOW,
        decision_policy_version="v1",
        evidence_refs=(),
    )
    recovery_outcome = WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.REALIZATION_REQUIRED,
        provenance=WorkerCapabilityRecoveryProvenance(
            worker_need_id="need-1",
            canonical_need_id="canonical-1",
            discovery_correlation_id="corr-1",
            discovery_completion_outcome="REALIZATION_REQUIRED",
        ),
        worker_acquisition_decision=acquisition_decision,
    )
    recovery_port = MagicMock()
    recovery_port.coordinate_recovery.return_value = recovery_outcome
    coordinator = WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery_port,
        resume=MagicMock(),
        direct_reuse=MagicMock(),
        realization=realization,
        configured_fulfillment=configured,
    )
    request = MagicMock()
    request.requested_at = _NOW
    request.acquisition_request = MagicMock()
    coordinator.fulfill(request)
    configured.fulfill_configure_existing.assert_called_once()
    realization.realize.assert_not_called()
