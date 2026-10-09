# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P3-R2 configured execution fulfillment and execution-bound adoption."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.autonomous_work.configured_capability_execution_subject_builder import (
    build_configured_capability_execution_subject,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)
from intergrax.autonomous_work.worker_configured_capability_execution_fulfillment_service import (
    WorkerConfiguredCapabilityExecutionFulfillmentService,
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
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentDisposition,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
    WorkerCapabilityRecoveryPhase,
    WorkerCapabilityRecoveryProvenance,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionResult,
)
from intergrax.contracts.autonomous_work.ids import mint_worker_instance_id
from intergrax.contracts.autonomous_work.profile_reference import initial_profile_version
from intergrax.contracts.capability_catalog import CapabilityKind, CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
)
from intergrax.integrations.contracts.existing_capability_configuration_opportunity import (
    ConfigurationOpportunityRef,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLiteRelationalStoreConfigurationPayload,
)
from intergrax.tools.marketplace_configured_capability_binding_provider import (
    MarketplaceConfiguredCapabilityBindingProvider,
)
from intergrax.tools.marketplace_configured_tool_execution_intent_preparation import (
    MarketplaceConfiguredToolExecutionIntentPreparation,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_TENANT = "tenant-a"
_CONFIG_REF = ConfigurationOpportunityRef("cfg/opportunity-1")


def _identity() -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.marketplace",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="tools.database.relational",
    )


def _adoption() -> ExecutionIntegrationConfigurationAdoption:
    payload = SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/data/tenant/store"),
        relational_db=Path("/data/tenant/store/custom.db"),
    )
    binding = ConfiguredCapabilityBinding(
        tenant_id=_TENANT,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        configuration_type=payload.configuration_type,
        configuration_version=payload.configuration_version,
        configuration_fingerprint=payload.configuration_fingerprint,
        realization_evidence_refs=("evidence-1",),
    )
    return ExecutionIntegrationConfigurationAdoption(
        configured_binding=binding,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        resource_scope="scope-a",
    )


def _decision() -> WorkerCapabilityAcquisitionDecision:
    worker = mint_worker_instance_id()
    return WorkerCapabilityAcquisitionDecision(
        decision_id="decision-1",
        worker_instance_id=worker,
        obstacle_id=f"{worker}:obs-1",
        recovery_decision_id="recovery-1",
        need_id="need-1",
        capability_profile_ref=CapabilityProfileRef(
            profile_id="profile/default",
            version=initial_profile_version(),
        ),
        disposition=CapabilityAcquisitionDisposition.CONFIGURE_EXISTING,
        reason_code=CapabilityAcquisitionReasonCode.EXISTING_CONFIGURATION_SELECTED,
        selected_candidate=WorkerCapabilityCandidate(
            candidate_id=derive_worker_capability_candidate_id(
                candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
                capability_ref="integration:sqlite",
                configuration_ref=str(_CONFIG_REF),
            ),
            candidate_kind=WorkerCapabilityCandidateKind.EXISTING_CONFIGURATION,
            capability_ref="integration:sqlite",
            source_domain="integrations",
            operations=("database.query",),
            risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
            evidence_refs=(),
            discovered_at=_NOW,
            configuration_ref=str(_CONFIG_REF),
            capability_identity=_identity(),
        ),
        autonomy_level=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        decided_at=_NOW,
        decision_policy_version="v1",
        evidence_refs=(),
    )


def test_subject_builder_rejects_tenant_mismatch() -> None:
    subject = build_configured_capability_execution_subject(
        tenant_id="other",
        worker_need_id="need-1",
        recovery_decision_id="recovery-1",
        decision=_decision(),
        adoption=_adoption(),
        selected_operations=("database.query",),
    )
    assert subject is None


def test_coordinator_routes_to_configured_execution_after_adoption() -> None:
    configured = MagicMock()
    configured.fulfill_configure_existing.return_value = MagicMock(
        adoption=_adoption(),
        failure_reason=None,
    )
    configured_execution = MagicMock()
    configured_execution.fulfill_after_adoption.return_value = MagicMock(
        disposition=WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED,
    )
    recovery_outcome = WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.CONFIGURE_EXISTING_REQUIRED,
        provenance=WorkerCapabilityRecoveryProvenance(
            worker_need_id="need-1",
            canonical_need_id="canonical-1",
            discovery_correlation_id="corr-1",
            discovery_completion_outcome="CONFIGURE_EXISTING",
        ),
        worker_acquisition_decision=_decision(),
    )
    recovery_port = MagicMock()
    recovery_port.coordinate_recovery.return_value = recovery_outcome
    coordinator = WorkerCapabilityFulfillmentCoordinator(
        recovery=recovery_port,
        resume=MagicMock(),
        direct_reuse=MagicMock(),
        configured_fulfillment=configured,
        configured_execution=configured_execution,
    )
    request = MagicMock()
    request.requested_at = _NOW
    request.acquisition_request = MagicMock()
    request.acquisition_request.need.required_operations = ("database.query",)
    result = coordinator.fulfill(request)
    configured_execution.fulfill_after_adoption.assert_called_once()
    assert (
        result.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_DISPATCHED
    )


def test_configured_execution_fulfillment_fail_closed_without_authority() -> None:
    from tests.unit.tools.test_marketplace_configured_tool_execution_intent_preparation import (
        _RecordingRepository,
    )

    repo = _RecordingRepository()
    service = WorkerConfiguredCapabilityExecutionFulfillmentService(
        binding=MarketplaceConfiguredCapabilityBindingProvider(),
        intent_preparation=MarketplaceConfiguredToolExecutionIntentPreparation(
            intent_repository=repo,
        ),
        execution=MagicMock(),
        authority_admission=None,
    )
    request = MagicMock()
    request.tenant_id = _TENANT
    request.task_id = MagicMock()
    request.worker_instance_id = mint_worker_instance_id()
    request.requested_authority_scopes = ()
    request.run_id = None
    request.attempt_id = None
    request.acquisition_request.need.recovery_decision_id = "recovery-1"
    request.acquisition_request.need.required_operations = ("database.query",)
    recovery = WorkerCapabilityRecoveryOutcome(
        phase=WorkerCapabilityRecoveryPhase.CONFIGURE_EXISTING_REQUIRED,
        provenance=WorkerCapabilityRecoveryProvenance(
            worker_need_id="need-1",
            canonical_need_id="canonical-1",
            discovery_correlation_id="corr-1",
            discovery_completion_outcome="CONFIGURE_EXISTING",
        ),
    )
    result = service.fulfill_after_adoption(
        request,
        recovery=recovery,
        decision=_decision(),
        adoption=_adoption(),
        decided_at=_NOW,
    )
    assert result.disposition is WorkerCapabilityFulfillmentDisposition.EXECUTION_FAILED
