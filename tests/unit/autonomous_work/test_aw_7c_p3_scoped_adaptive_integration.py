# © Artur Czarnecki. All rights reserved.

"""AW-7C-P3 pure A2 → Integrations → CQ_PENDING flow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.autonomous_work.scoped_adaptive_integration import (
    WorkerScopedAdaptiveIntegrationOrchestrationService,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityProfileRef,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
)
from intergrax.contracts.autonomous_work.ids import WorkerInstanceId
from intergrax.contracts.autonomous_work.profile_reference import (
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.scoped_adaptive_integration import (
    ScopedAdaptiveIntegrationCorrelation,
    ScopedAdaptiveIntegrationExecutionRequest,
    ScopedAdaptiveIntegrationPreparationStatus,
)
from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationScope,
)
from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
    REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
    ReferenceScopedIntegrationAdaptationStrategy,
    reference_read_operation,
    reference_write_operation,
)
from intergrax.integrations.scoped_integration_adaptation_service import (
    ScopedIntegrationAdaptationPortAdapter,
)
from intergrax.integrations.scoped_integration_adaptation_target_resolver import (
    IntegrationIdentityScopedIntegrationAdaptationTargetResolver,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_WORKER_ID = contract_suite.mint_worker_instance_id()
_TS = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)


def _decision() -> WorkerCapabilityAcquisitionDecision:
    candidate = WorkerCapabilityCandidate(
        candidate_id="cand-1",
        candidate_kind=WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION,
        capability_ref="integration:demo",
        source_domain="integrations",
        operations=("READ_CONFIGURATION", "WRITE_CONFIGURATION"),
        risk_class=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
        evidence_refs=(),
        discovered_at=_TS,
    )
    return WorkerCapabilityAcquisitionDecision(
        decision_id="dec-1",
        worker_instance_id=_WORKER_ID,
        obstacle_id="obs-1",
        recovery_decision_id="rec-1",
        need_id="need-1",
        disposition=CapabilityAcquisitionDisposition.SCOPED_ADAPTATION_CANDIDATE,
        capability_profile_ref=_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A2_ADAPTATION_REQUIRED,
        evidence_refs=(),
        decided_at=_TS,
        selected_candidate=candidate,
        autonomy_level=WorkerAutonomyLevel.A2_SCOPED_ADAPTIVE,
    )


def _scope() -> ScopedIntegrationAdaptationScope:
    return ScopedIntegrationAdaptationScope(
        tenant_id="tenant-a",
        integration_category=IntegrationCategory.MESSAGE_BUS,
        provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
        resource_scope="rs-1",
        permitted_operations=(reference_read_operation(), reference_write_operation()),
        network_allowlist=NetworkEgressAllowlist(
            hosts=(
                NetworkEgressHost(scheme="https", hostname="a.example.com", port=443),
                NetworkEgressHost(scheme="https", hostname="b.example.com", port=443),
            ),
        ),
        credential_grant_ref="grant-1",
        expires_at=_TS + timedelta(hours=1),
        candidate_id="cand-1",
        candidate_revision="rev-1",
    )


def test_p3_end_to_end_qualification_pending_without_side_effects() -> None:
    decision = _decision()
    candidate = decision.selected_candidate
    assert candidate is not None
    adaptation_port = ScopedIntegrationAdaptationPortAdapter(
        target_resolver=IntegrationIdentityScopedIntegrationAdaptationTargetResolver(),
        strategies=(ReferenceScopedIntegrationAdaptationStrategy(),),
    )
    service = WorkerScopedAdaptiveIntegrationOrchestrationService(
        adaptation_port=adaptation_port,
    )
    request = ScopedAdaptiveIntegrationExecutionRequest(
        worker_instance_id=_WORKER_ID,
        correlation=ScopedAdaptiveIntegrationCorrelation(tenant_id="tenant-a"),
        acquisition_decision=decision,
        selected_candidate=candidate,
        need_id="need-1",
        recovery_decision_id="rec-1",
        integration_capability_ref="integration:demo",
        required_operations=("READ_CONFIGURATION", "WRITE_CONFIGURATION"),
        adaptation_scope=_scope(),
        requested_at=_TS,
    )
    result = service.prepare(request)
    assert result.status is ScopedAdaptiveIntegrationPreparationStatus.QUALIFICATION_PENDING
    assert result.artifact is not None
    assert result.qualification_subject is not None
    assert result.qualification_request is not None
    assert result.qualification_subject.tenant_id == "tenant-a"
