# © Artur Czarnecki. All rights reserved.

"""AW-7C-P2 scoped adaptive integration orchestration."""

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
from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationOperation,
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationScope,
    build_scoped_integration_adaptation_artifact,
)
from intergrax.integrations.scoped_integration_adaptation_service import (
    ScopedIntegrationAdaptationService,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_WORKER_ID = contract_suite.mint_worker_instance_id()

_TS = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)


class _Spec:
    specification_type = "demo"
    specification_version = "v1"
    specification_fingerprint = "sha256:spec"


class _AdaptPort:
    def __init__(self, inner: ScopedIntegrationAdaptationService) -> None:
        self._inner = inner

    def adapt(self, request: ScopedIntegrationAdaptationRequest):
        return self._inner.adapt(request)


class _Strategy:
    @property
    def strategy_id(self) -> str:
        return "strategy-1"

    def supports(self, request, target) -> bool:
        return True

    def adapt(self, request, target):
        scope = request.scope
        return build_scoped_integration_adaptation_artifact(
            artifact_id="art-1",
            tenant_id=scope.tenant_id,
            integration_category=scope.integration_category,
            provider_id=scope.provider_id,
            resource_scope=scope.resource_scope,
            strategy_id=self.strategy_id,
            candidate_id=scope.candidate_id,
            candidate_revision=scope.candidate_revision,
            scope=scope,
            specification=_Spec(),
        )


def _decision() -> WorkerCapabilityAcquisitionDecision:
    candidate = WorkerCapabilityCandidate(
        candidate_id="cand-1",
        candidate_kind=WorkerCapabilityCandidateKind.ADAPTIVE_INTEGRATION,
        capability_ref="integration:demo",
        source_domain="integrations",
        operations=("READ_CONFIGURATION",),
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
        provider_id="provider-1",
        resource_scope="rs-1",
        permitted_operations=(ScopedIntegrationAdaptationOperation.READ_CONFIGURATION,),
        network_allowlist=NetworkEgressAllowlist(hosts=()),
        credential_grant_ref="grant-1",
        expires_at=_TS + timedelta(hours=1),
        candidate_id="cand-1",
        candidate_revision="rev-1",
    )


def test_valid_a2_path_qualification_pending() -> None:
    decision = _decision()
    candidate = decision.selected_candidate
    assert candidate is not None
    port = _AdaptPort(ScopedIntegrationAdaptationService(strategies=(_Strategy(),)))
    service = WorkerScopedAdaptiveIntegrationOrchestrationService(adaptation_port=port)
    request = ScopedAdaptiveIntegrationExecutionRequest(
        worker_instance_id=_WORKER_ID,
        correlation=ScopedAdaptiveIntegrationCorrelation(tenant_id="tenant-a"),
        acquisition_decision=decision,
        selected_candidate=candidate,
        need_id="need-1",
        recovery_decision_id="rec-1",
        integration_capability_ref="integration:demo",
        required_operations=("READ_CONFIGURATION",),
        adaptation_scope=_scope(),
        requested_at=_TS,
    )
    result = service.prepare(request)
    assert (
        result.status is ScopedAdaptiveIntegrationPreparationStatus.QUALIFICATION_PENDING
    )
    assert result.qualification_request is not None
    assert result.qualification_subject is not None


def test_a1_candidate_rejected() -> None:
    decision = _decision()
    candidate = decision.selected_candidate
    assert candidate is not None
    a1 = WorkerCapabilityCandidate(
        candidate_id=candidate.candidate_id,
        candidate_kind=WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL,
        capability_ref=candidate.capability_ref,
        source_domain=candidate.source_domain,
        operations=candidate.operations,
        risk_class=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
        evidence_refs=(),
        discovered_at=_TS,
    )
    port = _AdaptPort(ScopedIntegrationAdaptationService(strategies=(_Strategy(),)))
    service = WorkerScopedAdaptiveIntegrationOrchestrationService(adaptation_port=port)
    request = ScopedAdaptiveIntegrationExecutionRequest(
        worker_instance_id=_WORKER_ID,
        correlation=ScopedAdaptiveIntegrationCorrelation(tenant_id="tenant-a"),
        acquisition_decision=decision,
        selected_candidate=a1,
        need_id="need-1",
        recovery_decision_id="rec-1",
        integration_capability_ref="integration:demo",
        required_operations=("READ_CONFIGURATION",),
        adaptation_scope=_scope(),
        requested_at=_TS,
    )
    result = service.prepare(request)
    assert result.status is ScopedAdaptiveIntegrationPreparationStatus.DENIED
