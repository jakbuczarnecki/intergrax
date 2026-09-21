# © Artur Czarnecki. All rights reserved.

"""UCA-6B-R — canonical discovery entry and qualification provenance."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    NotConfiguredApprovedAlternateDiscovery,
    NotConfiguredConfigurationOpportunityDiscovery,
    StaticWorkerCapabilityProfileResolver,
    UnavailableIntegrationCapabilityDiscovery,
    UnavailableSkillCapabilityDiscovery,
    permissive_capability_policy,
)
from intergrax.autonomous_work.capability_acquisition_service import (
    WorkerCapabilityAcquisitionDecisionService,
)
from intergrax.autonomous_work.capability_discovery_adapters import (
    ToolRegistryCapabilityDiscoveryAdapter,
)
from intergrax.autonomous_work.catalog_canonical_discovery_service import (
    CatalogCanonicalCapabilityDiscoveryService,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityNeedKind,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityDiscoveryRequest,
    WorkerCapabilityNeed,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleKind,
    WorkerRecoveryDecision,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.catalog_discovery_test_support import (
    catalog_discovery_dependencies,
    catalog_snapshot_from_registries,
    host_availability_for_entries,
    tool_catalog_entry,
)
from tests.unit.autonomous_work.test_worker_capability_acquisition import (
    _OPERATION,
    _request,
    _tool_registry,
)
from tests.unit.autonomous_work.uca6b_test_support import (
    build_recording_acquisition,
    build_test_coordinator,
)

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 21, 9, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_EVIDENCE = ProblemReference("problem/evidence/uca6b-r-1")


class _CountingToolDiscovery(ToolRegistryCapabilityDiscoveryAdapter):
    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(registry)
        self.calls = 0

    def discover(self, request: WorkerCapabilityDiscoveryRequest):
        self.calls += 1
        return super().discover(request)


class _CountingSkillDiscovery:
    calls = 0

    def discover(self, request: WorkerCapabilityDiscoveryRequest):
        _CountingSkillDiscovery.calls += 1
        return UnavailableSkillCapabilityDiscovery().discover(request)


class _StaticDiscovery:
    def __init__(self, completion) -> None:
        self._completion = completion
        self.calls = 0

    def complete_discovery(self, request):
        self.calls += 1
        return self._completion


@dataclass
class _RecordingQualification:
    calls: int = 0
    last_request: object | None = None
    outcome: CapabilityQualificationOutcome = CapabilityQualificationOutcome.FAILED

    def qualify(self, request):
        from intergrax.contracts.capability_qualification.qualification_reason_code import (
            CapabilityQualificationReasonCode,
        )

        self.calls += 1
        self.last_request = request
        return CapabilityQualificationResult(
            qualification_request_id=request.qualification_request_id,
            acquisition_request_id=request.acquisition_request_id,
            gap_id=request.gap_id,
            strategy_id=request.strategy_id,
            outcome=self.outcome,
            reason_code=CapabilityQualificationReasonCode.PROVIDER_FAILED,
            started_at=request.requested_at,
            completed_at=request.requested_at,
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


def _recovery_decision(need: WorkerCapabilityNeed) -> WorkerRecoveryDecision:
    return WorkerRecoveryDecision(
        decision_id=need.recovery_decision_id,
        obstacle_id=need.obstacle_id,
        obstacle_kind=WorkerObstacleKind.CAPABILITY_MISSING,
        strategy=RecoveryStrategy.ACQUIRE_CAPABILITY,
        decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
        source_ref="recovery/source/uca6b-r",
    )


def test_acquire_capability_skips_legacy_tool_skill_ladder() -> None:
    tool_registry = _tool_registry(_OPERATION)
    skill_registry = SkillRegistry()
    counting_tool = _CountingToolDiscovery(tool_registry)
    _CountingSkillDiscovery.calls = 0
    coordinator = build_test_coordinator(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_PROFILE),
        ),
        tool_discovery=counting_tool,
        skill_discovery=_CountingSkillDiscovery(),
        integration_discovery=UnavailableIntegrationCapabilityDiscovery(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
        canonical_recovery=coordinator,
    )
    need = _request().need
    service.decide(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=_recovery_decision(need),
            capability_profile_ref=_PROFILE,
        ),
    )
    assert counting_tool.calls == 0
    assert _CountingSkillDiscovery.calls == 0


def test_catalog_canonical_discovery_single_pass_per_kind() -> None:
    tool_registry = _tool_registry(_OPERATION)
    skill_registry = SkillRegistry()
    snapshot = catalog_snapshot_from_registries(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )
    availability = host_availability_for_entries(
        *(tool_catalog_entry(reg.contract.tool_id) for reg in tool_registry.list()),
    )
    dependencies = catalog_discovery_dependencies(
        snapshot=snapshot,
        availability_evidence=availability,
    )
    discovery = CatalogCanonicalCapabilityDiscoveryService(
        dependencies=dependencies,
        skill_registry=skill_registry,
    )
    need = _request().need
    from intergrax.autonomous_work.worker_capability_need_projection import (
        project_worker_capability_need_to_capability_need,
    )
    from intergrax.autonomous_work.worker_capability_recovery_ports import (
        CanonicalCapabilityDiscoveryRequest,
    )

    canonical_need = project_worker_capability_need_to_capability_need(need)
    calls = {"count": 0}
    original = __import__(
        "intergrax.capability_catalog.discovery",
        fromlist=["discover_capability_candidates"],
    ).discover_capability_candidates

    def _counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    with patch(
        "intergrax.autonomous_work.capability_catalog_discovery_adapters.discover_capability_candidates",
        side_effect=_counting,
    ):
        discovery.complete_discovery(
            CanonicalCapabilityDiscoveryRequest(
                capability_need=canonical_need,
                worker_need=need,
                discovery_correlation_id="corr-single-pass",
                requested_at=_NOW,
            ),
        )
    assert calls["count"] == 2


def test_qualification_provenance_matches_acquisition_result() -> None:
    from intergrax.contracts.capability_acquisition.acquisition_outcome import (
        CapabilityAcquisitionOutcome,
    )
    from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
        CapabilityAcquisitionReasonCode,
    )
    from intergrax.contracts.capability_acquisition.acquisition_result import (
        CapabilityAcquisitionResult,
    )
    from intergrax.contracts.capability_acquisition.acquisition_evidence import (
        CapabilityAcquisitionEvidence,
    )

    need = _request().need
    from intergrax.autonomous_work.worker_capability_need_projection import (
        project_worker_capability_need_to_capability_need,
    )

    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-qual",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    bundle = build_recording_acquisition(strategy_id="custom.external.v1")

    class _FixedAcquisition:
        def acquire(self, request):
            bundle.strategy.acquire(request)
            return CapabilityAcquisitionResult(
                request_id=request.request_id,
                gap_id=request.capability_gap.gap_id,
                strategy_id="custom.external.v1",
                outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
                reason_code=CapabilityAcquisitionReasonCode.NONE,
                started_at=request.requested_at,
                completed_at=request.requested_at,
                evidence=CapabilityAcquisitionEvidence(
                    artifact_reference="artifact://custom/test",
                ),
                correlation_id="corr-A",
                causation_id="cause-B",
            )

    qualification = _RecordingQualification()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=_FixedAcquisition(),
        qualification=qualification,
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    outcome = coordinator.coordinate_recovery(request, decided_at=_NOW)
    assert outcome.phase is WorkerCapabilityRecoveryPhase.FAIL_CLOSED
    assert qualification.calls == 1
    assert qualification.last_request is not None
    assert qualification.last_request.correlation_id == "corr-A"
    assert qualification.last_request.causation_id == "cause-B"


def test_qualification_null_provenance_propagates() -> None:
    from intergrax.contracts.capability_acquisition.acquisition_outcome import (
        CapabilityAcquisitionOutcome,
    )
    from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
        CapabilityAcquisitionReasonCode,
    )
    from intergrax.contracts.capability_acquisition.acquisition_result import (
        CapabilityAcquisitionResult,
    )
    from intergrax.contracts.capability_acquisition.acquisition_evidence import (
        CapabilityAcquisitionEvidence,
    )

    need = _request().need
    from intergrax.autonomous_work.worker_capability_need_projection import (
        project_worker_capability_need_to_capability_need,
    )

    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-null",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )

    class _NullProvenanceAcquisition:
        def acquire(self, request):
            return CapabilityAcquisitionResult(
                request_id=request.request_id,
                gap_id=request.capability_gap.gap_id,
                strategy_id="custom.external.v1",
                outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
                reason_code=CapabilityAcquisitionReasonCode.NONE,
                started_at=request.requested_at,
                completed_at=request.requested_at,
                evidence=CapabilityAcquisitionEvidence(
                    artifact_reference="artifact://custom/test",
                ),
                correlation_id=None,
                causation_id=None,
            )

    qualification = _RecordingQualification()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=_NullProvenanceAcquisition(),
        qualification=qualification,
    )
    coordinator.coordinate_recovery(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=_recovery_decision(need),
            capability_profile_ref=_PROFILE,
        ),
        decided_at=_NOW,
    )
    assert qualification.last_request is not None
    assert qualification.last_request.correlation_id is None
    assert qualification.last_request.causation_id is None


def test_unavailable_discovery_does_not_invoke_uca() -> None:
    need = _request().need
    from intergrax.autonomous_work.worker_capability_need_projection import (
        project_worker_capability_need_to_capability_need,
    )

    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-unavail",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        unavailable=True,
    )
    bundle = build_recording_acquisition()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=bundle.service,
    )
    outcome = coordinator.coordinate_recovery(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=_recovery_decision(need),
            capability_profile_ref=_PROFILE,
        ),
        decided_at=_NOW,
    )
    assert bundle.strategy.calls == 0
    assert outcome.phase is WorkerCapabilityRecoveryPhase.FAIL_CLOSED
    assert completion.outcome is DiscoveryCompletionOutcome.UNAVAILABLE


def test_external_integration_acquire_uses_canonical_not_a3_synthetic() -> None:
    from dataclasses import replace

    from intergrax.autonomous_work.capability_acquisition_ports import (
        StaticCodecraftProfileResolver,
    )

    policy = permissive_capability_policy(_PROFILE)
    restricted = replace(
        policy,
        generated_capability_allowed=False,
        adaptive_integration_allowed=False,
        durable_change_allowed=True,
    )
    coordinator = build_test_coordinator(
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
        acquisition=build_recording_acquisition(),
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(restricted),
        tool_discovery=UnavailableIntegrationCapabilityDiscovery(),
        skill_discovery=UnavailableSkillCapabilityDiscovery(),
        integration_discovery=UnavailableIntegrationCapabilityDiscovery(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
        codecraft_profile_resolver=StaticCodecraftProfileResolver(allowed=False),
        canonical_recovery=coordinator,
    )
    need = _request(need_kind=CapabilityNeedKind.EXTERNAL_INTEGRATION).need
    result = service.decide(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=_recovery_decision(need),
            capability_profile_ref=_PROFILE,
        ),
    )
    assert (
        result.disposition
        is not CapabilityAcquisitionDisposition.PRODUCTION_CHANGE_REQUIRED
    )
    assert result.disposition in {
        CapabilityAcquisitionDisposition.PENDING_QUALIFICATION,
        CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY,
    }


def test_catalog_canonical_discovery_no_broad_exception_continue() -> None:
    path = Path(
        __import__(
            "intergrax.autonomous_work.catalog_canonical_discovery_service",
            fromlist=["__file__"],
        ).__file__,
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        if isinstance(node.type, ast.Name) and node.type.id == "Exception":
            for child in ast.walk(node):
                if isinstance(child, ast.Continue):
                    pytest.fail(
                        "broad except Exception: continue forbidden in canonical discovery"
                    )
