# © Artur Czarnecki. All rights reserved.

"""UCA-6B-R2 — public discovery seam and fail-closed authority wiring."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    FailClosedWorkerCapabilityAuthorityCompatibilityPort,
    WorkerCapabilityAuthorityCompatibilityPort,
    permissive_capability_policy,
)
from intergrax.autonomous_work.capability_catalog_discovery_adapters import (
    CatalogGovernedDiscoveryLayerResult,
    CapabilityCatalogGovernedDiscoveryService,
    SkillRegistryManifestLookup,
)
from intergrax.autonomous_work.catalog_canonical_discovery_service import (
    CatalogCanonicalCapabilityDiscoveryService,
)
from intergrax.autonomous_work.worker_capability_need_projection import (
    project_worker_capability_need_to_capability_need,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.autonomous_work.worker_capability_recovery_ports import (
    CanonicalCapabilityDiscoveryRequest,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityAuthorityCompatibility,
    WorkerCapabilityDiscoveryLayerOutcome,
    WorkerCapabilityDiscoveryRequest,
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
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.skills.registry.runtime import SkillRegistry
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
from tests.unit.autonomous_work.uca6b_test_support import build_recording_acquisition

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 21, 10, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_EVIDENCE = ProblemReference("problem/evidence/uca6b-r2-1")


def _recovery_decision(need) -> WorkerRecoveryDecision:
    return WorkerRecoveryDecision(
        decision_id=need.recovery_decision_id,
        obstacle_id=need.obstacle_id,
        obstacle_kind=WorkerObstacleKind.CAPABILITY_MISSING,
        strategy=RecoveryStrategy.ACQUIRE_CAPABILITY,
        decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
        source_ref="recovery/source/uca6b-r2",
    )


def _direct_reuse_completion(need):
    host_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    return build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-r2-direct",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        suitable_host_allowed_keys=(host_key,),
    )


class _StaticDiscovery:
    def __init__(self, completion) -> None:
        self._completion = completion

    def complete_discovery(self, request):
        return self._completion


@dataclass
class _RecordingGovernedDiscovery:
    tool_calls: int = 0
    skill_calls: int = 0
    federation: CapabilityCatalogFederationCompleteness = (
        CapabilityCatalogFederationCompleteness.COMPLETE
    )

    def discover_tool(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> CatalogGovernedDiscoveryLayerResult:
        del request
        self.tool_calls += 1
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return CatalogGovernedDiscoveryLayerResult(
            outcome=WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            ),
        )

    def discover_skill(
        self,
        request: WorkerCapabilityDiscoveryRequest,
    ) -> CatalogGovernedDiscoveryLayerResult:
        del request
        self.skill_calls += 1
        from intergrax.contracts.autonomous_work.capability_acquisition import (
            CapabilityDiscoveryDisposition,
        )

        return CatalogGovernedDiscoveryLayerResult(
            outcome=WorkerCapabilityDiscoveryLayerOutcome(
                disposition=CapabilityDiscoveryDisposition.NO_MATCH,
            ),
        )

    @property
    def federation_completeness(self) -> CapabilityCatalogFederationCompleteness:
        return self.federation


def test_catalog_canonical_service_no_private_discovery_imports() -> None:
    path = Path(
        __import__(
            "intergrax.autonomous_work.catalog_canonical_discovery_service",
            fromlist=["__file__"],
        ).__file__,
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if (
            node.module
            != "intergrax.autonomous_work.capability_catalog_discovery_adapters"
        ):
            continue
        for alias in node.names:
            assert not alias.name.startswith("_"), alias.name


def test_custom_governed_discovery_port_pluginable() -> None:
    recording = _RecordingGovernedDiscovery()
    service = CatalogCanonicalCapabilityDiscoveryService(
        governed_discovery=recording,
    )
    need = _request().need
    canonical_need = project_worker_capability_need_to_capability_need(need)
    service.complete_discovery(
        CanonicalCapabilityDiscoveryRequest(
            capability_need=canonical_need,
            worker_need=need,
            discovery_correlation_id="corr-plugin",
            requested_at=_NOW,
        ),
    )
    assert recording.tool_calls == 1
    assert recording.skill_calls == 1


def test_default_governed_discovery_single_pass_per_kind() -> None:
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
    governed = CapabilityCatalogGovernedDiscoveryService(
        dependencies,
        manifest_lookup=SkillRegistryManifestLookup(skill_registry),
    )
    discovery = CatalogCanonicalCapabilityDiscoveryService(
        governed_discovery=governed,
    )
    need = _request().need
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
                discovery_correlation_id="corr-r2-single",
                requested_at=_NOW,
            ),
        )
    assert calls["count"] == 2


def test_fail_closed_authority_blocks_direct_reuse_without_wiring() -> None:
    need = _request().need
    completion = _direct_reuse_completion(need)
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=build_recording_acquisition().service,
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    result = coordinator.coordinate_acquisition_decision(
        request,
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert result.disposition is CapabilityAcquisitionDisposition.UNAVAILABLE


def test_explicit_allow_all_permits_use_existing() -> None:
    need = _request().need
    completion = _direct_reuse_completion(need)
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=build_recording_acquisition().service,
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    result = coordinator.coordinate_acquisition_decision(
        request,
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING


class _AuthorityChangePort:
    def assess(self, *, worker_instance_id, candidate):
        del worker_instance_id, candidate
        return WorkerCapabilityAuthorityCompatibility.AUTHORITY_CHANGE_REQUIRED


class _CustomAuthorityPort(WorkerCapabilityAuthorityCompatibilityPort):
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, *, worker_instance_id, candidate):
        del worker_instance_id, candidate
        self.calls += 1
        return WorkerCapabilityAuthorityCompatibility.COMPATIBLE


def test_authority_change_required_mapping() -> None:
    need = _request().need
    completion = _direct_reuse_completion(need)
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=build_recording_acquisition().service,
        authority_compatibility=_AuthorityChangePort(),
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    result = coordinator.coordinate_acquisition_decision(
        request,
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert (
        result.disposition is CapabilityAcquisitionDisposition.AUTHORITY_CHANGE_REQUIRED
    )
    assert (
        result.decision is not None
        and result.decision.reason_code
        is CapabilityAcquisitionReasonCode.A4_AUTHORITY_CHANGE_REQUIRED
    )


def test_custom_authority_port_pluginable() -> None:
    need = _request().need
    completion = _direct_reuse_completion(need)
    authority = _CustomAuthorityPort()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=build_recording_acquisition().service,
        authority_compatibility=authority,
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    result = coordinator.coordinate_acquisition_decision(
        request,
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert authority.calls == 1
    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING


class _RecordingAuthority:
    def __init__(self) -> None:
        self.calls = 0

    def assess(self, *, worker_instance_id, candidate):
        del worker_instance_id, candidate
        self.calls += 1
        return WorkerCapabilityAuthorityCompatibility.COMPATIBLE


def test_authority_not_called_for_missing_capability_gap() -> None:
    need = _request().need
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-gap",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    authority = _RecordingAuthority()
    bundle = build_recording_acquisition()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=bundle.service,
        authority_compatibility=authority,
    )
    coordinator.coordinate_acquisition_decision(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=_recovery_decision(need),
            capability_profile_ref=_PROFILE,
        ),
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert authority.calls == 0
    assert bundle.strategy.calls == 1


def test_authority_not_called_for_blocked_discovery() -> None:
    need = _request().need
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-blocked",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        governance_blocked=True,
    )
    authority = _RecordingAuthority()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=build_recording_acquisition().service,
        authority_compatibility=authority,
    )
    result = coordinator.coordinate_acquisition_decision(
        WorkerCapabilityAcquisitionRequest(
            need=need,
            recovery_decision=_recovery_decision(need),
            capability_profile_ref=_PROFILE,
        ),
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert authority.calls == 0
    assert completion.outcome is DiscoveryCompletionOutcome.BLOCKED
    assert result.disposition is CapabilityAcquisitionDisposition.NO_SAFE_CAPABILITY


def test_fail_closed_authority_port_returns_unavailable() -> None:
    from intergrax.contracts.autonomous_work.capability_acquisition import (
        WorkerAutonomyLevel,
        WorkerCapabilityCandidate,
        WorkerCapabilityCandidateKind,
        derive_worker_capability_candidate_id,
    )

    port = FailClosedWorkerCapabilityAuthorityCompatibilityPort()
    candidate = WorkerCapabilityCandidate(
        candidate_id=derive_worker_capability_candidate_id(
            candidate_kind=WorkerCapabilityCandidateKind.TOOL,
            capability_ref="tool:builtin:builtin:document.parse_csv",
        ),
        candidate_kind=WorkerCapabilityCandidateKind.TOOL,
        capability_ref="tool:builtin:builtin:document.parse_csv",
        source_domain="builtin",
        operations=("document.parse_csv",),
        risk_class=WorkerAutonomyLevel.A0_KNOWN_CAPABILITY,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    assert (
        port.assess(worker_instance_id=_WORKER_ID, candidate=candidate)
        is WorkerCapabilityAuthorityCompatibility.UNAVAILABLE
    )
