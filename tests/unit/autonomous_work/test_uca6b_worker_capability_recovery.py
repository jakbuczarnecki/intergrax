# © Artur Czarnecki. All rights reserved.

"""UCA-6B — worker canonical discovery/UCA migration tests."""

from __future__ import annotations

import ast
import importlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.worker_capability_need_projection import (
    project_worker_capability_need_to_capability_need,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityNeedKind,
    WorkerCapabilityAcquisitionRequest,
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
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from tests.unit.autonomous_work import repository_contracts as contract_suite
from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    permissive_capability_policy,
)
from tests.unit.autonomous_work.uca6b_test_support import (
    build_recording_acquisition,
)

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 21, 8, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_EVIDENCE = ProblemReference("problem/evidence/uca6b-1")


class _StaticDiscovery:
    def __init__(self, completion) -> None:
        self._completion = completion
        self.calls = 0

    def complete_discovery(self, request):
        self.calls += 1
        return self._completion


def _worker_need() -> WorkerCapabilityNeed:
    recovery_id = "recovery:uca6b:1"
    return WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:obstacle:1",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=("document.parse_csv",),
        capability_profile_ref=_PROFILE,
        requested_at=_NOW,
        recovery_decision_id=recovery_id,
        evidence_refs=(_EVIDENCE,),
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
        source_ref="recovery/source/uca6b",
    )


def test_worker_need_projection_preserves_generic_fields() -> None:
    need = _worker_need()
    canonical = project_worker_capability_need_to_capability_need(need)
    assert canonical.need_id is not None
    assert canonical.kinds == (CapabilityKind.TOOL,)
    assert canonical.intent_summary == "document.parse_csv"
    assert "worker_instance_id" not in canonical.model_dump()


def test_direct_reuse_skips_uca_acquisition() -> None:
    need = _worker_need()
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
    from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind

    host_key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="builtin",
        source_kind=CapabilitySourceKind.BUILTIN,
        logical_id="document.parse_csv",
    )
    completion = build_discovery_completion(
        need_id=project_worker_capability_need_to_capability_need(need).need_id or "n",
        discovery_correlation_id="corr-direct",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        suitable_host_allowed_keys=(host_key,),
    )
    discovery = _StaticDiscovery(completion)
    bundle = build_recording_acquisition()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=bundle.service,
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    policy = permissive_capability_policy(_PROFILE)
    result = coordinator.coordinate_acquisition_decision(
        request,
        policy=policy,
        decided_at=_NOW,
    )
    assert bundle.strategy.calls == 0
    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING


def test_true_gap_invokes_uca_once() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-gap",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    discovery = _StaticDiscovery(completion)
    bundle = build_recording_acquisition(strategy_id="marketplace.gap_acquisition.v1")
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=bundle.service,
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    outcome = coordinator.coordinate_recovery(request, decided_at=_NOW)
    assert bundle.strategy.calls == 1
    assert outcome.phase is WorkerCapabilityRecoveryPhase.PENDING_QUALIFICATION
    assert outcome.provenance.gap_id is not None
    assert outcome.provenance.acquisition_strategy_id == "marketplace.gap_acquisition.v1"


def test_custom_strategy_id_opaque_to_aw_core() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-custom",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
    )
    bundle = build_recording_acquisition(strategy_id="custom.external.v1")
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=bundle.service,
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    mapped = coordinator.coordinate_acquisition_decision(
        request,
        policy=permissive_capability_policy(_PROFILE),
        decided_at=_NOW,
    )
    assert mapped.disposition is CapabilityAcquisitionDisposition.PENDING_QUALIFICATION


def test_blocked_discovery_fail_closed_no_uca() -> None:
    need = _worker_need()
    canonical_id = project_worker_capability_need_to_capability_need(need).need_id
    completion = build_discovery_completion(
        need_id=canonical_id or "need",
        discovery_correlation_id="corr-blocked",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_NOW,
        governance_blocked=True,
    )
    bundle = build_recording_acquisition()
    coordinator = WorkerCapabilityRecoveryCoordinator(
        discovery=_StaticDiscovery(completion),
        acquisition=bundle.service,
    )
    request = WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=_recovery_decision(need),
        capability_profile_ref=_PROFILE,
    )
    outcome = coordinator.coordinate_recovery(request, decided_at=_NOW)
    assert bundle.strategy.calls == 0
    assert outcome.phase is WorkerCapabilityRecoveryPhase.FAIL_CLOSED
    assert completion.outcome is DiscoveryCompletionOutcome.BLOCKED


def test_uca6b_core_modules_forbidden_marketplace_codecraft_imports() -> None:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    modules = (
        "worker_capability_recovery_coordinator.py",
        "worker_capability_need_projection.py",
        "worker_capability_recovery_ports.py",
        "catalog_canonical_discovery_service.py",
    )
    forbidden = ("marketplace", "codecraft", "runtime.execution")
    for name in modules:
        source = (base / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        joined = "\n".join(imported).lower()
        for token in forbidden:
            assert token not in joined, f"{name} imports forbidden {token}"


def test_integration_tool_registry_direct_reuse_via_catalog() -> None:
    from tests.unit.autonomous_work.test_worker_capability_acquisition import (
        _service,
        _request,
        _tool_registry,
    )

    service = _service(tool_registry=_tool_registry("document.parse_csv"))
    result = service.decide(_request())
    assert result.disposition is CapabilityAcquisitionDisposition.USE_EXISTING
