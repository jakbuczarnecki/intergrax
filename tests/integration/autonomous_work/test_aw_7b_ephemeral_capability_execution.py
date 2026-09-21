# © Artur Czarnecki. All rights reserved.

"""AW-7B integration — canonical UCA acquisition vs CodeCraft runtime lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.capability_acquisition_ports import (
    AllowAllAuthorityCompatibilityPort,
    NotConfiguredApprovedAlternateDiscovery,
    NotConfiguredConfigurationOpportunityDiscovery,
    StaticCodecraftProfileResolver,
    StaticWorkerCapabilityProfileResolver,
    permissive_capability_policy,
)
from intergrax.autonomous_work.capability_acquisition_service import (
    WorkerCapabilityAcquisitionDecisionService,
)
from intergrax.autonomous_work.capability_discovery_adapters import (
    IntegrationCatalogCapabilityDiscoveryAdapter,
)
from intergrax.autonomous_work.capability_catalog_discovery_adapters import (
    CapabilityCatalogGovernedDiscoveryService,
    SkillRegistryManifestLookup,
)
from intergrax.autonomous_work.catalog_canonical_discovery_service import (
    CatalogCanonicalCapabilityDiscoveryService,
)
from intergrax.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionService,
)
from intergrax.autonomous_work.worker_capability_recovery_coordinator import (
    WorkerCapabilityRecoveryCoordinator,
)
from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.permit_acquisition_authorization import (
    PermitCapabilityAcquisitionAuthorizationPort,
)
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.autonomous_work.capability_acquisition import (
    ACQUISITION_DECISION_POLICY_VERSION,
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityNeedKind,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityCandidate,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityNeed,
    derive_worker_capability_acquisition_decision_id,
    derive_worker_capability_need_id,
)
from intergrax.contracts.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionCorrelation,
    WorkerEphemeralCapabilityExecutionRequest,
    WorkerEphemeralCapabilityExecutionStatus,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleKind,
    WorkerObstacleSourceKind,
    WorkerRecoveryDecision,
    derive_recovery_decision_id,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryPhase,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    DiscoveryCompletionOutcome,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.codecraft.acquisition import (
    CodeCraftGapCapabilityAcquisitionStrategy,
    CodeCraftOrchestratorGapSynthesisPort,
)
from intergrax.runtime.codecraft.autonomous_work_adapter import (
    CodeCraftEphemeralCapabilityExecutionAdapter,
)
from intergrax.runtime.codecraft.ephemeral_registry import (
    EphemeralToolRegistryStore,
    get_ephemeral_registry_store,
)
from intergrax.runtime.codecraft.orchestrator import CodeCraftOrchestrator
from intergrax.runtime.codecraft.ownership import (
    codecraft_exec_hitl_notes,
    resolve_codecraft_ownership,
)
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.human.models import (
    HumanResponseVerdict,
    build_human_decision_record,
)
from intergrax.runtime.human.persistence_contract import (
    InMemoryHumanDecisionPersistence,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.codecraft_execution_environment import (
    codecraft_sandbox_execution_profile,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.catalog_discovery_test_support import (
    catalog_discovery_dependencies,
    catalog_snapshot_from_registries,
    catalog_tool_skill_adapters,
    host_availability_for_entries,
    tool_catalog_entry,
)

pytestmark = pytest.mark.integration

_UTC = UTC
_NOW = datetime(2026, 9, 7, 11, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_EVIDENCE = ProblemReference("problem/evidence/aw7b-integration-1")
_CAPABILITY_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_CODECRAFT_PROFILE = CodecraftProfileRef(
    profile_id="codecraft/default",
    version=initial_profile_version(),
)
_OPERATION = "document.parse_csv"
_TENANT = "tenant-aw7b"
_TASK = "task-aw7b"
_CODECRAFT_OPS = frozenset(
    {"echo", "write_file", "read_file", "list_files", "run_python", "run_script"},
)


def _recovery_decision() -> WorkerRecoveryDecision:
    obstacle_id = (
        f"{_WORKER_ID}:"
        f"{WorkerObstacleSourceKind.CAPABILITY_RESOLUTION.value}:"
        f"capability/missing/integration:occurrence-1"
    )
    return WorkerRecoveryDecision(
        decision_id=derive_recovery_decision_id(obstacle_id),
        obstacle_id=obstacle_id,
        obstacle_kind=WorkerObstacleKind.CAPABILITY_MISSING,
        strategy=RecoveryStrategy.ACQUIRE_CAPABILITY,
        decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
        source_ref="recovery/source/integration",
    )


def _acquisition_request(
    recovery: WorkerRecoveryDecision,
) -> WorkerCapabilityAcquisitionRequest:
    need = WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=recovery.obstacle_id,
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=(_OPERATION,),
        capability_profile_ref=_CAPABILITY_PROFILE,
        requested_at=_NOW,
        recovery_decision_id=recovery.decision_id,
        evidence_refs=(_EVIDENCE,),
        codecraft_profile_ref=_CODECRAFT_PROFILE,
    )
    return WorkerCapabilityAcquisitionRequest(
        need=need,
        recovery_decision=recovery,
        capability_profile_ref=_CAPABILITY_PROFILE,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
    )


def _unwired_acquisition_service() -> WorkerCapabilityAcquisitionDecisionService:
    tool_discovery, skill_discovery = catalog_tool_skill_adapters(
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
    )
    return WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=tool_discovery,
        skill_discovery=skill_discovery,
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
        codecraft_profile_resolver=StaticCodecraftProfileResolver(allowed=True),
    )


def _build_codecraft_canonical_coordinator(
    ctx: ToolWiringContext,
) -> WorkerCapabilityRecoveryCoordinator:
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    snapshot = catalog_snapshot_from_registries(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
    )
    host_entries = [
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id
        in {reg.contract.tool_id for reg in tool_registry.list()}
        or entry.identity.logical.logical_id
        in {reg.manifest.skill_id for reg in skill_registry.list()}
    ]
    availability = (
        host_availability_for_entries(*host_entries) if host_entries else None
    )
    if availability is None:
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
    synthesis_port = CodeCraftOrchestratorGapSynthesisPort(ctx)
    strategy = CodeCraftGapCapabilityAcquisitionStrategy(synthesis_port)
    acquisition = CapabilityAcquisitionService(
        (strategy,),
        authorization=PermitCapabilityAcquisitionAuthorizationPort(),
    )
    return WorkerCapabilityRecoveryCoordinator(
        discovery=discovery,
        acquisition=acquisition,
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
    )


def _canonical_acquisition_service(
    ctx: ToolWiringContext,
) -> tuple[
    WorkerCapabilityAcquisitionDecisionService, WorkerCapabilityRecoveryCoordinator
]:
    coordinator = _build_codecraft_canonical_coordinator(ctx)
    tool_discovery, skill_discovery = catalog_tool_skill_adapters(
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
    )
    service = WorkerCapabilityAcquisitionDecisionService(
        profile_resolver=StaticWorkerCapabilityProfileResolver(
            permissive_capability_policy(_CAPABILITY_PROFILE),
        ),
        tool_discovery=tool_discovery,
        skill_discovery=skill_discovery,
        integration_discovery=IntegrationCatalogCapabilityDiscoveryAdapter(),
        approved_alternate_discovery=NotConfiguredApprovedAlternateDiscovery(),
        configuration_discovery=NotConfiguredConfigurationOpportunityDiscovery(),
        authority_compatibility=AllowAllAuthorityCompatibilityPort(),
        codecraft_profile_resolver=StaticCodecraftProfileResolver(allowed=True),
        canonical_recovery=coordinator,
    )
    return service, coordinator


def _craft_context(
    tmp_path: Path,
    *,
    hitl_store: InMemoryHumanDecisionPersistence | None = None,
    profile: CodeCraftProfile | None = None,
) -> ToolWiringContext:
    sandbox = SandboxSession.create(
        tmp_path,
        tenant_id=_TENANT,
        task_id=_TASK,
        allowed_operations=_CODECRAFT_OPS,
    )
    resolved_profile = profile or CodeCraftProfile(
        mode="autonomous",
        isolation_tier="local",
        require_tests=False,
        max_iterations=4,
        forbidden_imports=["os"],
    )
    return ToolWiringContext(
        sandbox_session=sandbox,
        human_decision_store=hitl_store,
        extras={
            "codecraft_profile": resolved_profile,
            "codecraft_session_manager": CodeCraftSessionManager(),
            "codecraft_ephemeral_registry": EphemeralToolRegistryStore(),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )


def _legacy_runtime_codecraft_decision() -> WorkerCapabilityAcquisitionDecision:
    """Runtime-only legacy AW-7B execution binding — not canonical generic acquisition."""
    need = _acquisition_request(_recovery_decision()).need
    selected = WorkerCapabilityCandidate(
        candidate_id="CODECRAFT_EPHEMERAL:ephemeral:codecraft",
        candidate_kind=WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL,
        capability_ref="ephemeral:codecraft",
        source_domain="autonomous_work",
        operations=(_OPERATION,),
        risk_class=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
        evidence_refs=(_EVIDENCE,),
        discovered_at=_NOW,
    )
    need_id = derive_worker_capability_need_id(need)
    return WorkerCapabilityAcquisitionDecision(
        decision_id=derive_worker_capability_acquisition_decision_id(
            worker_instance_id=_WORKER_ID,
            obstacle_id=need.obstacle_id,
            recovery_decision_id=need.recovery_decision_id,
            need_id=need_id,
            capability_profile_version=need.capability_profile_ref.version.value,
            selected_candidate_id=selected.candidate_id,
            decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
        ),
        worker_instance_id=_WORKER_ID,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        disposition=CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE,
        selected_candidate=selected,
        autonomy_level=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
        capability_profile_ref=_CAPABILITY_PROFILE,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A1_CANDIDATE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
    )


def _execution_request(
    decision: WorkerCapabilityAcquisitionDecision,
    *,
    idempotency_key: str | None = None,
    run_id=None,
) -> WorkerEphemeralCapabilityExecutionRequest:
    assert decision.selected_candidate is not None
    return WorkerEphemeralCapabilityExecutionRequest(
        worker_instance_id=_WORKER_ID,
        acquisition_decision=decision,
        recovery_decision_id=decision.recovery_decision_id,
        obstacle_id=decision.obstacle_id,
        need_id=decision.need_id,
        selected_candidate=decision.selected_candidate,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
        generation_goal="generate csv parser helper for document.parse_csv",
        required_operations=(_OPERATION,),
        correlation=WorkerEphemeralCapabilityExecutionCorrelation(
            tenant_id=_TENANT,
            task_id=_TASK,
            run_id=run_id,
        ),
        requested_at=_NOW,
        evidence_refs=(_EVIDENCE,),
        constraints="parser helper only; no network",
        idempotency_key=idempotency_key,
    )


def _approve_hitl(
    store: InMemoryHumanDecisionPersistence,
    *,
    craft_id: str,
    run_id: str | None = None,
) -> None:
    store.record(
        build_human_decision_record(
            task_id=_TASK,
            tenant_id=_TENANT,
            approver=local_development_approver_evidence(
                tenant_id=_TENANT, actor_id="operator"
            ),
            verdict=HumanResponseVerdict.APPROVE,
            response_text="approved",
            run_id=run_id,
            notes=codecraft_exec_hitl_notes(craft_id),
        ),
    )


def _ownership(ctx: ToolWiringContext):
    return resolve_codecraft_ownership(
        ctx,
        caller_tenant_id=_TENANT,
        caller_task_id=_TASK,
    )


def test_integration_missing_canonical_recovery_fail_closed() -> None:
    result = _unwired_acquisition_service().decide(
        _acquisition_request(_recovery_decision())
    )

    assert result.disposition is CapabilityAcquisitionDisposition.UNAVAILABLE
    assert result.decision is not None
    assert (
        result.decision.reason_code
        is CapabilityAcquisitionReasonCode.CANONICAL_UCA_NOT_CONFIGURED
    )


def test_integration_canonical_acquire_not_ephemeral_generation_candidate(
    tmp_path: Path,
) -> None:
    ctx = _craft_context(tmp_path)
    service, _ = _canonical_acquisition_service(ctx)
    result = service.decide(_acquisition_request(_recovery_decision()))

    assert (
        result.disposition
        is not CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE
    )


def test_integration_canonical_acquire_not_codecraft_ephemeral_candidate(
    tmp_path: Path,
) -> None:
    ctx = _craft_context(tmp_path)
    service, _ = _canonical_acquisition_service(ctx)
    result = service.decide(_acquisition_request(_recovery_decision()))

    decision = result.decision
    assert decision is not None
    assert decision.selected_candidate is None or (
        decision.selected_candidate.candidate_kind
        is not WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL
    )


def test_aw_canonical_codecraft_acquisition_to_qualification_boundary(
    tmp_path: Path,
) -> None:
    ctx = _craft_context(tmp_path)
    _, coordinator = _canonical_acquisition_service(ctx)
    request = _acquisition_request(_recovery_decision())

    recovery = coordinator.coordinate_recovery(request, decided_at=_NOW)

    assert recovery.discovery_completion is not None
    assert (
        recovery.discovery_completion.outcome
        is DiscoveryCompletionOutcome.MISSING_CAPABILITY
    )
    assert recovery.acquisition_result is not None
    assert recovery.acquisition_result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED
    evidence = recovery.acquisition_result.evidence
    assert evidence is not None
    assert evidence.artifact_reference is not None
    assert evidence.artifact_reference.startswith("codecraft:artifact:")
    assert recovery.phase is WorkerCapabilityRecoveryPhase.PENDING_QUALIFICATION


def test_aw_canonical_acquire_maps_pending_qualification_on_decide_surface(
    tmp_path: Path,
) -> None:
    ctx = _craft_context(tmp_path)
    service, _ = _canonical_acquisition_service(ctx)
    aw_result = service.decide(_acquisition_request(_recovery_decision()))

    assert (
        aw_result.disposition is CapabilityAcquisitionDisposition.PENDING_QUALIFICATION
    )
    assert aw_result.decision is not None
    assert (
        aw_result.decision.reason_code
        is CapabilityAcquisitionReasonCode.CANONICAL_GAP_ACQUIRED_PENDING_QUALIFICATION
    )


def test_codecraft_ephemeral_runtime_verified_closed_loop(tmp_path: Path) -> None:
    decision = _legacy_runtime_codecraft_decision()
    execution_request = _execution_request(decision)
    ctx = _craft_context(tmp_path)
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(ctx)
    service = WorkerEphemeralCapabilityExecutionService(execution_port=adapter)
    result = service.execute(execution_request)

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED
    assert result.ephemeral_capability is not None
    assert result.craft_correlation == result.ephemeral_capability.craft_id
    assert result.need_id == derive_worker_capability_need_id(
        _acquisition_request(_recovery_decision()).need,
    )

    craft_id = result.ephemeral_capability.craft_id
    ephemeral_tool_id = result.ephemeral_capability.ephemeral_tool_id
    assert ephemeral_tool_id is not None

    registry_store = get_ephemeral_registry_store(ctx)
    assert ephemeral_tool_id in registry_store.for_craft(craft_id).list_tools()

    orch = CodeCraftOrchestrator(ctx)
    ownership = _ownership(ctx)
    live_session = orch.get_state(craft_id, ownership=ownership)
    assert live_session is not None
    assert live_session.disposed is False

    disposed_session = orch.dispose(craft_id, tenant_id=_TENANT, task_id=_TASK)
    assert disposed_session is not None
    assert orch.get_state(craft_id, ownership=ownership) is None
    assert registry_store.for_craft(craft_id).list_tools() == ()

    stale_handle = result.ephemeral_capability
    assert (
        stale_handle.ephemeral_tool_id
        not in registry_store.for_craft(
            stale_handle.craft_id,
        ).list_tools()
    )


def test_codecraft_hitl_pending_preserves_session_for_continuation(
    tmp_path: Path,
) -> None:
    craft_id = "craft_aw7b_hitl_continue"
    run_id = mint_run_id()
    hitl_store = InMemoryHumanDecisionPersistence()
    ctx = _craft_context(
        tmp_path,
        hitl_store=hitl_store,
        profile=CodeCraftProfile(
            mode="supervised",
            isolation_tier="local",
            require_hitl_before_exec=True,
            require_tests=False,
            max_iterations=4,
        ),
    )
    orch = CodeCraftOrchestrator(ctx, run_id=str(run_id))
    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        started_session, start_deny = orch.start(
            goal="generate csv parser helper for document.parse_csv",
            task_id=_TASK,
            tenant_id=_TENANT,
            constraints="parser helper only; no network",
            craft_id=craft_id,
        )
    finally:
        reset_active_execution_identity(token)

    assert start_deny is None
    assert started_session is not None
    assert started_session.craft_id == craft_id

    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        _, pending_result = orch.iterate(
            craft_id=craft_id,
            tenant_id=_TENANT,
            task_id=_TASK,
        )
    finally:
        reset_active_execution_identity(token)

    assert pending_result.error == "hitl_pending"

    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        ownership = _ownership(ctx)
        session_before = orch.get_state(craft_id, ownership=ownership)
        assert session_before is not None
        assert session_before.disposed is False

        _approve_hitl(hitl_store, craft_id=craft_id, run_id=str(run_id))
        _session, iterate_result = orch.iterate(
            craft_id=craft_id,
            tenant_id=_TENANT,
            task_id=_TASK,
        )
    finally:
        reset_active_execution_identity(token)

    assert iterate_result.error != "hitl_pending"
    assert _session is not None
    assert _session.craft_id == craft_id

    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        resumed = orch.get_state(craft_id, ownership=_ownership(ctx))
    finally:
        reset_active_execution_identity(token)
    assert resumed is not None
    assert resumed.craft_id == craft_id
