# © Artur Czarnecki. All rights reserved.

"""AW-7B integration — AW-7A decision to CodeCraft verified ephemeral result."""

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
from intergrax.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionService,
)
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.autonomous_work.capability_acquisition import (
    CapabilityAcquisitionDisposition,
    CapabilityNeedKind,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionRequest,
    WorkerCapabilityCandidateKind,
    WorkerCapabilityNeed,
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
from intergrax.runtime.codecraft.autonomous_work_adapter import (
    CodeCraftEphemeralCapabilityExecutionAdapter,
)
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.codecraft_execution_environment import codecraft_sandbox_execution_profile
from tests.unit.autonomous_work import repository_contracts as contract_suite
from tests.unit.autonomous_work.catalog_discovery_test_support import catalog_tool_skill_adapters

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


def _acquisition_request(recovery: WorkerRecoveryDecision) -> WorkerCapabilityAcquisitionRequest:
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


def _acquisition_service() -> WorkerCapabilityAcquisitionDecisionService:
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


def _craft_context(tmp_path: Path) -> ToolWiringContext:
    sandbox = SandboxSession.create(
        tmp_path,
        tenant_id=_TENANT,
        task_id=_TASK,
        allowed_operations=_CODECRAFT_OPS,
    )
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="local",
        require_tests=False,
        max_iterations=4,
    )
    return ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_profile": profile,
            "codecraft_session_manager": CodeCraftSessionManager(),
            "codecraft_ephemeral_registry": EphemeralToolRegistryStore(),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )


def test_aw7a_to_aw7b_verified_ephemeral_closed_loop(tmp_path: Path) -> None:
    recovery = _recovery_decision()
    acquisition = _acquisition_service().decide(_acquisition_request(recovery))

    assert acquisition.disposition is CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE
    assert acquisition.decision is not None
    assert acquisition.decision.autonomy_level is WorkerAutonomyLevel.A1_EPHEMERAL_SAFE
    assert acquisition.decision.selected_candidate is not None
    assert (
        acquisition.decision.selected_candidate.candidate_kind
        is WorkerCapabilityCandidateKind.CODECRAFT_EPHEMERAL
    )

    decision = acquisition.decision
    execution_request = WorkerEphemeralCapabilityExecutionRequest(
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
        ),
        requested_at=_NOW,
        evidence_refs=(_EVIDENCE,),
        constraints="parser helper only; no network",
    )
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(_craft_context(tmp_path))
    service = WorkerEphemeralCapabilityExecutionService(execution_port=adapter)
    result = service.execute(execution_request)

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED
    assert result.ephemeral_capability is not None
    assert result.craft_correlation == result.ephemeral_capability.craft_id
    assert result.need_id == derive_worker_capability_need_id(_acquisition_request(recovery).need)
