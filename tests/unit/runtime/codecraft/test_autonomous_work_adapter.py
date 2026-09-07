# © Artur Czarnecki. All rights reserved.

"""AW-7B — CodeCraft ephemeral capability execution adapter tests."""

from __future__ import annotations

from pathlib import Path

from datetime import UTC, datetime

import pytest

from intergrax.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionService,
)
from intergrax.codecraft.codegen_adapter import CodeGenerationAdapter
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.autonomous_work.capability_acquisition import (
    ACQUISITION_DECISION_POLICY_VERSION,
    CapabilityAcquisitionDisposition,
    CapabilityAcquisitionReasonCode,
    CapabilityNeedKind,
    WorkerAutonomyLevel,
    WorkerCapabilityAcquisitionDecision,
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
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    CodecraftProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import ProblemReference
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.codecraft.autonomous_work_adapter import (
    CodeCraftEphemeralCapabilityExecutionAdapter,
)
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.ownership import codecraft_exec_hitl_notes
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry.wiring import ToolWiringContext
from testing_support.codecraft_execution_environment import codecraft_sandbox_execution_profile
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

TENANT = "tenant-a"
TASK = "task-a"
_UTC = UTC
_NOW = datetime(2026, 9, 7, 10, 0, tzinfo=_UTC)
_WORKER_ID = contract_suite.mint_worker_instance_id()
_EVIDENCE = ProblemReference("problem/evidence/adapter-1")
_CAPABILITY_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)
_CODECRAFT_PROFILE = CodecraftProfileRef(
    profile_id="codecraft/default",
    version=initial_profile_version(),
)
_OPERATION = "document.parse_csv"
_CODECRAFT_OPS = frozenset(
    {"echo", "write_file", "read_file", "list_files", "run_python", "run_script"},
)


class UnsafeCodegenAdapter:
    def generate(self, *, goal: str, constraints: str = "", language: str = "python") -> str:
        del goal, constraints, language
        return 'eval("unsafe")\n'

    def patch(
        self,
        *,
        goal: str,
        code: str,
        diagnostics: str,
        language: str = "python",
    ) -> str:
        del goal, diagnostics, language
        return code


def _sandbox(tmp_path: Path) -> SandboxSession:
    return SandboxSession.create(
        tmp_path,
        tenant_id=TENANT,
        task_id=TASK,
        allowed_operations=_CODECRAFT_OPS,
    )


def _ctx(
    sandbox: SandboxSession,
    *,
    profile: CodeCraftProfile,
    codegen: CodeGenerationAdapter | None = None,
    hitl_store: InMemoryHumanDecisionPersistence | None = None,
) -> ToolWiringContext:
    extras: dict[str, object] = {
        "codecraft_profile": profile,
        "codecraft_session_manager": CodeCraftSessionManager(),
        "codecraft_ephemeral_registry": EphemeralToolRegistryStore(),
        "effective_environment_profile": codecraft_sandbox_execution_profile(),
    }
    if codegen is not None:
        extras["codecraft_codegen_adapter"] = codegen
    return ToolWiringContext(
        sandbox_session=sandbox,
        human_decision_store=hitl_store,
        extras=extras,
    )


def _a1_execution_request(
    *,
    idempotency_key: str | None = None,
    run_id=None,
) -> WorkerEphemeralCapabilityExecutionRequest:
    need = WorkerCapabilityNeed(
        worker_instance_id=_WORKER_ID,
        obstacle_id=f"{_WORKER_ID}:capability/missing:1:occurrence-1",
        need_kind=CapabilityNeedKind.TOOL_OPERATION,
        required_operations=(_OPERATION,),
        capability_profile_ref=_CAPABILITY_PROFILE,
        requested_at=_NOW,
        recovery_decision_id="recovery-a1-adapter",
        evidence_refs=(_EVIDENCE,),
        codecraft_profile_ref=_CODECRAFT_PROFILE,
    )
    candidate = WorkerCapabilityCandidate(
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
    decision = WorkerCapabilityAcquisitionDecision(
        decision_id=derive_worker_capability_acquisition_decision_id(
            worker_instance_id=_WORKER_ID,
            obstacle_id=need.obstacle_id,
            recovery_decision_id=need.recovery_decision_id,
            need_id=need_id,
            capability_profile_version=need.capability_profile_ref.version.value,
            selected_candidate_id=candidate.candidate_id,
            decision_policy_version=ACQUISITION_DECISION_POLICY_VERSION,
        ),
        worker_instance_id=_WORKER_ID,
        obstacle_id=need.obstacle_id,
        recovery_decision_id=need.recovery_decision_id,
        need_id=need_id,
        disposition=CapabilityAcquisitionDisposition.EPHEMERAL_GENERATION_CANDIDATE,
        selected_candidate=candidate,
        autonomy_level=WorkerAutonomyLevel.A1_EPHEMERAL_SAFE,
        capability_profile_ref=_CAPABILITY_PROFILE,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
        reason_code=CapabilityAcquisitionReasonCode.A1_CANDIDATE_ALLOWED,
        evidence_refs=(_EVIDENCE,),
        decided_at=_NOW,
    )
    return WorkerEphemeralCapabilityExecutionRequest(
        worker_instance_id=_WORKER_ID,
        acquisition_decision=decision,
        recovery_decision_id=decision.recovery_decision_id,
        obstacle_id=decision.obstacle_id,
        need_id=decision.need_id,
        selected_candidate=candidate,
        codecraft_profile_ref=_CODECRAFT_PROFILE,
        generation_goal="build csv parser helper",
        required_operations=(_OPERATION,),
        correlation=WorkerEphemeralCapabilityExecutionCorrelation(
            tenant_id=TENANT,
            task_id=TASK,
            run_id=run_id,
        ),
        requested_at=_NOW,
        evidence_refs=(_EVIDENCE,),
        idempotency_key=idempotency_key,
    )


def test_adapter_happy_path_verified_ephemeral_result(tmp_path: Path) -> None:
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="local",
        require_tests=False,
        max_iterations=4,
    )
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(_ctx(_sandbox(tmp_path), profile=profile))
    result = adapter.execute(_a1_execution_request())

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED
    assert result.ephemeral_capability is not None
    assert result.ephemeral_capability.craft_id
    assert result.craft_correlation == result.ephemeral_capability.craft_id


def test_adapter_static_gate_failure_no_success(tmp_path: Path) -> None:
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="local",
        require_tests=False,
        max_iterations=2,
    )
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(
        _ctx(_sandbox(tmp_path), profile=profile, codegen=UnsafeCodegenAdapter()),
    )
    result = adapter.execute(_a1_execution_request())

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.FAILED
    assert result.status is not WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED


def test_adapter_isolation_unavailable_fail_closed(tmp_path: Path) -> None:
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="cloud",
        require_tests=False,
        max_iterations=2,
    )
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(_ctx(_sandbox(tmp_path), profile=profile))
    result = adapter.execute(_a1_execution_request())

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.UNAVAILABLE


def test_adapter_hitl_pending_no_execution(tmp_path: Path) -> None:
    profile = CodeCraftProfile(
        mode="supervised",
        isolation_tier="local",
        require_hitl_before_exec=True,
        require_tests=False,
        max_iterations=2,
    )
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(
        _ctx(_sandbox(tmp_path), profile=profile, hitl_store=InMemoryHumanDecisionPersistence()),
    )
    result = adapter.execute(_a1_execution_request())

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.PENDING_HITL


def test_adapter_hitl_deny_no_retry(tmp_path: Path) -> None:
    profile = CodeCraftProfile(
        mode="supervised",
        isolation_tier="local",
        require_hitl_before_exec=True,
        require_tests=False,
        max_iterations=2,
    )
    store = InMemoryHumanDecisionPersistence()
    craft_id = "craft_hitl_deny_test"
    run_id = mint_run_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        store.record(
            build_human_decision_record(
                task_id=TASK,
                tenant_id=TENANT,
                approver=local_development_approver_evidence(tenant_id=TENANT, actor_id="operator"),
                verdict=HumanResponseVerdict.REJECT,
                response_text="no",
                run_id=str(run_id),
                notes=codecraft_exec_hitl_notes(craft_id),
            ),
        )
        adapter = CodeCraftEphemeralCapabilityExecutionAdapter(
            _ctx(_sandbox(tmp_path), profile=profile, hitl_store=store),
        )
        result = adapter.execute(
            _a1_execution_request(idempotency_key=craft_id, run_id=run_id),
        )
    finally:
        reset_active_execution_identity(token)

    assert result.status is WorkerEphemeralCapabilityExecutionStatus.DENIED


def test_service_with_adapter_boundary(tmp_path: Path) -> None:
    profile = CodeCraftProfile(
        mode="autonomous",
        isolation_tier="local",
        require_tests=False,
        max_iterations=4,
    )
    adapter = CodeCraftEphemeralCapabilityExecutionAdapter(_ctx(_sandbox(tmp_path), profile=profile))
    service = WorkerEphemeralCapabilityExecutionService(execution_port=adapter)
    result = service.execute(_a1_execution_request())
    assert result.status is WorkerEphemeralCapabilityExecutionStatus.SUCCEEDED
