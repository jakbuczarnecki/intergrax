# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5-R5 — end-to-end governance approval evidence propagation."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    build_production_codecraft_qualified_capability_execution_handler,
)
from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityResumeOutcome,
    WorkerQualifiedCapabilityResumeRequest,
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_subject import (
    qualified_capability_subject_from_result,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution_intake import CanonicalExecutionInvocationFailed
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
)
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from intergrax.runtime.nexus.tools.governance_approval_evidence_adapter import (
    require_invocation_evidence_matches_request,
)
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.providers.sandbox.contracts import CodeExecInput
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.autonomous_work.test_uca6c_r5_r2_strict_governance_composition import (
    _RecordingMsePort,
    _codecraft_context,
    _strict_tool_wiring,
)
from intergrax.runtime.codecraft.artifact_reference import artifact_reference_for_craft
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _CRAFT_ID,
    _authority_admission,
    _qualification,
    _resume_request,
    _wiring,
)
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_high_risk_tool_approval_evidence,
    uca6c_high_risk_tool_approval_evidence_for_execution_request,
    uca6c_strict_sandbox_env_profile,
    uca6c_strict_worker_manifest,
    uca6c_strict_worker_registry,
)
from tests.unit.autonomous_work.uca6c_bound_execution_fixtures import (
    recording_codecraft_execution_handler,
)
from tests.unit.runtime.nexus.tools.test_gr10_r8_orchestration_inner_guard import (
    _RecordingGuard,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skip(
        reason="UCA-6C-R6-R3 removed legacy TIGAE worker transport; use strict HITL E2E",
    ),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERIC_EVIDENCE_PATH = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "tool_invocation_governance_approval_evidence.py"
)
_OTHER_TENANT = "tenant-other-r5r5"


def _execution_request_id_for_resume(
    resume_operation_id: str,
    qualification,
) -> str:
    subject = qualified_capability_subject_from_result(qualification)
    assert subject is not None
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_operation_id,
        qualified_subject_reference=subject.qualified_subject_reference,
    )
    return derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_operation_id,
        binding_operation_id=binding_id,
    )


def _collect_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_generic_governance_evidence_contract_has_no_declarative_import() -> None:
    source = _GENERIC_EVIDENCE_PATH.read_text(encoding="utf-8")
    imported = "\n".join(_collect_imports(source))
    assert "declarative_hitl" not in imported


def test_generic_governance_evidence_contract_has_no_nexus_import() -> None:
    source = _GENERIC_EVIDENCE_PATH.read_text(encoding="utf-8")
    imported = "\n".join(_collect_imports(source))
    assert "runtime.nexus" not in imported


_SCOPED_NO_NEXUS_PATHS = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "tool_invocation_governance_approval_evidence.py",
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "autonomous_work"
    / "worker_qualified_capability_resume.py",
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "execution"
    / "qualified_capability_execution_dispatch.py",
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "execution"
    / "qualified_capability_execution_intake.py",
    _REPO_ROOT
    / "intergrax"
    / "autonomous_work"
    / "worker_qualified_capability_resume_coordinator.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "codecraft"
    / "qualified_capability_execution_handler.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "codecraft"
    / "wiring_bound_capability_execution.py",
)


def test_uca_public_contracts_reject_consumer_root_execution_id_field() -> None:
    assert "execution_id" not in {
        f.name for f in dataclasses.fields(WorkerQualifiedCapabilityResumeRequest)
    }
    assert "execution_id" not in {
        f.name for f in dataclasses.fields(WorkerQualifiedCapabilityExecutionRequest)
    }
    assert "execution_id" not in {
        f.name for f in dataclasses.fields(QualifiedCapabilityExecutionDispatchRequest)
    }


def test_uca_dispatch_service_does_not_forward_consumer_execution_id() -> None:
    source = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "execution"
        / "qualified_capability_execution_dispatch_service.py"
    ).read_text(encoding="utf-8")
    assert "execution_id=request.execution_id" not in source


@pytest.mark.parametrize("path", _SCOPED_NO_NEXUS_PATHS, ids=lambda p: p.name)
def test_uca6c_r5_r5_changed_scope_has_no_nexus_imports(path: Path) -> None:
    joined = "\n".join(_collect_imports(path.read_text(encoding="utf-8")))
    assert "intergrax.runtime.nexus" not in joined, f"{path} imports Nexus"


def test_resume_request_rejects_wrong_tenant_evidence() -> None:
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_OTHER_TENANT,
        task_id=str(_TASK_ID),
        run_id=str(mint_run_id()),
        step_id="uca6c.bound:tenant-check",
    )
    base = _resume_request(_qualification())
    with pytest.raises(ValueError, match="tenant_id"):
        WorkerQualifiedCapabilityResumeRequest(
            worker_instance_id=base.worker_instance_id,
            worker_need_id=base.worker_need_id,
            recovery_decision_id=base.recovery_decision_id,
            provenance=base.provenance,
            acquisition_result=base.acquisition_result,
            qualification_result=base.qualification_result,
            resume_operation_id=base.resume_operation_id,
            tenant_id=base.tenant_id,
            task_id=base.task_id,
            requested_at=base.requested_at,
            requested_authority_scopes=base.requested_authority_scopes,
            governance_approval_evidence=evidence,
        )


def test_catalog_adapter_rejects_wrong_tenant_evidence() -> None:
    run_id = str(mint_run_id())
    evidence = uca6c_high_risk_tool_approval_evidence(
        tenant_id=_OTHER_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        step_id="uca6c.bound:tenant-catalog",
    )
    request = ExecutionBoundCatalogToolInvokeRequest(
        tool_id=CODE_EXEC_TOOL_ID,
        input=CodeExecInput(code="1", language="python", timeout_s=5),
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=run_id,
        agent_id="worker-uca6c-qualified",
        step_id="uca6c.bound:tenant-catalog",
        governance_approval_evidence=evidence,
    )
    with pytest.raises(ValueError, match="tenant_id mismatch"):
        require_invocation_evidence_matches_request(evidence, request)


def test_worker_resume_propagates_same_evidence_object_to_codecraft_port() -> None:
    ctx = _wiring()
    handler, execution_port = recording_codecraft_execution_handler()
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    qual = _qualification()
    request = _resume_request(qual)
    execution_request_id = _execution_request_id_for_resume(
        request.resume_operation_id,
        qual,
    )
    evidence = uca6c_high_risk_tool_approval_evidence_for_execution_request(
        execution_request_id=execution_request_id,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=str(mint_run_id()),
    )
    request = WorkerQualifiedCapabilityResumeRequest(
        worker_instance_id=request.worker_instance_id,
        worker_need_id=request.worker_need_id,
        recovery_decision_id=request.recovery_decision_id,
        provenance=request.provenance,
        acquisition_result=request.acquisition_result,
        qualification_result=request.qualification_result,
        resume_operation_id=request.resume_operation_id,
        tenant_id=request.tenant_id,
        task_id=request.task_id,
        requested_at=request.requested_at,
        requested_authority_scopes=request.requested_authority_scopes,
        governance_approval_evidence=evidence,
    )
    result = coordinator.resume(request)
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert len(execution_port._requests) == 1
    assert execution_port._requests[0].governance_approval_evidence is evidence


def test_worker_resume_strict_high_risk_success_with_evidence(tmp_path: Path) -> None:
    craft_id = _CRAFT_ID
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ctx)
    mse: _RecordingMsePort = _RecordingMsePort(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    qual = _qualification()
    base = _resume_request(qual)
    execution_request_id = _execution_request_id_for_resume(
        base.resume_operation_id,
        qual,
    )
    evidence = uca6c_high_risk_tool_approval_evidence_for_execution_request(
        execution_request_id=execution_request_id,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=str(run_id),
        agent_id="worker-uca6c-qualified",
    )
    request = WorkerQualifiedCapabilityResumeRequest(
        worker_instance_id=base.worker_instance_id,
        worker_need_id=base.worker_need_id,
        recovery_decision_id=base.recovery_decision_id,
        provenance=base.provenance,
        acquisition_result=base.acquisition_result,
        qualification_result=base.qualification_result,
        resume_operation_id=base.resume_operation_id,
        tenant_id=base.tenant_id,
        task_id=base.task_id,
        requested_at=base.requested_at,
        requested_authority_scopes=base.requested_authority_scopes,
        run_id=run_id,
        attempt_id=attempt_id,
        governance_approval_evidence=evidence,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6cr",
            principal_id="principal-uca6cr",
        ),
    )
    try:
        result = coordinator.resume(request)
    finally:
        reset_active_execution_governance_identity(gov_token)
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert mse.calls == 1
    port = handler._execution_port
    assert port.runtime_execution_calls == 1
    assert result.execution_result is not None
    minted = result.execution_result.execution_id
    assert minted is not None
    assert str(minted) != str(run_id)


def test_worker_resume_two_execution_requests_receive_distinct_execution_ids(
    tmp_path: Path,
) -> None:
    craft_id = _CRAFT_ID
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ctx)
    mse: _RecordingMsePort = _RecordingMsePort(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    run_id = mint_run_id()

    def _resume_with_evidence(recovery: str):
        qual = _qualification(artifact=artifact_reference_for_craft(craft_id))
        base = _resume_request(qual)
        base = WorkerQualifiedCapabilityResumeRequest(
            worker_instance_id=base.worker_instance_id,
            worker_need_id=base.worker_need_id,
            recovery_decision_id=recovery,
            provenance=base.provenance,
            acquisition_result=base.acquisition_result,
            qualification_result=qual,
            resume_operation_id=derive_worker_capability_resume_operation_id(
                recovery_decision_id=recovery,
                qualification_request_id=qual.qualification_request_id,
            ),
            tenant_id=base.tenant_id,
            task_id=base.task_id,
            requested_at=base.requested_at,
            requested_authority_scopes=base.requested_authority_scopes,
            run_id=run_id,
            governance_approval_evidence=uca6c_high_risk_tool_approval_evidence_for_execution_request(
                execution_request_id=_execution_request_id_for_resume(
                    derive_worker_capability_resume_operation_id(
                        recovery_decision_id=recovery,
                        qualification_request_id=qual.qualification_request_id,
                    ),
                    qual,
                ),
                tenant_id=_TENANT,
                task_id=str(_TASK_ID),
                run_id=str(run_id),
            ),
        )
        return coordinator.resume(base)

    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6cr",
            principal_id="principal-uca6cr",
        ),
    )
    try:
        first = _resume_with_evidence("recovery:r5r6:a")
        second = _resume_with_evidence("recovery:r5r6:b")
    finally:
        reset_active_execution_governance_identity(gov_token)
    assert first.execution_result is not None
    assert second.execution_result is not None
    assert first.execution_result.execution_id != second.execution_result.execution_id


def test_worker_resume_strict_without_evidence_fails_before_mse(tmp_path: Path) -> None:
    craft_id = "craft-r5r5-no-evidence"
    ctx = _codecraft_context(tmp_path, craft_id)
    manifest = uca6c_strict_worker_manifest()
    registry = uca6c_strict_worker_registry(manifest)
    tool_wiring = _strict_tool_wiring(ctx)
    mse: _RecordingMsePort = _RecordingMsePort(allow=True)
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        uca6c_strict_sandbox_env_profile(),
        caller_agent_id="worker-uca6c-qualified",
        tenant_id=_TENANT,
        manifest=manifest,
        agent_registry=registry,
        meaningful_side_effect_authorization=mse,
        canonical_inner_execution_guard=_RecordingGuard(allow=True),
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    qual = _qualification(artifact=artifact_reference_for_craft(craft_id))
    with pytest.raises(CanonicalExecutionInvocationFailed):
        coordinator.resume(_resume_request(qual))
    assert mse.calls == 0


def test_sequential_resume_second_without_evidence_does_not_reuse_prior_evidence() -> (
    None
):
    ctx = _wiring()
    handler, execution_port = recording_codecraft_execution_handler()
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    execution = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=execution,
        authority_admission=_authority_admission(),
    )
    qual = _qualification()
    base = _resume_request(qual)
    execution_request_id = _execution_request_id_for_resume(
        base.resume_operation_id,
        qual,
    )
    evidence = uca6c_high_risk_tool_approval_evidence_for_execution_request(
        execution_request_id=execution_request_id,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        run_id=str(mint_run_id()),
    )
    with_evidence = WorkerQualifiedCapabilityResumeRequest(
        worker_instance_id=base.worker_instance_id,
        worker_need_id=base.worker_need_id,
        recovery_decision_id=base.recovery_decision_id,
        provenance=base.provenance,
        acquisition_result=base.acquisition_result,
        qualification_result=base.qualification_result,
        resume_operation_id=base.resume_operation_id,
        tenant_id=base.tenant_id,
        task_id=base.task_id,
        requested_at=base.requested_at,
        requested_authority_scopes=base.requested_authority_scopes,
        governance_approval_evidence=evidence,
    )
    first = coordinator.resume(with_evidence)
    assert first.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert execution_port._requests[0].governance_approval_evidence is evidence
    recovery_2 = "recovery:uca6cr:second-no-evidence"
    qual = _qualification()
    resume_id_2 = derive_worker_capability_resume_operation_id(
        recovery_decision_id=recovery_2,
        qualification_request_id=qual.qualification_request_id,
    )
    second = WorkerQualifiedCapabilityResumeRequest(
        worker_instance_id=base.worker_instance_id,
        worker_need_id=base.worker_need_id,
        recovery_decision_id=recovery_2,
        provenance=base.provenance,
        acquisition_result=base.acquisition_result,
        qualification_result=qual,
        resume_operation_id=resume_id_2,
        tenant_id=base.tenant_id,
        task_id=base.task_id,
        requested_at=base.requested_at,
        requested_authority_scopes=base.requested_authority_scopes,
    )
    second_result = coordinator.resume(second)
    assert (
        second_result.outcome
        is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    )
    assert execution_port._requests[-1].governance_approval_evidence is None
