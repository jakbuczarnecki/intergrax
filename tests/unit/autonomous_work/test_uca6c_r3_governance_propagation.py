# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R3 — trusted governance propagation and authority zero-mint."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityResumeOutcome,
)
from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityExecutionTarget,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from tests.unit.autonomous_work.uca6c_bound_execution_fixtures import (
    recording_codecraft_execution_handler,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
    DenyingRuntimeExecutionPolicyAdmission,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _NOW,
    _QUAL_REQUEST,
    _TENANT,
    _WORKER_ID,
    _acquisition,
    _authority_admission,
    _execution_governance,
    _qualification,
    _resume_request,
    _subject,
    _wiring,
)
from tests.unit.autonomous_work.uca6c_worker_authority_fixtures import (
    _READ,
    build_worker_execution_admission_for_uca6c,
    trusted_governance_from_admission,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingRequest,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.root_execution_launch import RootExecutionLaunchRequest

pytestmark = pytest.mark.unit

_DISPATCH_MODULE = Path(
    "intergrax/runtime/execution/qualified_capability_execution_dispatch_service.py",
)
_TASK_ID = TaskId("task_" + "e" * 32)
_DENY_DECISION = EffectiveAuthorityDecision(
    decision=PolicyDecision(action=PolicyAction.DENY, reason="upstream_fixture_deny"),
)


def test_dispatch_module_architecture_gates_no_identity_or_allow_mint() -> None:
    source = _DISPATCH_MODULE.read_text(encoding="utf-8")
    assert "AdmittedRootGovernanceIdentity(" not in source
    assert "PolicyAction.ALLOW" not in source
    assert "_default_governance_identity_resolver" not in source
    assert "workspace.read" not in source
    assert "qualified_capability_resume" not in source


def test_missing_authority_admission_fails_closed_before_execution() -> None:
    ctx = _wiring()
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=WorkerQualifiedCapabilityExecutionEngineAdapter(
            dispatch=build_qualified_capability_execution_dispatch_service(
                handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry(()),
                runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
            )[0],
        ),
        authority_admission=None,
    )
    result = coordinator.resume(_resume_request(_qualification()))
    assert (
        result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_UNAVAILABLE
    )
    assert result.execution_result is None


def test_tenant_workspace_principal_preserved_in_launch() -> None:
    workspace = "workspace-distinct-from-tenant"
    principal = "human-principal-abc"
    admission = build_worker_execution_admission_for_uca6c(
        worker_instance_id=_WORKER_ID,
        tenant_id=_TENANT,
        workspace_id=workspace,
        principal_id=principal,
    )
    admitted, decision, scopes = trusted_governance_from_admission(
        admission=admission,
        worker_instance_id=_WORKER_ID,
        requested_scopes=(_READ,),
    )
    assert admitted.tenant_id == _TENANT
    assert admitted.workspace_id == workspace
    assert admitted.principal_id == principal
    assert workspace != _TENANT
    assert not principal.startswith("worker:")

    captured: list[RootExecutionLaunchRequest] = []

    class _CapturingLauncher:
        async def launch(self, request: RootExecutionLaunchRequest):
            captured.append(request)
            from intergrax.contracts.root_execution_launch import (
                RootExecutionLaunchDisposition,
                RootExecutionLaunchResult,
            )

            return RootExecutionLaunchResult(
                disposition=RootExecutionLaunchDisposition.DENIED,
            )

    dispatch, delegate, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry(()),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    dispatch._launcher = _CapturingLauncher()  # type: ignore[attr-defined]
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id="recovery:r3:launch-capture",
        qualification_request_id=_QUAL_REQUEST,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    ctx = _wiring()
    target = (
        CodeCraftQualifiedCapabilityBindingProvider(ctx)
        .bind(
            QualifiedCapabilityBindingRequest(
                binding_operation_id=binding_id,
                resume_operation_id=resume_id,
                qualified_subject=_subject(),
                qualification_result=_qualification(),
                worker_need_id="worker-need:r3",
                worker_instance_id=str(_WORKER_ID),
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                requested_at=_NOW,
            ),
        )
        .execution_target
    )
    assert target is not None
    execution_request_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    adapter.execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_request_id,
            execution_target=target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need:r3",
            tenant_id=_TENANT,
            task_id=_TASK_ID,
            qualification_request_id=_QUAL_REQUEST,
            acquisition_request_id=_acquisition().request_id,
            qualified_subject_reference=_subject().qualified_subject_reference,
            requested_at=_NOW,
            admitted_governance_identity=admitted,
            effective_authority_decision=decision,
            collaborative_authority_scopes=scopes,
        ),
    )
    assert len(captured) == 1
    launch = captured[0]
    assert launch.admitted_governance_identity.tenant_id == _TENANT
    assert launch.admitted_governance_identity.workspace_id == workspace
    assert launch.admitted_governance_identity.principal_id == principal
    assert launch.collaborative_authority_scopes == scopes
    assert launch.effective_authority_decision is decision
    assert delegate.execute_calls == 0


def test_upstream_deny_does_not_invoke_runtime_delegate() -> None:
    ctx = _wiring()
    handler, _ = recording_codecraft_execution_handler(side_effect_recorder=[])
    dispatch, delegate, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    admitted, _, scopes = _execution_governance()
    resume_id = "worker-capability-resume:r3:deny"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    target = (
        CodeCraftQualifiedCapabilityBindingProvider(ctx)
        .bind(
            QualifiedCapabilityBindingRequest(
                binding_operation_id=binding_id,
                resume_operation_id=resume_id,
                qualified_subject=_subject(),
                qualification_result=_qualification(),
                worker_need_id="worker-need:r3",
                worker_instance_id=str(_WORKER_ID),
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                requested_at=_NOW,
            ),
        )
        .execution_target
    )
    assert target is not None
    execution_request_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    result = adapter.execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_request_id,
            execution_target=target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need:r3",
            tenant_id=_TENANT,
            task_id=_TASK_ID,
            qualification_request_id=_QUAL_REQUEST,
            acquisition_request_id=_acquisition().request_id,
            qualified_subject_reference=_subject().qualified_subject_reference,
            requested_at=_NOW,
            admitted_governance_identity=admitted,
            effective_authority_decision=_DENY_DECISION,
            collaborative_authority_scopes=scopes,
        ),
    )
    assert result.disposition is WorkerQualifiedCapabilityExecutionDisposition.REJECTED
    assert delegate.execute_calls == 0


def test_forged_tenant_on_dispatch_request_rejected_at_construction() -> None:
    admitted, decision, scopes = _execution_governance()
    target = QualifiedCapabilityExecutionTarget(
        execution_target_reference="execution-target:fixture",
        binding_provider_id="fixture.provider",
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    with pytest.raises(ValueError, match="tenant_id must match"):
        QualifiedCapabilityExecutionDispatchRequest(
            execution_request_id="qualified-capability-execution:a:b",
            execution_target=target,
            tenant_id="forged-tenant",
            task_id=_TASK_ID,
            worker_instance_id=_WORKER_ID,
            worker_need_id="need",
            resume_operation_id="worker-capability-resume:x:y",
            binding_operation_id="binding-op",
            qualification_request_id=_QUAL_REQUEST,
            acquisition_request_id=_acquisition().request_id,
            qualified_subject_reference=_subject().qualified_subject_reference,
            requested_at=_NOW,
            admitted_governance_identity=admitted,
            effective_authority_decision=decision,
            collaborative_authority_scopes=scopes,
        )


def test_upstream_allow_runtime_deny_no_delegate() -> None:
    ctx = _wiring()
    side_effects: list[str] = []
    handler, _ = recording_codecraft_execution_handler(side_effect_recorder=side_effects)
    dispatch, delegate, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
    )
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch),
        authority_admission=_authority_admission(),
    )
    result = coordinator.resume(_resume_request(_qualification()))
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_REJECTED
    assert delegate.execute_calls == 0
    assert side_effects == []


def test_upstream_allow_runtime_allow_reaches_execution_runtime() -> None:
    ctx = _wiring()
    side_effects: list[str] = []
    handler, execution_port = recording_codecraft_execution_handler(
        side_effect_recorder=side_effects,
    )
    dispatch, delegate, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch),
        authority_admission=_authority_admission(),
    )
    result = coordinator.resume(_resume_request(_qualification()))
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    assert delegate.execute_calls == 1
    assert execution_port.runtime_execution_calls == 1
    assert len(side_effects) == 1
