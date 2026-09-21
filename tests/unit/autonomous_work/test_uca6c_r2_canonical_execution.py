# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R2 — canonical ExecutionRuntime adoption and zero-bypass gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.autonomous_work.worker_qualified_capability_resume_coordinator import (
    WorkerQualifiedCapabilityResumeCoordinator,
)
from intergrax.capability_qualification.qualified_capability_binding_service import (
    QualifiedCapabilityBindingService,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityExecutionDisposition,
    WorkerQualifiedCapabilityExecutionRequest,
    WorkerQualifiedCapabilityResumeOutcome,
    derive_qualified_capability_execution_request_id,
    derive_worker_capability_resume_operation_id,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingRequest,
    derive_qualified_capability_binding_operation_id,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from tests.unit.autonomous_work.uca6c_bound_execution_fixtures import (
    recording_codecraft_execution_handler,
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
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    DenyingRuntimeExecutionPolicyAdmission,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _NOW,
    _QUAL_REQUEST,
    _RECOVERY_DECISION,
    _TENANT,
    _WORKER_ID,
    _acquisition,
    _execution_governance,
    _production_stack,
    _qualification,
    _resume_request,
    _subject,
    _wiring,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_" + "e" * 32)

_CODECRAFT_HANDLER = Path(
    "intergrax/runtime/codecraft/qualified_capability_execution_handler.py",
)


def test_codecraft_qualified_handler_does_not_import_mint_execution_id() -> None:
    tree = ast.parse(_CODECRAFT_HANDLER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == (
            "intergrax.contracts.execution_identity"
        ):
            for alias in node.names:
                assert alias.name != "mint_execution_id"


def test_execution_request_id_differs_from_execution_id() -> None:
    ctx = _wiring()
    coordinator, _, _, _ = _production_stack(ctx)
    result = coordinator.resume(_resume_request(_qualification()))
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_DISPATCHED
    execution = result.execution_result
    assert execution is not None
    assert execution.execution_request_id is not None
    assert execution.execution_id is not None
    assert execution.execution_request_id != str(execution.execution_id)


def test_ee_admission_denied_rejected_no_delegate_side_effect() -> None:
    ctx = _wiring()
    side_effects: list[str] = []
    handler, _ = recording_codecraft_execution_handler(side_effect_recorder=side_effects)
    dispatch, delegate, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=DenyingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=_QUAL_REQUEST,
    )
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
                worker_need_id="worker-need:uca6cr:1",
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
    admitted, decision, scopes = _execution_governance()
    result = adapter.execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_request_id,
            execution_target=target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need:uca6cr:1",
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
    assert result.disposition is WorkerQualifiedCapabilityExecutionDisposition.REJECTED
    assert delegate.execute_calls == 0
    assert side_effects == []


def test_canonical_runtime_delegate_invoked_once_per_execution_request() -> None:
    ctx = _wiring()
    coordinator, dispatch, _, _ = _production_stack(ctx)
    request = _resume_request(_qualification())
    coordinator.resume(request)
    coordinator.resume(request)
    assert dispatch.dispatch_side_effects == 1


@dataclass
class _MismatchDispatch:
    def dispatch(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            execution_request_id="wrong-execution-request-id",
        )


def test_execution_request_id_mismatch_still_fail_closed() -> None:
    ctx = _wiring()
    from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
        _authority_admission,
    )

    coordinator = WorkerQualifiedCapabilityResumeCoordinator(
        binding=QualifiedCapabilityBindingService(
            (CodeCraftQualifiedCapabilityBindingProvider(ctx),),
        ),
        execution=WorkerQualifiedCapabilityExecutionEngineAdapter(
            dispatch=_MismatchDispatch(),
        ),
        authority_admission=_authority_admission(),
    )
    result = coordinator.resume(_resume_request(_qualification()))
    assert result.outcome is WorkerQualifiedCapabilityResumeOutcome.EXECUTION_FAILED
