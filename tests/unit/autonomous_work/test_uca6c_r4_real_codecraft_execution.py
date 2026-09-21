# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R4 — real bound CodeCraft capability execution under ExecutionRuntime."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
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
from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
    CodeCraftBoundCapabilityExecutionResult,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.qualified_capability_binding_provider import (
    CodeCraftQualifiedCapabilityBindingProvider,
)
from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
    CodeCraftQualifiedCapabilityExecutionHandler,
)
from intergrax.runtime.codecraft.qualified_capability_execution_wiring import (
    build_codecraft_qualified_capability_execution_handler,
)
from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
    WiringCodeCraftBoundCapabilityExecution,
)
from intergrax.runtime.execution.qualified_capability_execution_composition import (
    build_qualified_capability_execution_dispatch_service,
)
from intergrax.runtime.execution.qualified_capability_execution_handlers import (
    QualifiedCapabilityExecutionBindingHandler,
    QualifiedCapabilityExecutionBindingHandlerRegistry,
)
from intergrax.runtime.execution.worker_qualified_capability_execution_adapter import (
    WorkerQualifiedCapabilityExecutionEngineAdapter,
)
from intergrax.runtime.governance.runtime_execution_policy_admission import (
    AllowingRuntimeExecutionPolicyAdmission,
)
from tests.unit.autonomous_work.test_uca6c_r_production_resume import (
    _CRAFT_ID,
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
from tests.unit.autonomous_work.uca6c_bound_execution_fixtures import (
    RecordingCodeCraftBoundCapabilityExecution,
    recording_codecraft_execution_handler,
)

pytestmark = pytest.mark.unit

_HANDLER_PATH = Path(
    "intergrax/runtime/codecraft/qualified_capability_execution_handler.py"
)
_TASK_ID = TaskId("task_" + "e" * 32)


def test_handler_static_gate_no_aw_or_sandbox_bypass_imports() -> None:
    tree = ast.parse(_HANDLER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.autonomous_work")
            assert node.module != "intergrax.tools.providers.sandbox.extended_service"


def test_handler_static_gate_no_worker_ephemeral_service() -> None:
    source = _HANDLER_PATH.read_text(encoding="utf-8")
    assert "WorkerEphemeralCapabilityExecutionService" not in source
    assert "CodeCraftEphemeralCapabilityExecutionAdapter" not in source


def test_e2e_resume_invokes_runtime_execution_port_once() -> None:
    ctx = _wiring()
    coordinator, dispatch, _, execution_port = _production_stack(ctx)
    coordinator.resume(_resume_request(_qualification()))
    coordinator.resume(_resume_request(_qualification()))
    assert dispatch.dispatch_side_effects == 1
    assert execution_port.runtime_execution_calls == 1
    assert len(execution_port._requests) == 1
    assert execution_port._requests[0].craft_id == _CRAFT_ID


def test_distinct_execution_request_ids_invoke_twice() -> None:
    ctx = _wiring()
    handler, execution_port = recording_codecraft_execution_handler()
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    target = _bound_target(ctx)
    admitted, decision, scopes = _execution_governance()
    for suffix in ("a", "b"):
        resume_id = f"worker-capability-resume:r4:distinct:{suffix}"
        binding_id = derive_qualified_capability_binding_operation_id(
            resume_operation_id=resume_id,
            qualified_subject_reference=_subject().qualified_subject_reference,
        )
        execution_id = derive_qualified_capability_execution_request_id(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
        )
        adapter.execute(
            WorkerQualifiedCapabilityExecutionRequest(
                resume_operation_id=resume_id,
                binding_operation_id=binding_id,
                execution_request_id=execution_id,
                execution_target=target,
                worker_instance_id=_WORKER_ID,
                worker_need_id="worker-need:r4",
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
    assert execution_port.runtime_execution_calls == 2


def test_stale_artifact_at_execution_unavailable() -> None:
    ctx = _wiring()
    target = _bound_target(ctx)
    assert target is not None
    registry = ctx.extras["codecraft_ephemeral_registry"]
    assert isinstance(registry, EphemeralToolRegistryStore)
    registry.dispose(_CRAFT_ID)
    execution_port = WiringCodeCraftBoundCapabilityExecution(ctx)
    handler = CodeCraftQualifiedCapabilityExecutionHandler(
        execution_port=execution_port
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    admitted, decision, scopes = _execution_governance()
    resume_id = "worker-capability-resume:r4:stale"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    execution_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    result = adapter.execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_id,
            execution_target=target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need:r4",
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
    assert (
        result.disposition is WorkerQualifiedCapabilityExecutionDisposition.UNAVAILABLE
    )
    assert execution_port.runtime_execution_calls == 0


def test_port_failure_maps_failed_without_runtime_invocation_count() -> None:
    ctx = _wiring()
    handler, execution_port = recording_codecraft_execution_handler(
        forced_outcome=CodeCraftBoundCapabilityExecutionOutcome.FAILED,
        forced_reason_detail="typed_runtime_failure",
    )
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((handler,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    adapter = WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch)
    result = _execute_once(adapter, ctx)
    assert result.disposition is WorkerQualifiedCapabilityExecutionDisposition.FAILED
    assert execution_port.runtime_execution_calls == 1


def test_custom_binding_handler_without_aw_changes() -> None:
    class _CustomHandler:
        def __init__(self) -> None:
            self.runtime_execution_calls = 0

        @property
        def binding_provider_id(self) -> str:
            return "custom.provider.v1"

        def dispatch_once(self, request, *, run_id, attempt_id, execution_id):
            self.runtime_execution_calls += 1
            from intergrax.contracts.execution.qualified_capability_execution_intake import (
                QualifiedCapabilityExecutionDelegateResult,
            )

            return QualifiedCapabilityExecutionDelegateResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            )

    custom: QualifiedCapabilityExecutionBindingHandler = _CustomHandler()
    dispatch, _, _ = build_qualified_capability_execution_dispatch_service(
        handler_registry=QualifiedCapabilityExecutionBindingHandlerRegistry((custom,)),
        runtime_policy_admission=AllowingRuntimeExecutionPolicyAdmission(),
    )
    from intergrax.contracts.capability_qualification.qualified_capability_binding import (
        QualifiedCapabilityExecutionTarget,
    )

    admitted, decision, scopes = _execution_governance()
    resume_id = "worker-capability-resume:custom:1"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    execution_request_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    target = QualifiedCapabilityExecutionTarget(
        execution_target_reference="custom:target:1",
        binding_provider_id="custom.provider.v1",
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    WorkerQualifiedCapabilityExecutionEngineAdapter(dispatch=dispatch).execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_request_id,
            execution_target=target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need:custom",
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
    assert custom.runtime_execution_calls == 1


def test_production_wiring_port_requires_active_execution_id(tmp_path: Path) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        reset_active_execution_identity,
    )
    from intergrax.runtime.codecraft.ownership import CodeCraftSessionOwnership
    from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
    from intergrax.runtime.sandbox.session import SandboxSession
    from intergrax.tools.registry.wiring import ToolWiringContext
    from testing_support.codecraft_execution_environment import (
        codecraft_sandbox_execution_profile,
    )

    sandbox = SandboxSession.create(
        tmp_path,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
        allowed_operations=frozenset(
            {"echo", "write_file", "read_file", "list_files", "run_python"},
        ),
    )
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    ownership = CodeCraftSessionOwnership(tenant_id=_TENANT, task_id=str(_TASK_ID))
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="autonomous",
        craft_id="craft-r4-wiring",
    )
    sessions.save_owned(
        session.model_copy(update={"code": "print('ok')"}),
        ownership,
    )
    registry.for_craft("craft-r4-wiring").register("ephemeral.craft-r4-wiring.helper")
    ctx = ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous",
                require_tests=False,
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )
    from tests.unit.autonomous_work.uca6c_r5_tool_runtime_fixtures import (
        build_r5_catalog_tool_binding,
    )

    run_id = mint_run_id()
    tool_binding, _, _ = build_r5_catalog_tool_binding(
        ctx,
        run_seed=str(run_id),
    )
    port = WiringCodeCraftBoundCapabilityExecution(ctx, tool_invocation=tool_binding)
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id="craft-r4-wiring",
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                run_id=None,
                execution_id=execution_id,
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED
    assert port.runtime_execution_calls == 1


def _bound_target(ctx):
    resume_id = derive_worker_capability_resume_operation_id(
        recovery_decision_id=_RECOVERY_DECISION,
        qualification_request_id=_QUAL_REQUEST,
    )
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    return (
        CodeCraftQualifiedCapabilityBindingProvider(ctx)
        .bind(
            QualifiedCapabilityBindingRequest(
                binding_operation_id=binding_id,
                resume_operation_id=resume_id,
                qualified_subject=_subject(),
                qualification_result=_qualification(),
                worker_need_id="worker-need:r4",
                worker_instance_id=str(_WORKER_ID),
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                requested_at=_NOW,
            ),
        )
        .execution_target
    )


def _execute_once(adapter, ctx):
    admitted, decision, scopes = _execution_governance()
    resume_id = "worker-capability-resume:r4:once"
    binding_id = derive_qualified_capability_binding_operation_id(
        resume_operation_id=resume_id,
        qualified_subject_reference=_subject().qualified_subject_reference,
    )
    execution_id = derive_qualified_capability_execution_request_id(
        resume_operation_id=resume_id,
        binding_operation_id=binding_id,
    )
    target = _bound_target(ctx)
    assert target is not None
    return adapter.execute(
        WorkerQualifiedCapabilityExecutionRequest(
            resume_operation_id=resume_id,
            binding_operation_id=binding_id,
            execution_request_id=execution_id,
            execution_target=target,
            worker_instance_id=_WORKER_ID,
            worker_need_id="worker-need:r4",
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
