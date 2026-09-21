# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5 — qualified CodeCraft execution through canonical ToolRuntime."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.codecraft.ephemeral_registry import EphemeralToolRegistryStore
from intergrax.runtime.codecraft.ownership import CodeCraftSessionOwnership
from intergrax.runtime.codecraft.session_manager import CodeCraftSessionManager
from intergrax.runtime.codecraft.wiring_bound_capability_execution import (
    WiringCodeCraftBoundCapabilityExecution,
)
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.catalog_tool_invocation_port import (
    CatalogToolInvocationBinding,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol
from intergrax.tools.core.contracts import ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TENANT,
    _TASK_ID,
)
from tests.unit.autonomous_work.uca6c_r5_tool_runtime_fixtures import (
    build_r5_catalog_tool_binding,
    build_sandbox_session,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy

pytestmark = pytest.mark.unit

_WIRING_PATH = Path("intergrax/runtime/codecraft/wiring_bound_capability_execution.py")


def test_wiring_static_gate_no_direct_code_exec_or_sandbox_execute() -> None:
    source = _WIRING_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "intergrax.tools.providers.sandbox.extended_service"
            assert "sandbox.session" not in (node.module or "")
    assert "write_file" not in source
    assert "code_exec(" not in source
    assert "ToolRegistry.register" not in source
    assert "production_mode=False" not in source
    assert "PolicyAction.ALLOW" not in source


def test_wiring_static_gate_no_registry_mutation_tokens() -> None:
    source = _WIRING_PATH.read_text(encoding="utf-8")
    assert ".register(" not in source
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        raise AssertionError("backend must not run")


def test_scope_deny_blocks_before_backend(tmp_path: Path) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.tools.registry.wiring import ToolWiringContext
    from testing_support.codecraft_execution_environment import (
        codecraft_sandbox_execution_profile,
    )

    sandbox = build_sandbox_session(
        tmp_path,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
    )
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    craft_id = "craft-r5-scope-deny"
    ownership = CodeCraftSessionOwnership(tenant_id=_TENANT, task_id=str(_TASK_ID))
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="autonomous",
        craft_id=craft_id,
    )
    sessions.save_owned(
        session.model_copy(update={"code": "print('x')"}),
        ownership,
    )
    registry.for_craft(craft_id).register("ephemeral.r5.helper")
    ctx = ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous", require_tests=False
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )
    run_id = mint_run_id()
    binding, tool_registry, _ = build_r5_catalog_tool_binding(
        ctx,
        run_seed=str(run_id),
        allowed_tool_ids=set(),
    )
    executor = _CountingExecutor()
    deny_invoker = RuntimeToolInvoker(
        registry=tool_registry,
        executor=executor,
        sandbox_availability=sandbox_availability_provider(ctx),
        scope_policy=StaticToolScopePolicy(allowed_tools=set()),
    )
    binding = CatalogToolInvocationBinding(
        tool_invoker=deny_invoker,
        state_supplier=binding.state_supplier,
        caller_agent_id=binding.caller_agent_id,
    )
    port = WiringCodeCraftBoundCapabilityExecution(ctx, tool_invocation=binding)
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                run_id=None,
                execution_id=execution_id,
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert result.outcome in {
        CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
        CodeCraftBoundCapabilityExecutionOutcome.FAILED,
    }
    assert executor.calls == 0


def test_runtime_success_via_code_exec(tmp_path: Path) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.tools.registry.wiring import ToolWiringContext
    from testing_support.codecraft_execution_environment import (
        codecraft_sandbox_execution_profile,
    )

    sandbox = build_sandbox_session(
        tmp_path,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
    )
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    craft_id = "craft-r5-success"
    ownership = CodeCraftSessionOwnership(tenant_id=_TENANT, task_id=str(_TASK_ID))
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="autonomous",
        craft_id=craft_id,
    )
    sessions.save_owned(
        session.model_copy(update={"code": "print('r5-ok')"}),
        ownership,
    )
    registry.for_craft(craft_id).register("ephemeral.r5.success")
    ctx = ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous", require_tests=False
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )
    run_id = mint_run_id()
    binding, tool_registry, _invoker = build_r5_catalog_tool_binding(
        ctx,
        run_seed=str(run_id),
    )
    assert tool_registry.has(CODE_EXEC_TOOL_ID)
    port = WiringCodeCraftBoundCapabilityExecution(ctx, tool_invocation=binding)
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                run_id=None,
                execution_id=execution_id,
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED
    contract = tool_registry.get(CODE_EXEC_TOOL_ID).contract
    assert contract.risk_level is ToolRiskLevel.HIGH
    assert contract.side_effects is True
    assert tool_registry.has(CODE_EXEC_TOOL_ID)
    state = binding.runtime_state_for_invocation()
    assert any(e.step == "tool_invocation_start" for e in state.trace_events)


def test_missing_tool_invocation_binding_unavailable(tmp_path: Path) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.tools.registry.wiring import ToolWiringContext
    from testing_support.codecraft_execution_environment import (
        codecraft_sandbox_execution_profile,
    )

    sandbox = build_sandbox_session(
        tmp_path,
        tenant_id=_TENANT,
        task_id=str(_TASK_ID),
    )
    sessions = CodeCraftSessionManager()
    registry = EphemeralToolRegistryStore()
    craft_id = "craft-r5-unconfigured"
    ownership = CodeCraftSessionOwnership(tenant_id=_TENANT, task_id=str(_TASK_ID))
    session = sessions.open(
        goal="exec",
        ownership=ownership,
        mode="autonomous",
        craft_id=craft_id,
    )
    sessions.save_owned(
        session.model_copy(update={"code": "print(1)"}),
        ownership,
    )
    registry.for_craft(craft_id).register("ephemeral.r5.unconfigured")
    ctx = ToolWiringContext(
        sandbox_session=sandbox,
        extras={
            "codecraft_session_manager": sessions,
            "codecraft_ephemeral_registry": registry,
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous", require_tests=False
            ),
            "effective_environment_profile": codecraft_sandbox_execution_profile(),
        },
    )
    port = WiringCodeCraftBoundCapabilityExecution(ctx)
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        result = port.execute(
            CodeCraftBoundCapabilityExecutionRequest(
                craft_id=craft_id,
                tenant_id=_TENANT,
                task_id=_TASK_ID,
                run_id=None,
                execution_id=execution_id,
            ),
        )
    finally:
        reset_active_execution_identity(token)
    assert result.outcome is CodeCraftBoundCapabilityExecutionOutcome.UNAVAILABLE
    assert result.reason_detail == "canonical_tool_invocation_unconfigured"


@dataclass
class _RecordingInvoker:
    registry: object
    calls: int = 0

    def invoke(
        self,
        *,
        state: object,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> ToolExecutionResult[BaseModel]:
        self.calls += 1
        return ToolExecutionResult.fail(
            RuntimeErrorCode.TOOL_ERROR.value,
            "injected",
        )


def test_custom_tool_invoker_injection() -> None:
    from intergrax.tools.registry.wiring import ToolWiringContext

    recording = _RecordingInvoker(registry=object())
    binding = CatalogToolInvocationBinding(
        tool_invoker=recording,
        state_supplier=lambda: object(),
        caller_agent_id="worker-test",
    )
    ctx = ToolWiringContext()
    port = WiringCodeCraftBoundCapabilityExecution(ctx, tool_invocation=binding)
    assert isinstance(port._tool_invocation.tool_invoker, ToolInvokerProtocol)
