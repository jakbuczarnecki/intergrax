# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5 — qualified CodeCraft execution through canonical ToolRuntime."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.tool_wiring import ApplicationToolWiring
from intergrax.applications._shared.uca6c_codecraft_qualified_execution_composition import (
    build_production_codecraft_qualified_capability_execution_handler,
)
from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
    ExecutionBoundCatalogToolInvoker,
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
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.sandbox.isolation_gate import sandbox_availability_provider
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from intergrax.tools.registry import ToolProfile
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TENANT,
    _TASK_ID,
)
from tests.unit.autonomous_work.uca6c_r5_tool_runtime_fixtures import (
    build_r5_production_catalog_tool_invoker,
    build_sandbox_session,
    sandbox_env_profile,
)
from intergrax.tools.registry.runtime import ToolRegistry

pytestmark = pytest.mark.unit

_WIRING_PATH = Path("intergrax/runtime/codecraft/wiring_bound_capability_execution.py")
_CONTRACT_PATH = Path(
    "intergrax/contracts/execution_bound_catalog_tool_invocation.py",
)


def test_wiring_static_gate_no_direct_code_exec_or_sandbox_execute() -> None:
    source = _WIRING_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "intergrax.tools.providers.sandbox.extended_service"
            assert "sandbox.session" not in (node.module or "")
            assert "runtime_state" not in (node.module or "")
            assert "runtime.nexus.engine.runtime_state" not in (node.module or "")
    assert "write_file" not in source
    assert "code_exec(" not in source
    assert "ToolRegistry.register" not in source
    assert "production_mode=False" not in source
    assert "PolicyAction.ALLOW" not in source
    assert "RuntimeState" not in source
    assert "runtime_state_for_invocation" not in source
    assert "Callable[[], object]" not in source


def test_wiring_static_gate_no_registry_mutation_tokens() -> None:
    source = _WIRING_PATH.read_text(encoding="utf-8")
    assert ".register(" not in source
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source


def test_public_contract_location_and_typing() -> None:
    assert _CONTRACT_PATH.is_file()
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    assert "class ExecutionBoundCatalogToolInvoker" in source
    assert "ExecutionBoundCatalogToolInvokeRequest" in source
    assert "-> object" not in source
    assert "Callable[[], object]" not in source


class _CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request) -> BaseModel:
        self.calls += 1
        raise AssertionError("backend must not run")


def test_scope_deny_blocks_before_backend(tmp_path: Path) -> None:
    from intergrax.codecraft.profile import CodeCraftProfile
    from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
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
    catalog_invoker, tool_registry, _ = build_r5_production_catalog_tool_invoker(
        ctx,
        tenant_id=_TENANT,
    )
    executor = _CountingExecutor()
    deny_invoker = RuntimeToolInvoker(
        registry=tool_registry,
        executor=executor,
        sandbox_availability=sandbox_availability_provider(ctx),
        scope_policy=StaticToolScopePolicy(allowed_tools=set()),
    )
    assert isinstance(catalog_invoker, NexusExecutionBoundCatalogToolInvoker)
    catalog_invoker = NexusExecutionBoundCatalogToolInvoker(
        tool_invoker=deny_invoker,
        policy_bundle=catalog_invoker.policy_bundle,
        caller_agent_id=catalog_invoker.caller_agent_id,
    )
    port = WiringCodeCraftBoundCapabilityExecution(
        ctx,
        catalog_tool_invoker=catalog_invoker,
    )
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


def test_runtime_success_via_production_composition(tmp_path: Path) -> None:
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
    tool_wiring = ApplicationToolWiring(
        profile=ToolProfile(enabled_bundles=frozenset({"sandbox"})),
        wiring_context=ctx,
        registry=ToolRegistry(),
    )
    handler = build_production_codecraft_qualified_capability_execution_handler(
        tool_wiring,
        sandbox_env_profile(),
        caller_agent_id="worker-r5-e2e",
        tenant_id=_TENANT,
    )
    port = handler._execution_port
    assert isinstance(port, WiringCodeCraftBoundCapabilityExecution)
    assert port._catalog_tool_invoker is not None
    catalog_invoker = port._catalog_tool_invoker
    assert isinstance(catalog_invoker, NexusExecutionBoundCatalogToolInvoker)
    tool_registry = catalog_invoker.tool_invoker.registry
    assert tool_registry.has(CODE_EXEC_TOOL_ID)
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
    assert "tool_invocation_start" in catalog_invoker.last_invocation_trace_steps


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
class _RecordingCatalogInvoker:
    caller_agent_id: str = "worker-test"
    calls: int = 0

    def bind_execution_identity(
        self,
        *,
        tenant_id: str,
        run_id: str,
        task_id: str,
        agent_id: str,
    ) -> None:
        _ = tenant_id, run_id, task_id, agent_id

    def invoke(
        self,
        request: ExecutionBoundCatalogToolInvokeRequest,
    ) -> ToolExecutionResult[BaseModel]:
        self.calls += 1
        _ = request
        return ToolExecutionResult.fail(
            RuntimeErrorCode.TOOL_ERROR.value,
            "injected",
        )


def test_custom_catalog_tool_invoker_injection() -> None:
    from intergrax.tools.registry.wiring import ToolWiringContext

    recording: ExecutionBoundCatalogToolInvoker = _RecordingCatalogInvoker()
    ctx = ToolWiringContext()
    port = WiringCodeCraftBoundCapabilityExecution(
        ctx,
        catalog_tool_invoker=recording,
    )
    assert port._catalog_tool_invoker is recording
