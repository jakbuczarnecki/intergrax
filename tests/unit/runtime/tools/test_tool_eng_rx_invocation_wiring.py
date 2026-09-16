# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-RX core wiring resolution gates."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import BaseModel
from unittest.mock import MagicMock

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.runtime_bound_catalog import RUNTIME_BOUND_TOOL_IDS
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.tools.core.handler import WiringContextToolHandler
from intergrax.tools.invocation_wiring import (
    ToolInvocationContext,
    ToolWiringOverlay,
    ToolWiringResolutionError,
    ensure_tool_wiring_overlay,
    merge_invocation_wiring,
    registration_wiring_for_handler,
)
from intergrax.tools.providers.invocation_requirements import REQUIRE_SHADOW_WORKSPACE
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace
from testing_support.builder import (
    FakeLLMAdapter,
    build_in_memory_session_manager,
    build_runtime_request_for_tests,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
    tools_agent_make_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


class _In(BaseModel):
    path: str = "a.txt"


class _Out(BaseModel):
    content: str = "ok"


class _EchoWorkspaceService:
    @staticmethod
    def run(ctx: ToolWiringContext, params: _In) -> _Out:
        ws = ctx.shadow_workspace
        if ws is None:
            return _Out(content="missing")
        return _Out(content=f"ws:{ws.task_id}:{params.path}")


class _WorkspaceHandler(ServiceToolHandler[_In, _Out]):
    _service = _EchoWorkspaceService.run


class _CaptureOverlayResolver:
    def __init__(self, overlay: ToolWiringOverlay) -> None:
        self._overlay = overlay
        self.calls = 0

    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolWiringContext,
    ) -> ToolWiringOverlay:
        self.calls += 1
        return self._overlay


def _state_with_registry(registry: ToolRegistry, seed: str = "rx-wiring") -> RuntimeState:
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=RegistryToolExecutor(registry),
    )
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tool_invoker=invoker,
    )
    ctx = RuntimeContext(
        config=config,
        session_manager=build_in_memory_session_manager(),
        prompt_registry=MagicMock(),
    )
    return RuntimeState(
        context=ctx,
        request=build_runtime_request_for_tests(
            seed=seed,
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            tenant_id="tenant-1",
            message="rx",
        ),
        run_id=canonical_run_id_for_tests(seed),
        tool_traces=[],
    )


def test_rx_t1_static_wired_tool_unchanged_without_invocation_context() -> None:
    registry = ToolRegistry()
    static_ctx = ToolWiringContext(extras={"marker": "static"})
    contract = tools_agent_make_contract("rx.static", _In, _Out)

    class _StaticHandler(ServiceToolHandler[_In, _Out]):
        @staticmethod
        def _svc(ctx: ToolWiringContext, params: _In) -> _Out:
            return _Out(content=str(ctx.extras.get("marker", "")))

        _service = _svc

    registry.register(contract, _StaticHandler(static_ctx))
    state = _state_with_registry(registry)
    with canonical_governed_execution_scope("rx-wiring"):
        result = state.context.config.tool_invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=state.run_id,
                step_id="s1",
                tool_id="rx.static",
                input=_In(),
            ),
        )
    assert result.success
    assert result.output is not None
    assert result.output.content == "static"


def test_rx_t2_invocation_overlay_reaches_handler(tmp_path: Path) -> None:
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="workspace.echo",
        name="workspace.echo",
        description="echo",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        invocation_wiring_requirements=REQUIRE_SHADOW_WORKSPACE,
    )
    registry.register(contract, _WorkspaceHandler(ToolWiringContext()))
    workspace = ShadowWorkspace.create(tmp_path, tenant_id="t1", task_id="task-rx")
    overlay = ToolWiringOverlay(shadow_workspace=workspace)
    resolver = _CaptureOverlayResolver(overlay)
    state = _state_with_registry(registry)
    with canonical_governed_execution_scope("rx-wiring"):
        result = state.context.config.tool_invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=state.run_id,
                step_id="s1",
                tool_id="workspace.echo",
                input=_In(),
                invocation_context=ToolInvocationContext(
                    run_id=state.run_id,
                    step_id="s1",
                    tool_id="workspace.echo",
                    wiring_resolver=resolver,
                ),
            ),
        )
    assert resolver.calls == 1
    assert result.success
    assert result.output is not None
    assert "task-rx" in result.output.content


def test_rx_t3_concurrent_invocations_isolated(tmp_path: Path) -> None:
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="workspace.echo",
        name="workspace.echo",
        description="echo",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        invocation_wiring_requirements=REQUIRE_SHADOW_WORKSPACE,
    )
    registry.register(contract, _WorkspaceHandler(ToolWiringContext()))
    ws_a = ShadowWorkspace.create(tmp_path / "a", tenant_id="t1", task_id="task-a")
    ws_b = ShadowWorkspace.create(tmp_path / "b", tenant_id="t1", task_id="task-b")

    def _call(task_id: str, workspace: ShadowWorkspace) -> str:
        seed = f"rx-wiring-{task_id}"
        state = _state_with_registry(registry, seed=seed)
        invoker = state.context.config.tool_invoker
        resolver = _CaptureOverlayResolver(ToolWiringOverlay(shadow_workspace=workspace))
        with canonical_governed_execution_scope(seed):
            result = invoker.invoke(
                state=state,
                agent_id="agent-1",
                request=ToolExecutionRequest(
                    run_id=state.run_id,
                    step_id=f"s-{task_id}",
                    tool_id="workspace.echo",
                    input=_In(),
                    invocation_context=ToolInvocationContext(
                        run_id=state.run_id,
                        step_id=f"s-{task_id}",
                        tool_id="workspace.echo",
                        wiring_resolver=resolver,
                    ),
                ),
            )
        assert result.success and result.output is not None
        return result.output.content

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(_call, "a", ws_a).result()
        b = pool.submit(_call, "b", ws_b).result()
    assert "task-a" in a
    assert "task-b" in b


def test_rx_t4_missing_required_wiring_fail_closed() -> None:
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="workspace.echo",
        name="workspace.echo",
        description="echo",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=False,
        invocation_wiring_requirements=REQUIRE_SHADOW_WORKSPACE,
    )
    registry.register(contract, _WorkspaceHandler(ToolWiringContext()))
    state = _state_with_registry(registry)
    with canonical_governed_execution_scope("rx-wiring"):
        result = state.context.config.tool_invoker.invoke(
            state=state,
            agent_id="agent-1",
            request=ToolExecutionRequest(
                run_id=state.run_id,
                step_id="s1",
                tool_id="workspace.echo",
                input=_In(),
                invocation_context=ToolInvocationContext(
                    run_id=state.run_id,
                    step_id="s1",
                    tool_id="workspace.echo",
                    wiring_resolver=_CaptureOverlayResolver(ToolWiringOverlay.empty()),
                ),
            ),
        )
    assert not result.success
    assert result.error is not None
    assert "shadow_workspace" in result.error.error_message


def test_rx_t6_custom_resolver_injection_without_core_patch() -> None:
    class _CustomResolver:
        def resolve(
            self,
            *,
            tool_id: str,
            invocation_context: ToolInvocationContext,
            registration_wiring: ToolWiringContext,
        ) -> ToolWiringOverlay:
            return ToolWiringOverlay(run_budget=RunBudget(max_tool_calls=42))

    merged = merge_invocation_wiring(
        ToolWiringContext(),
        _CustomResolver().resolve(
            tool_id="x",
            invocation_context=ToolInvocationContext(run_id="r", step_id="s", tool_id="x"),
            registration_wiring=ToolWiringContext(),
        ),
    )
    assert merged.run_budget is not None
    assert merged.run_budget.max_tool_calls == 42


def test_c1_t1_overlay_public_abi_has_no_any() -> None:
    source = (_REPO_ROOT / "intergrax/tools/invocation_wiring.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ToolWiringOverlay":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.annotation is not None:
                    ann = ast.unparse(item.annotation)
                    assert "Any" not in ann, f"ToolWiringOverlay field uses Any: {ann}"


def test_c1_t2_overlay_public_abi_has_no_object() -> None:
    source = (_REPO_ROOT / "intergrax/tools/invocation_wiring.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ToolWiringOverlay":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.annotation is not None:
                    ann = ast.unparse(item.annotation)
                    assert ann != "object" and "object |" not in ann


def test_c1_t3_registration_wiring_explicit_property() -> None:
    handler = _WorkspaceHandler(ToolWiringContext(extras={"x": "1"}))
    assert registration_wiring_for_handler(handler).extras["x"] == "1"
    assert isinstance(handler, WiringContextToolHandler)
    assert handler.registration_wiring.extras["x"] == "1"


def test_c1_t4_no_external_ctx_access_in_registration_helper() -> None:
    source = (_REPO_ROOT / "intergrax/tools/invocation_wiring.py").read_text(encoding="utf-8")
    assert "handler._ctx" not in source
    assert "_ctx" not in source.split("def registration_wiring_for_handler")[1].split("def ")[0]


def test_c1_t6_invalid_resolver_overlay_fail_closed() -> None:
    with pytest.raises(ToolWiringResolutionError) as exc_info:
        ensure_tool_wiring_overlay({"not": "overlay"})  # type: ignore[arg-type]
    assert exc_info.value.code == "wiring_overlay_invalid_type"


def test_c1_t11_runtime_bound_catalog_no_service_imports() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/nexus/tools/runtime_bound_catalog.py").read_text(
        encoding="utf-8",
    )
    assert ".service" not in source


def test_c1_t14_no_reflection_in_invocation_wiring_module() -> None:
    source = (_REPO_ROOT / "intergrax/tools/invocation_wiring.py").read_text(encoding="utf-8")
    forbidden = ("getattr(", "hasattr(", "setattr(", "inspect.signature")
    for token in forbidden:
        assert token not in source


def test_rx_static_gate_uaep_gateway_no_sandbox_execute() -> None:
    source = (_REPO_ROOT / "intergrax/runtime/nexus/tools/uaep_tool_gateway.py").read_text(
        encoding="utf-8",
    )
    assert "session.execute(" not in source
    assert "invoke_runtime_bound_tool" not in source
    assert "_invoke_sandbox" not in source


def test_rx_static_gate_runtime_bound_catalog_no_service_dispatch() -> None:
    path = _REPO_ROOT / "intergrax/runtime/nexus/tools/runtime_bound_catalog.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "service":
                raise AssertionError("runtime_bound_catalog must not call service()")
    source = path.read_text(encoding="utf-8")
    assert "def invoke_runtime_bound_tool" not in source


def test_runtime_bound_ids_catalog_only() -> None:
    assert "workspace.write_file" in RUNTIME_BOUND_TOOL_IDS
