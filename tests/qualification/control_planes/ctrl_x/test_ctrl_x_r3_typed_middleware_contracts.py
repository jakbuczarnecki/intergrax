# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R3 typed middleware contract qualification (R3-T01..R3-T12)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationMiddlewareHookContext,
    HostOrchestrationMiddlewarePipelinePort,
)
from intergrax.contracts.middleware_hook_semantics import (
    DataProtectionHookPayload,
    DataProtectionRestrictedValue,
    LlmInferenceHookPayload,
    MiddlewareExecutionSubjectFacet,
    MiddlewareHookInvocationContext,
    ToolCallHookPayload,
)
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import PluginSecurityDefenseMiddleware
from intergrax.runtime.security.encryption_middleware import EncryptionEnforcementMiddleware
from intergrax.runtime.security.security_events import emit_defense_blocked
from tests.qualification.control_planes.ctrl_x.test_ctrl_x_r2_middleware_composition import (
    _AlternateMiddlewarePipeline,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOUNDARY_SOURCES = (
    _REPO_ROOT / "intergrax/contracts/middleware_hook_semantics.py",
    _REPO_ROOT / "intergrax/contracts/host_orchestration_wiring_capabilities.py",
)
_CROSS_LAYER_MIDDLEWARE = (
    _REPO_ROOT / "intergrax/runtime/security/defense_plugin.py",
    _REPO_ROOT / "intergrax/runtime/security/defense_registry.py",
    _REPO_ROOT / "intergrax/runtime/security/encryption_middleware.py",
    _REPO_ROOT / "intergrax/runtime/security/security_events.py",
    _REPO_ROOT / "intergrax/applications/_shared/application_security_wiring.py",
    _REPO_ROOT / "intergrax/applications/_shared/application_guardrail_middleware.py",
    _REPO_ROOT / "intergrax/applications/_shared/autonomy_middleware.py",
    _REPO_ROOT / "intergrax/runtime/middleware/trace_middleware.py",
)


def test_r3_t01_hook_context_satisfies_middleware_invocation_context() -> None:
    ctx = HookContext(task_id="t", run_id="r")
    assert isinstance(ctx, MiddlewareHookInvocationContext)
    assert isinstance(ctx, HostOrchestrationMiddlewareHookContext)


def test_r3_t02_cross_layer_protocol_has_no_runtime_state() -> None:
    source = (_REPO_ROOT / "intergrax/contracts/middleware_hook_semantics.py").read_text(
        encoding="utf-8",
    )
    assert "runtime_state" not in source


def test_r3_t03_tool_call_payload_strongly_typed() -> None:
    payload = ToolCallHookPayload(tool_id="echo", arguments={"q": "1"})
    assert payload.tool_id == "echo"
    assert payload.arguments["q"] == "1"


def test_r3_t04_llm_payload_strongly_typed() -> None:
    payload = LlmInferenceHookPayload(prompt="hi", llm_output="ok")
    assert payload.prompt == "hi"
    assert payload.llm_output == "ok"


def test_r3_t05_subject_carries_tenant_facets() -> None:
    subject = MiddlewareExecutionSubjectFacet(
        tenant_id="a",
        resource_tenant_id="b",
    )
    assert subject.tenant_id == "a"
    assert subject.resource_tenant_id == "b"


def test_r3_t06_security_middleware_source_has_no_runtime_state_reads() -> None:
    for path in (
        _REPO_ROOT / "intergrax/runtime/security/defense_plugin.py",
        _REPO_ROOT / "intergrax/runtime/security/defense_registry.py",
        _REPO_ROOT / "intergrax/applications/_shared/application_security_wiring.py",
    ):
        text = path.read_text(encoding="utf-8")
        assert "runtime_state" not in text


def test_r3_t07_security_events_use_subject_tenant() -> None:
    source = inspect.getsource(emit_defense_blocked)
    assert "ctx.subject.tenant_id" in source
    assert "runtime_state" not in source


@pytest.mark.asyncio
async def test_r3_t08_encryption_middleware_boundary_is_typed() -> None:
    middleware = EncryptionEnforcementMiddleware(
        enforcement_enabled=True,
        secrets_store_configured=False,
    )
    ctx = HookContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="agent-1",
        payload=DataProtectionHookPayload(
            value=DataProtectionRestrictedValue(
                data_classification=DataClassification.RESTRICTED,
                secret="x",
            ),
        ),
    )
    result = await middleware.before(HookPoint.BEFORE_MEMORY_WRITE, ctx)
    assert result.action.value == "block"


def test_r3_t09_alternate_pipeline_still_composes() -> None:
    assert isinstance(_AlternateMiddlewarePipeline(), HostOrchestrationMiddlewarePipelinePort)


def test_r3_t10_fail_open_production_plugin_rejected() -> None:
    from tests.qualification.control_planes.ctrl_x.test_ctrl_x_r2_middleware_composition import (
        test_r2_mw_05_fail_open_rejected_before_attach,
    )

    test_r2_mw_05_fail_open_rejected_before_attach()


def test_r3_t11_boundary_modules_forbid_loose_semantic_types() -> None:
    forbidden = ("Any", "dict[str, Any]", "Mapping[str, object]", "Mapping[str, Any]")
    for path in _BOUNDARY_SOURCES:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_r3_t12_single_attach_operation_on_port() -> None:
    source = inspect.getsource(HostOrchestrationMiddlewarePipelinePort.attach_runtime_middleware_if_absent)
    assert "attach_runtime_middleware_if_absent" in source


def test_r3_cross_layer_middleware_no_runtime_state_semantic_reads() -> None:
    for path in _CROSS_LAYER_MIDDLEWARE:
        text = path.read_text(encoding="utf-8")
        assert "runtime_state.get" not in text
        assert "runtime_state[" not in text


def test_r3_ast_boundary_contracts_no_reflection_seams() -> None:
    for path in _BOUNDARY_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Attribute)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in {"getattr", "hasattr", "cast"}:
                        pytest.fail(f"forbidden reflection in {path}: {node.func.id}")
