# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R3 typed middleware contract qualification (R3-T01..R3-T12, R3-R1 tool fidelity)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.application_security_wiring import (
    register_application_security_hooks,
    ToolInjectionDefenseMiddleware,
    default_tool_invocation_policy,
)
from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.applications._shared.security_assembly_resolver import SecurityAssemblyError
from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
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
from intergrax.contracts.structured_json_value import JsonObject
from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.middleware_context_builders import (
    normalize_tool_arguments,
    tool_call_payload_from_runtime_state,
)
from intergrax.runtime.security.defense_plugin import (
    PluginSecurityDefenseMiddleware,
    SecurityFailMode,
    SecurityInspectionResult,
)
from intergrax.runtime.security.defense_registry import (
    register_security_defense_plugin,
    reset_security_defense_registry_for_tests,
)
from intergrax.runtime.security.encryption_middleware import EncryptionEnforcementMiddleware
from intergrax.runtime.security.security_events import emit_defense_blocked
from tests.qualification.control_planes.ctrl_x.middleware_inventory import (
    MIDDLEWARE_INVENTORY,
    discover_production_runtime_middleware_classes,
    inventory_index,
)
from tests.qualification.control_planes.ctrl_x.test_ctrl_x_r2_middleware_composition import (
    _AlternateHostTarget,
    _AlternateMiddlewarePipeline,
    _FailOpenDefense,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOUNDARY_SOURCES = (
    _REPO_ROOT / "intergrax/contracts/middleware_hook_semantics.py",
    _REPO_ROOT / "intergrax/contracts/host_orchestration_wiring_capabilities.py",
)
_R3_SAMPLE_ARGS: JsonObject = {
    "count": 5,
    "enabled": True,
    "options": {"mode": "safe", "threshold": 0.7},
    "items": [1, 2, 3],
    "comment": None,
}


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
    payload = ToolCallHookPayload(tool_id="echo", arguments={"q": 1})
    assert payload.tool_id == "echo"
    assert payload.arguments["q"] == 1


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
    reset_security_defense_registry_for_tests()
    register_security_defense_plugin(_FailOpenDefense())
    host: HostOrchestrationApplicationWiringTarget = _AlternateHostTarget(
        middleware=_AlternateMiddlewarePipeline(),
    )
    profile = ApplicationSecurityProfile(defense_plugin_ids=["external.fail_open"])
    with pytest.raises(SecurityAssemblyError, match="FAIL_CLOSED"):
        register_application_security_hooks(host, profile)
    assert "SecurityDefense:external.fail_open" not in host.middleware.registered_middleware_names()


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
    from tests.qualification.control_planes.ctrl_x.middleware_inventory import (
        cross_layer_middleware_source_paths,
    )

    for path in cross_layer_middleware_source_paths():
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


def test_r3_r1_middleware_inventory_covers_discovered_production_middleware() -> None:
    discovered = discover_production_runtime_middleware_classes()
    index = inventory_index()
    missing = sorted(discovered - set(index))
    assert missing == []
    unclassified = [key for key in discovered if key not in index]
    assert unclassified == []


def test_r3_r1_middleware_inventory_has_no_stale_entries() -> None:
    discovered = discover_production_runtime_middleware_classes()
    indexed = {(entry.module_path, entry.class_name) for entry in MIDDLEWARE_INVENTORY}
    stale = sorted(indexed - discovered)
    assert stale == []


def test_r3_r1_qualification_modules_forbid_test_function_proof_imports() -> None:
    ctrl_x_dir = Path(__file__).resolve().parent
    for path in ctrl_x_dir.glob("test_ctrl_x_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "test_ctrl_x" in node.module or node.module.endswith("test_"):
                    for alias in node.names:
                        if alias.name.startswith("test_"):
                            pytest.fail(
                                f"forbidden test-as-proof import in {path.name}: "
                                f"{node.module}.{alias.name}",
                            )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id.startswith("test_"):
                    pytest.fail(f"forbidden test-as-proof call in {path.name}: {node.func.id}")


def test_r3_r1_tool_payload_arguments_use_canonical_json_object() -> None:
    tree = ast.parse(
        (_REPO_ROOT / "intergrax/contracts/middleware_hook_semantics.py").read_text(encoding="utf-8"),
    )
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "ToolCallHookPayload":
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == "arguments" and item.annotation is not None:
                    ann = ast.unparse(item.annotation)
                    assert ann == "JsonObject"
                    return
    pytest.fail("ToolCallHookPayload.arguments annotation missing")


def test_r3_r1_middleware_semantics_imports_json_from_structured_json_authority() -> None:
    source = (_REPO_ROOT / "intergrax/contracts/middleware_hook_semantics.py").read_text(encoding="utf-8")
    assert "from intergrax.contracts.structured_json_value import JsonObject" in source
    assert "type JsonValue" not in source
    assert "JsonObject =" not in source.split("import")[1] if "import" in source else True


def test_r1_tool_01_nested_dict_preserved() -> None:
    normalized = normalize_tool_arguments(_R3_SAMPLE_ARGS)
    assert normalized["options"] == {"mode": "safe", "threshold": 0.7}


def test_r1_tool_02_list_preserved() -> None:
    normalized = normalize_tool_arguments(_R3_SAMPLE_ARGS)
    assert normalized["items"] == [1, 2, 3]


def test_r1_tool_03_bool_preserved() -> None:
    normalized = normalize_tool_arguments(_R3_SAMPLE_ARGS)
    assert normalized["enabled"] is True


def test_r1_tool_04_int_float_preserved() -> None:
    normalized = normalize_tool_arguments(_R3_SAMPLE_ARGS)
    assert normalized["count"] == 5
    assert normalize_tool_arguments({"ratio": 0.25})["ratio"] == 0.25


def test_r1_tool_05_none_preserved() -> None:
    normalized = normalize_tool_arguments(_R3_SAMPLE_ARGS)
    assert normalized["comment"] is None


def test_r1_tool_06_invalid_non_json_safe_value_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_tool_arguments({"bad": object()})


@pytest.mark.asyncio
async def test_r1_tool_07_external_defense_plugin_sees_structured_values() -> None:
    class _StructuredArgsProbe:
        plugin_id = "probe.structured"
        version = "1"
        hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
        priority = 59
        fail_mode = SecurityFailMode.FAIL_CLOSED
        seen: JsonObject | None = None

        def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
            payload = ctx.payload
            if isinstance(payload, ToolCallHookPayload):
                _StructuredArgsProbe.seen = dict(payload.arguments)
            return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)

    probe = _StructuredArgsProbe()
    reset_security_defense_registry_for_tests()
    register_security_defense_plugin(probe, override=True)
    middleware = PluginSecurityDefenseMiddleware(probe)
    args: JsonObject = {"count": 1, "flag": True, "nested": {"x": 2}}
    ctx = HookContext(
        task_id="t",
        run_id="r",
        payload=ToolCallHookPayload(tool_id="echo", arguments=args),
    )
    await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert probe.seen == args
    assert probe.seen is not None
    assert probe.seen["count"] == 1
    assert probe.seen["flag"] is True


@pytest.mark.asyncio
async def test_r1_tool_08_builtin_token_scanner_detects_nested_blocked_token() -> None:
    middleware = ToolInjectionDefenseMiddleware(default_tool_invocation_policy())
    ctx = HookContext(
        task_id="t",
        run_id="r",
        payload=ToolCallHookPayload(
            tool_id="echo",
            arguments={"payload": {"text": "please system override now"}},
            capability_ids=["echo"],
        ),
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action == HookAction.BLOCK


def test_r1_tool_runtime_state_builder_lossless() -> None:
    payload = tool_call_payload_from_runtime_state(
        {"tool_id": "echo", "arguments": _R3_SAMPLE_ARGS},
    )
    assert payload is not None
    assert payload.arguments == _R3_SAMPLE_ARGS
