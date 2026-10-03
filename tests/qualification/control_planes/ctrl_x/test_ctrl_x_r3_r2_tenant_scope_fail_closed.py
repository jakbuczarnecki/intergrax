# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R3-R2 tenant scope fail-closed qualification."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.applications._shared.application_security_wiring import TenantSecurityMiddleware
from intergrax.contracts.middleware_hook_semantics import (
    MiddlewareExecutionSubjectFacet,
    ToolCallHookPayload,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import (
    PluginSecurityDefenseMiddleware,
    SecurityFailMode,
    SecurityInspectionResult,
)
from tests.qualification.control_planes.ctrl_x.middleware_inventory import (
    discover_production_runtime_middleware_classes,
    inventory_index,
)
from tests.support.middleware_hook_test_context import tenant_intake_hook_context_for_test

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRODUCTION_SCAN_ROOTS = (
    _REPO_ROOT / "intergrax/runtime",
    _REPO_ROOT / "intergrax/applications/_shared",
    _REPO_ROOT / "intergrax/harness",
)


@dataclass(frozen=True, slots=True)
class _SubjectProducerClassification:
    producer_id: str
    request_tenant_source: str
    resource_tenant_source: str
    normalization: str
    unscoped_legitimate: str


_SUBJECT_PRODUCER_INVENTORY: tuple[_SubjectProducerClassification, ...] = (
    _SubjectProducerClassification(
        "middleware_context_builders.subject_from_runtime_state",
        "runtime_state tenant_id (optional str, no Security normalize at build)",
        "runtime_state resource_tenant_id (optional str)",
        "Security middleware normalizes at enforcement",
        "when both absent in runtime_state",
    ),
    _SubjectProducerClassification(
        "tool_hooks.tool_hook_context",
        "RuntimeState.tenant_id",
        "same as request (request-scoped tool)",
        "Security middleware normalizes at enforcement",
        "no — tool path sets both from state.tenant_id",
    ),
    _SubjectProducerClassification(
        "llm_hooks.llm_hook_context",
        "tenant_id arg strip-only at build",
        "absent unless extended by caller",
        "Security middleware normalizes at enforcement",
        "yes when tenant omitted",
    ),
    _SubjectProducerClassification(
        "nexus_lifecycle_hooks.nexus_lifecycle_hook_context",
        "task.tenant_id",
        "resource_tenant_id_for_task(task)",
        "Security middleware normalizes at enforcement",
        "only when task has no tenant and no resource tenant",
    ),
    _SubjectProducerClassification(
        "context_manager context build hook",
        "task.tenant_id",
        "absent on subject facet",
        "Security middleware normalizes at enforcement",
        "when task.tenant_id absent",
    ),
    _SubjectProducerClassification(
        "tests.support.middleware_hook_test_context",
        "explicit test tenant_id",
        "explicit test resource_tenant_id (no substitution)",
        "Security middleware normalizes at enforcement",
        "test-controlled",
    ),
    _SubjectProducerClassification(
        "ctrl_x qualification fixtures",
        "explicit MiddlewareExecutionSubjectFacet fields",
        "explicit MiddlewareExecutionSubjectFacet fields",
        "Security middleware normalizes at enforcement",
        "test-controlled",
    ),
)


def test_r3_r2_subject_producer_inventory_classified() -> None:
    assert len(_SUBJECT_PRODUCER_INVENTORY) >= 7
    assert all(entry.producer_id for entry in _SUBJECT_PRODUCER_INVENTORY)


@pytest.mark.asyncio
async def test_r3_r2_task_intake_blank_or_missing_tenant_blocked() -> None:
    middleware = TenantSecurityMiddleware()
    for tenant_id in (None, "", "   "):
        ctx = tenant_intake_hook_context_for_test(
            tenant_id=tenant_id,
            resource_tenant_id=None,
            phase=ExecutionPhase.INTAKE,
        )
        result = await middleware.before(HookPoint.BEFORE_TASK_INTAKE, ctx)
        assert result.action == HookAction.BLOCK


@pytest.mark.asyncio
async def test_r3_r2_task_intake_request_resource_allow_and_deny() -> None:
    middleware = TenantSecurityMiddleware()
    allow_ctx = tenant_intake_hook_context_for_test(
        tenant_id="A",
        resource_tenant_id=None,
    )
    allow = await middleware.before(HookPoint.BEFORE_TASK_INTAKE, allow_ctx)
    assert allow.action != HookAction.BLOCK

    deny_ctx = tenant_intake_hook_context_for_test(
        tenant_id="A",
        resource_tenant_id="B",
    )
    deny = await middleware.before(HookPoint.BEFORE_TASK_INTAKE, deny_ctx)
    assert deny.action == HookAction.BLOCK


class _InspectCountProbe:
    plugin_id = "probe.tenant.order"
    version = "1"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 59
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def __init__(self) -> None:
        self.inspect_calls = 0

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        self.inspect_calls += 1
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)


@pytest.mark.asyncio
async def test_r3_r2_defense_plugin_tenant_fail_closed_before_inspect() -> None:
    probe = _InspectCountProbe()
    middleware = PluginSecurityDefenseMiddleware(probe, enforce_tenant_scope=True)
    payload = ToolCallHookPayload(tool_id="echo", arguments={})

    unscoped_ok = HookContext(
        task_id="t",
        run_id="r",
        subject=MiddlewareExecutionSubjectFacet(tenant_id=None, resource_tenant_id=None),
        payload=payload,
    )
    await middleware.before(HookPoint.BEFORE_TOOL_CALL, unscoped_ok)
    assert probe.inspect_calls == 1

    probe = _InspectCountProbe()
    middleware = PluginSecurityDefenseMiddleware(probe, enforce_tenant_scope=True)
    missing_request = HookContext(
        task_id="t",
        run_id="r",
        subject=MiddlewareExecutionSubjectFacet(tenant_id=None, resource_tenant_id="B"),
        payload=payload,
    )
    blocked = await middleware.before(HookPoint.BEFORE_TOOL_CALL, missing_request)
    assert blocked.action == HookAction.BLOCK
    assert probe.inspect_calls == 0

    probe = _InspectCountProbe()
    middleware = PluginSecurityDefenseMiddleware(probe, enforce_tenant_scope=True)
    matched = HookContext(
        task_id="t",
        run_id="r",
        subject=MiddlewareExecutionSubjectFacet(tenant_id="A", resource_tenant_id="A"),
        payload=payload,
    )
    await middleware.before(HookPoint.BEFORE_TOOL_CALL, matched)
    assert probe.inspect_calls == 1

    probe = _InspectCountProbe()
    middleware = PluginSecurityDefenseMiddleware(probe, enforce_tenant_scope=True)
    mismatch = HookContext(
        task_id="t",
        run_id="r",
        subject=MiddlewareExecutionSubjectFacet(tenant_id="A", resource_tenant_id="B"),
        payload=payload,
    )
    blocked_mismatch = await middleware.before(HookPoint.BEFORE_TOOL_CALL, mismatch)
    assert blocked_mismatch.action == HookAction.BLOCK
    assert probe.inspect_calls == 0


@pytest.mark.asyncio
async def test_r3_r2_defense_plugin_emits_block_on_missing_request_tenant() -> None:
    from intergrax.contracts.execution_identity import canonical_execution_identity_scope
    from intergrax.runtime.events.spine_consolidation import KIND_DEFENSE_BLOCKED

    bus = RuntimeEventBus()
    from intergrax.runtime.security.security_observability import wire_security_spine_subscriber

    wire_security_spine_subscriber(bus)

    class _AllowPlugin:
        plugin_id = "allow.all"
        version = "1"
        hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
        priority = 59
        fail_mode = SecurityFailMode.FAIL_CLOSED

        def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
            return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)

    middleware = PluginSecurityDefenseMiddleware(_AllowPlugin(), event_bus=bus)
    ctx = HookContext(
        task_id="t",
        run_id="r",
        subject=MiddlewareExecutionSubjectFacet(tenant_id=None, resource_tenant_id="tenant-b"),
        payload=ToolCallHookPayload(tool_id="echo"),
    )
    with canonical_execution_identity_scope("r"):
        result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action == HookAction.BLOCK
    kinds = [event.event_kind for event in bus.history]
    assert KIND_DEFENSE_BLOCKED in kinds


def _discover_structural_middleware_shapes() -> frozenset[tuple[str, str]]:
    structural: set[tuple[str, str]] = set()
    for root in _PRODUCTION_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            module_path = path.relative_to(_REPO_ROOT).as_posix()
            inherits_runtime: set[str] = set()
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                has_shape = False
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name in {"before", "after"}:
                            has_shape = True
                has_priority = any(
                    isinstance(item, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "priority"
                        for target in item.targets
                    )
                    for item in node.body
                )
                if not (has_shape and has_priority):
                    continue
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "RuntimeMiddleware":
                        inherits_runtime.add(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == "RuntimeMiddleware":
                        inherits_runtime.add(node.name)
                if node.name not in inherits_runtime:
                    structural.add((module_path, node.name))
    return frozenset(structural)


def test_r3_r2_structural_middleware_inventory_complete() -> None:
    discovered = discover_production_runtime_middleware_classes()
    index = inventory_index()
    structural = _discover_structural_middleware_shapes()
    omitted = sorted(structural - discovered)
    assert omitted == [], f"structural middleware not in inventory: {omitted}"
    unclassified = [key for key in discovered if key not in index]
    assert unclassified == []
    stale = sorted(set(index) - discovered)
    assert stale == []
