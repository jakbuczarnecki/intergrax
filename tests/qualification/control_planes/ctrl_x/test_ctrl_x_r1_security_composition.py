# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R1 security fail-closed and canonical composition proofs."""

from __future__ import annotations

import inspect

import pytest

from intergrax.applications._shared.application_security_wiring import register_application_security_hooks
from intergrax.applications._shared.security_assembly_resolver import SecurityAssemblyError
from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.host_orchestration_wiring_capabilities import HostOrchestrationRuntimeEventPort
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.security.defense_plugin import (
    PluginSecurityDefenseMiddleware,
    SecurityFailMode,
    SecurityInspectionResult,
)
from intergrax.runtime.security.defense_registry import (
    register_security_defense_plugin,
    reset_security_defense_registry_for_tests,
)
from intergrax.runtime.security.security_events import emit_defense_blocked

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FailOpenDefense:
    plugin_id = "external.fail_open"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 57
    fail_mode = SecurityFailMode.FAIL_OPEN

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        return SecurityInspectionResult(
            allowed=False,
            reasons=["fail-open probe"],
            plugin_id=self.plugin_id,
            hook_point=point.value,
        )


class _FailClosedDefense:
    plugin_id = "external.fail_closed"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 57
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_security_defense_registry_for_tests()


def _pipeline_names(loop: NexusLoop) -> set[str]:
    pipeline = loop._middleware  # noqa: SLF001
    assert isinstance(pipeline, MiddlewarePipeline)
    return {mw.name for mw in pipeline._middleware}  # noqa: SLF001


def test_r1_sec_01_canonical_composition_rejects_external_fail_open() -> None:
    register_security_defense_plugin(_FailOpenDefense())
    loop = NexusLoop(AgentRegistry())
    profile = ApplicationSecurityProfile(defense_plugin_ids=["external.fail_open"])
    with pytest.raises(SecurityAssemblyError, match="FAIL_CLOSED"):
        register_application_security_hooks(loop, profile)


def test_r1_sec_02_fail_closed_external_defense_accepted() -> None:
    register_security_defense_plugin(_FailClosedDefense())
    loop = NexusLoop(AgentRegistry())
    profile = ApplicationSecurityProfile(defense_plugin_ids=["external.fail_closed"])
    register_application_security_hooks(loop, profile)
    assert "SecurityDefense:external.fail_closed" in _pipeline_names(loop)


def test_r1_sec_03_rejected_fail_open_middleware_not_attached() -> None:
    register_security_defense_plugin(_FailOpenDefense())
    loop = NexusLoop(AgentRegistry())
    profile = ApplicationSecurityProfile(defense_plugin_ids=["external.fail_open"])
    with pytest.raises(SecurityAssemblyError):
        register_application_security_hooks(loop, profile)
    assert "SecurityDefense:external.fail_open" not in _pipeline_names(loop)


@pytest.mark.asyncio
async def test_r1_sec_04_rejected_fail_open_no_protected_hook_effect() -> None:
    register_security_defense_plugin(_FailOpenDefense())
    loop = NexusLoop(AgentRegistry())
    profile = ApplicationSecurityProfile(defense_plugin_ids=["external.fail_open"])
    with pytest.raises(SecurityAssemblyError):
        register_application_security_hooks(loop, profile)
    pipeline = loop._middleware  # noqa: SLF001
    assert isinstance(pipeline, MiddlewarePipeline)
    ctx = HookContext(
        task_id="t1",
        run_id="r1",
        phase=ExecutionPhase.STEP_EXECUTION,
        runtime_state={"tool_id": "echo", "arguments": {}},
    )
    result = await pipeline.run_before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action != HookAction.BLOCK


@pytest.mark.asyncio
async def test_r1_sec_05_tenant_scoped_fail_closed_defense_still_blocks() -> None:
    register_security_defense_plugin(_FailClosedDefense())
    middleware = PluginSecurityDefenseMiddleware(
        _FailClosedDefense(),
        enforce_tenant_scope=True,
    )
    ctx = HookContext(
        task_id="t1",
        run_id="r1",
        phase=ExecutionPhase.STEP_EXECUTION,
        runtime_state={
            "tenant_id": "tenant-a",
            "resource_tenant_id": "tenant-b",
            "tool_id": "echo",
        },
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action == HookAction.BLOCK


def test_r1_sec_06_security_emitters_use_host_orchestration_event_port() -> None:
    source = inspect.getsource(emit_defense_blocked)
    assert "HostOrchestrationRuntimeEventPort" in source
    assert "_DefenseEventBusPort" not in source
    assert isinstance(RuntimeEventBus(), HostOrchestrationRuntimeEventPort)


def test_r1_sec_lab_manual_fail_open_middleware_still_constructible() -> None:
    middleware = PluginSecurityDefenseMiddleware(_FailOpenDefense())
    assert middleware.name == "SecurityDefense:external.fail_open"
