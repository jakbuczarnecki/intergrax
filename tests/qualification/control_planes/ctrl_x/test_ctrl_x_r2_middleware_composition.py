# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R2 contract-pure middleware composition proofs (R2-A)."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.applications._shared.application_security_wiring import register_application_security_hooks
from intergrax.applications._shared.security_assembly_resolver import SecurityAssemblyError
from intergrax.applications.contracts.environment_profile import ApplicationSecurityProfile
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationHookRegistryPort,
    HostOrchestrationMiddlewarePipelinePort,
    HostOrchestrationRuntimeEventPort,
    HostOrchestrationRuntimeMiddlewareRegistration,
)
from intergrax.contracts.middleware_hook_point import HookPoint
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.hooks.hook_context import HookContext, HookResult
from intergrax.runtime.middleware.base import RuntimeMiddleware
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.security.defense_plugin import SecurityFailMode, SecurityInspectionResult
from intergrax.runtime.security.defense_registry import (
    register_security_defense_plugin,
    reset_security_defense_registry_for_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SECURITY_WIRING = _REPO_ROOT / "intergrax/applications/_shared/application_security_wiring.py"


class _StubHookRegistry:
    def register(
        self,
        point: str,
        handler: object,
        *,
        priority: int = 100,
        name: str | None = None,
        hook_id: str | None = None,
    ) -> str:
        _ = (point, handler, priority, name, hook_id)
        return "stub-hook"


@dataclass
class _AlternateMiddlewarePipeline:
    """Non-MiddlewarePipeline host port implementation for replaceability proof."""

    _middleware: list[HostOrchestrationRuntimeMiddlewareRegistration] = field(default_factory=list)
    _hooks: HostOrchestrationHookRegistryPort = field(default_factory=_StubHookRegistry)
    _hook_timeout_seconds: float | None = None
    _event_bus: HostOrchestrationRuntimeEventPort | None = None

    @property
    def hooks(self) -> HostOrchestrationHookRegistryPort:
        return self._hooks

    @property
    def hook_timeout_seconds(self) -> float | None:
        return self._hook_timeout_seconds

    def configure_hook_runtime(
        self,
        *,
        hook_timeout_seconds: float | None,
        event_bus: HostOrchestrationRuntimeEventPort | None,
    ) -> None:
        self._hook_timeout_seconds = hook_timeout_seconds
        self._event_bus = event_bus

    def registered_middleware_names(self) -> frozenset[str]:
        return frozenset(mw.name for mw in self._middleware)

    def attach_runtime_middleware_if_absent(
        self,
        middleware: HostOrchestrationRuntimeMiddlewareRegistration,
    ) -> None:
        if any(mw.name == middleware.name for mw in self._middleware):
            return
        self._middleware = sorted(
            [*self._middleware, middleware],
            key=lambda item: item.priority,
        )


@dataclass
class _AlternateHostTarget:
    middleware: HostOrchestrationMiddlewarePipelinePort
    event_bus: HostOrchestrationRuntimeEventPort = field(default_factory=RuntimeEventBus)


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


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_security_defense_registry_for_tests()


def test_r2_mw_01_application_security_wiring_imports_no_middleware_pipeline() -> None:
    tree = ast.parse(_SECURITY_WIRING.read_text(encoding="utf-8"))
    imports = {
        node.names[0].name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.middleware.pipeline"
    }
    assert imports == set()
    source = inspect.getsource(
        __import__(
            "intergrax.applications._shared.application_security_wiring",
            fromlist=["register_application_security_hooks"],
        ),
    )
    assert "MiddlewarePipeline" not in source
    assert "isinstance(pipeline, MiddlewarePipeline)" not in source


def test_r2_mw_02_alternate_pipeline_port_accepts_security_middleware() -> None:
    pipeline = _AlternateMiddlewarePipeline()
    host: HostOrchestrationApplicationWiringTarget = _AlternateHostTarget(middleware=pipeline)
    profile = ApplicationSecurityProfile(prompt_defense_enabled=True)
    register_application_security_hooks(host, profile)
    assert "PromptDefenseMiddleware" in pipeline.registered_middleware_names()


def test_r2_mw_03_middleware_pipeline_satisfies_host_port() -> None:
    pipeline = MiddlewarePipeline()
    assert isinstance(pipeline, HostOrchestrationMiddlewarePipelinePort)


def test_r2_mw_04_exactly_one_sanctioned_attach_operation() -> None:
    port_source = inspect.getsource(HostOrchestrationMiddlewarePipelinePort.attach_runtime_middleware_if_absent)
    assert "attach_runtime_middleware_if_absent" in port_source
    pipeline_source = inspect.getsource(MiddlewarePipeline)
    assert "attach_tier1_middleware_if_absent" not in pipeline_source


def test_r2_mw_05_fail_open_rejected_before_attach() -> None:
    register_security_defense_plugin(_FailOpenDefense())
    host: HostOrchestrationApplicationWiringTarget = _AlternateHostTarget(
        middleware=_AlternateMiddlewarePipeline(),
    )
    profile = ApplicationSecurityProfile(defense_plugin_ids=["external.fail_open"])
    with pytest.raises(SecurityAssemblyError, match="FAIL_CLOSED"):
        register_application_security_hooks(host, profile)
    assert "SecurityDefense:external.fail_open" not in host.middleware.registered_middleware_names()


def test_r2_mw_06_nexus_loop_uses_port_attach_not_concrete_narrowing() -> None:
    loop = NexusLoop(AgentRegistry())
    profile = ApplicationSecurityProfile(prompt_defense_enabled=True)
    register_application_security_hooks(loop, profile)
    assert isinstance(loop.middleware, HostOrchestrationMiddlewarePipelinePort)
    assert "PromptDefenseMiddleware" in loop.middleware.registered_middleware_names()
