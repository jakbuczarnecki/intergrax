# © Artur Czarnecki. All rights reserved.

"""EBH-2D-B-R5 — Tool registry lifecycle / execution separation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.tools.contracts.tool_profile import ToolProfile as CanonicalToolProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_CONTEXT_PATH = _REPO_ROOT / "intergrax/runtime/nexus/engine/runtime_context.py"


def _runtime_context_build_register_tools_sites() -> list[ast.Call]:
    source = _RUNTIME_CONTEXT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "register_tools":
            sites.append(node)
    return sites


def test_runtime_context_provider_registration_only_in_fallback_branch() -> None:
    sites = _runtime_context_build_register_tools_sites()
    assert len(sites) == 1
    call = sites[0]
    assert isinstance(call.func, ast.Attribute)
    loop = call.func.value
    assert isinstance(loop, ast.Name) and loop.id == "provider"
    first_arg = call.args[0]
    assert isinstance(first_arg, ast.Name) and first_arg.id == "mutable_registry"
    source = _RUNTIME_CONTEXT_PATH.read_text(encoding="utf-8")
    assert "provider.register_tools(registry" not in source
    assert "provider.register_tools(mutable_registry" in source


def test_prebuilt_probe_registry_runtime_context_skips_provider_registration() -> None:
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.tools.registry.bootstrap import register_default_tools
    from intergrax.tools.registry.factory import build_registry_from_profile
    from intergrax.tools.registry.runtime import ToolRegistry
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    register_default_tools()
    backing = ToolRegistry()
    build_registry_from_profile(
        CanonicalToolProfile(enabled_bundles=["harness"]),
        registry=backing,
    )

    class ProbeRegistry:
        def has(self, tool_id: str) -> bool:
            return backing.has(tool_id)

        def get(self, tool_id: str):
            return backing.get(tool_id)

        def activation_metadata(self, tool_id: str):
            return backing.activation_metadata(tool_id)

    class SpyProvider:
        register_calls = 0

        def register_tools(self, registry: ToolRegistry, ctx=None) -> None:
            SpyProvider.register_calls += 1
            raise AssertionError("prebuilt path must not register providers")

    SpyProvider.register_calls = 0
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_mode="off",
        tool_registry=ProbeRegistry(),
        tool_providers=(SpyProvider(),),
    )
    ctx = RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    assert SpyProvider.register_calls == 0
    assert ctx.config.tool_invoker is not None


def test_fallback_runtime_context_registers_providers_once() -> None:
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.tools.registry import ToolRegistry
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    class CountingProvider:
        register_calls = 0

        def register_tools(self, registry: ToolRegistry, ctx=None) -> None:
            CountingProvider.register_calls += 1
            assert isinstance(registry, ToolRegistry)

    CountingProvider.register_calls = 0
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tool_providers=(CountingProvider(),),
    )
    RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    assert CountingProvider.register_calls == 1


def test_application_tool_wiring_registry_complete_before_runtime_context() -> None:
    from intergrax.applications._shared.tool_wiring import build_application_tool_wiring
    from intergrax.runtime.nexus.config import RuntimeConfig
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
    from intergrax.tools.registry.bootstrap import register_default_tools
    from intergrax.tools.registry.runtime import ToolRegistry
    from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

    register_default_tools()
    profile = CanonicalToolProfile(enabled_bundles=["harness"])
    wiring = build_application_tool_wiring(profile)
    assert wiring.registry.has("harness.get_run")

    class SpyProvider:
        register_calls = 0

        def register_tools(self, registry: ToolRegistry, ctx=None) -> None:
            SpyProvider.register_calls += 1

    SpyProvider.register_calls = 0
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        enable_rag=False,
        enable_websearch=False,
        tools_mode="off",
        tool_registry=wiring.registry,
        tool_providers=(SpyProvider(),),
    )
    RuntimeContext.build(
        config=config,
        session_manager=build_in_memory_session_manager(),
    )
    assert SpyProvider.register_calls == 0
