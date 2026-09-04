# © Artur Czarnecki. All rights reserved.

"""Unit tests for shared AgentRegistry lifecycle AST primitives."""

from __future__ import annotations

import ast

import pytest

from intergrax.runtime.architecture.agent_lifecycle_bypass_ast import (
    LifecycleAstViolationKind,
    collect_agent_registry_lifecycle_violations,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _violations(source: str) -> tuple[str, ...]:
    tree = ast.parse(source)
    return tuple(
        violation.symbol
        for violation in collect_agent_registry_lifecycle_violations(tree)
    )


def test_direct_agent_registry_construction_is_violation() -> None:
    source = (
        "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry()\n"
    )
    assert _violations(source) == ("AgentRegistry",)


def test_aliased_agent_registry_class_construction_is_violation() -> None:
    source = (
        "from intergrax.runtime.registry.agent_registry import AgentRegistry as AR\n"
        "registry = AR()\n"
    )
    assert _violations(source) == ("AgentRegistry",)


def test_aliased_module_agent_registry_construction_is_violation() -> None:
    source = (
        "import intergrax.runtime.registry.agent_registry as registry_module\n"
        "registry = registry_module.AgentRegistry()\n"
    )
    assert _violations(source) == ("AgentRegistry",)


def test_qualified_module_agent_registry_construction_is_violation() -> None:
    source = (
        "import intergrax.runtime.registry.agent_registry\n"
        "registry = intergrax.runtime.registry.agent_registry.AgentRegistry()\n"
    )
    assert _violations(source) == ("AgentRegistry",)


def test_from_agents_is_violation() -> None:
    source = (
        "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry.from_agents({})\n"
    )
    assert _violations(source) == ("AgentRegistry.from_agents",)


def test_local_register_on_constructed_registry_is_violation() -> None:
    source = (
        "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "from some_agent_package import SomeAgent\n"
        "registry = AgentRegistry()\n"
        "registry.register(SomeAgent())\n"
    )
    symbols = _violations(source)
    assert "AgentRegistry" in symbols
    assert "register" in symbols


def test_unrelated_agent_registry_passes() -> None:
    source = (
        "import some_other_package as other\n"
        "registry = other.AgentRegistry()\n"
    )
    assert _violations(source) == ()


def test_agent_registry_read_passes() -> None:
    source = (
        "from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead\n"
        "def handler(registry: AgentRegistryRead) -> None:\n"
        "    return None\n"
    )
    assert _violations(source) == ()


def test_agent_binding_mount_passes() -> None:
    source = (
        "from intergrax.applications.contracts.manifest import AgentBinding\n"
        "BINDING = AgentBinding.mount(contract_id='demo.agent', capabilities=['demo.run'])\n"
    )
    assert _violations(source) == ()


def test_violation_kinds_are_typed() -> None:
    source = (
        "from intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry.from_agents({})\n"
    )
    tree = ast.parse(source)
    violations = collect_agent_registry_lifecycle_violations(tree)
    assert len(violations) == 1
    assert violations[0].kind is LifecycleAstViolationKind.AGENT_REGISTRY_FROM_AGENTS
