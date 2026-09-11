# © Artur Czarnecki. All rights reserved.

"""U3 — agent / plugin execution closure static and contract gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_CONTEXT = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py"
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_UAEP_GATEWAY = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "uaep_tool_gateway.py"
_RUNTIME_TOOL_HELPERS = _REPO_ROOT / "intergrax" / "agents" / "authoring" / "runtime_tool_helpers.py"
_U3_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_U3_AGENT_PLUGIN_EXECUTION_QUALIFICATION.md"
)
_PRODUCTION_AGENT_SURFACES = (
    _REPO_ROOT / "intergrax" / "agents" / "uaep.py",
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "runtime_tool_helpers.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "uaep_tool_gateway.py",
)
_GOVERNANCE_GRANT_WIRING = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared" / "agent_runtime_governance_wiring.py"
)


def test_u3_qualification_artifact_present() -> None:
    assert _U3_QUALIFICATION.is_file()


def test_u3_runtime_context_wires_agent_runtime_governance_into_invoker() -> None:
    source = _RUNTIME_CONTEXT.read_text(encoding="utf-8")
    assert "agent_runtime_governance=config.agent_runtime_governance" in source
    assert "agent_runtime_governance is required when production_mode=True" in source


def test_u3_invoker_fail_closed_when_production_without_governance() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    assert "agent_runtime_governance_not_configured" in source
    assert "state.context.config.production_mode" in source


def test_u3_uaep_tool_path_uses_runtime_tool_gateway_not_local_invoker() -> None:
    gateway_source = _UAEP_GATEWAY.read_text(encoding="utf-8")
    assert "RuntimeToolGateway.for_state" in gateway_source
    assert "RuntimeToolInvoker(" not in gateway_source
    helpers_source = _RUNTIME_TOOL_HELPERS.read_text(encoding="utf-8")
    assert "exec_ctx.invoke_tool" in helpers_source
    assert "RuntimeToolInvoker(" not in helpers_source


def test_u3_production_agent_surfaces_do_not_construct_runtime_tool_invoker() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_AGENT_SURFACES:
        source = path.read_text(encoding="utf-8")
        if "RuntimeToolInvoker(" in source:
            violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert violations == [], (
        "supported production agent tooling must not construct RuntimeToolInvoker locally:\n"
        + "\n".join(violations)
    )


def test_u3_runtime_config_declares_agent_runtime_governance_field() -> None:
    config_path = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "config.py"
    source = config_path.read_text(encoding="utf-8")
    assert "agent_runtime_governance:" in source
    assert "AgentRuntimeGovernancePort" in source


def test_u3_governance_grant_materialization_must_not_instantiate_agent_types() -> None:
    source = _GOVERNANCE_GRANT_WIRING.read_text(encoding="utf-8")
    assert "resolved_agent_type()()" not in source
    assert "resolved_agent_type()" not in source
    assert "build_agent_from_binding" not in source
    assert "invoke_agent_factory" not in source
    assert "resolve_agent_type(" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"Agent", "resolve_agent_type"}:
                raise AssertionError(
                    f"governance grant wiring must not call {node.func.id}() for discovery",
                )
