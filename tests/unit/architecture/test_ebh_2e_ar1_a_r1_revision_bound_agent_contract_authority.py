# © Artur Czarnecki. All rights reserved.

"""EBH-2E-AR1-A-R1 — revision-bound AgentContract snapshot authority gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.applications._shared import agent_certification_wiring
from intergrax.applications._shared import capability_graph_deploy_gate
from intergrax.applications._shared import health_score_wiring
from intergrax.applications._shared import package_wiring

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_STRICT_SHARED_MODULES = (
    agent_certification_wiring,
    capability_graph_deploy_gate,
    health_score_wiring,
    package_wiring,
)


def _resolve_calls_in_module(module_path: Path) -> list[ast.Call]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "resolve_agent_contract_from_binding":
            calls.append(node)
        if isinstance(func, ast.Attribute) and func.attr == "resolve_agent_contract_from_binding":
            calls.append(node)
    return calls


@pytest.mark.parametrize(
    "module",
    _STRICT_SHARED_MODULES,
    ids=[item.__name__.split(".")[-1] for item in _STRICT_SHARED_MODULES],
)
def test_strict_shared_modules_do_not_call_dynamic_contract_resolver(module) -> None:
    path = Path(module.__file__)
    assert path.is_file()
    calls = _resolve_calls_in_module(path)
    assert calls == []


def test_roster_authority_module_owns_compatibility_resolver() -> None:
    from intergrax.applications._shared import roster_agent_contract_authority

    path = Path(roster_agent_contract_authority.__file__)
    source = path.read_text(encoding="utf-8")
    assert "resolve_agent_contract_from_binding" in source
    assert "materialize_manifest_contract_authority_lab_compat" in source
