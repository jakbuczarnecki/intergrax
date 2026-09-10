# © Artur Czarnecki. All rights reserved.

"""U2 — tool / integration side-effect closure static and contract gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPENSATION_WORKER = (
    _REPO_ROOT / "intergrax" / "agents" / "persistence" / "compensation_queue_worker.py"
)
_COMPENSATION_TOOL_SESSION = (
    _REPO_ROOT
    / "intergrax"
    / "agents"
    / "persistence"
    / "compensation_tool_invoke_session.py"
)
_U2_COMPENSATION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "compensation_side_effect_execution.py"
)
_U2_QUALIFICATION = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_U2_TOOL_INTEGRATION_SIDE_EFFECT_QUALIFICATION.md"
)


def test_u2_compensation_worker_uses_admitted_side_effect_port() -> None:
    source = _COMPENSATION_WORKER.read_text(encoding="utf-8")
    assert "CompensationSideEffectExecutionPort" in source
    assert "side_effect_execution" in source
    assert "DeclarativeToolInvoker" not in source
    tree = ast.parse(source)
    invoke_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(getattr(node.func, "attr", None), str)
        and node.func.attr == "invoke"
    ]
    assert invoke_calls == [], "compensation worker must not call tool invoker.invoke directly"


def test_u2_qualification_artifact_present() -> None:
    assert _U2_QUALIFICATION.is_file()


def test_u2_compensation_contract_has_no_any_on_public_boundary() -> None:
    source = _U2_COMPENSATION_CONTRACT.read_text(encoding="utf-8")
    assert "Any" not in source
    assert "dict[str, Any]" not in source


def test_u2_compensation_session_has_no_catalog_invoker_type_discrimination() -> None:
    source = _COMPENSATION_TOOL_SESSION.read_text(encoding="utf-8")
    assert "CatalogDeclarativeToolInvoker" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "isinstance":
            for arg in node.args[1:]:
                if isinstance(arg, ast.Name) and arg.id == "CatalogDeclarativeToolInvoker":
                    raise AssertionError(
                        "compensation session must not isinstance CatalogDeclarativeToolInvoker",
                    )
