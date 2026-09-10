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
