# © Artur Czarnecki. All rights reserved.

"""UE-8B2 — budget consumption architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUDGET_TICKS = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "budget" / "budget_ticks.py"
_CONSUMPTION = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "budget" / "consumption.py"
_LLM_ADAPTER = _REPO_ROOT / "intergrax" / "llm_adapters" / "contracts" / "llm_adapter.py"
_TOOL_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "tool_loop.py"
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_FORBIDDEN_NEXUS_LEDGER = frozenset({"consume_budget", "create_execution_budget_ledger"})


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(path: Path, forbidden: frozenset[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno} calls {name}")
    return violations


def test_budget_ticks_uses_consumption_layer() -> None:
    source = _BUDGET_TICKS.read_text(encoding="utf-8")
    assert "intergrax.runtime.execution.budget.consumption" in source
    assert "consume_rag_invocation" in source
    assert "consume_tool_call" in source


def test_consumption_layer_is_explicit_typed_api() -> None:
    source = _CONSUMPTION.read_text(encoding="utf-8")
    assert "def consume_llm_call" in source
    assert "def consume_tool_call" in source
    assert "getattr" not in source
    assert "consume(" not in source


def test_llm_adapter_usage_log_delegates_to_consumption() -> None:
    source = _LLM_ADAPTER.read_text(encoding="utf-8")
    assert "consume_llm_call" in source
    assert "consume_llm_token_usage" in source


def test_tool_loop_uses_record_tool_call_and_enforce() -> None:
    source = _TOOL_LOOP.read_text(encoding="utf-8")
    assert "record_tool_call_and_enforce" in source


def test_nexus_loop_does_not_consume_budget_directly() -> None:
    violations = _collect_forbidden_calls(_NEXUS_LOOP, _FORBIDDEN_NEXUS_LEDGER)
    assert violations == []
