# © Artur Czarnecki. All rights reserved.

"""Architecture gate — GraphExecutor must not bypass AgentExecutor (UE-7B)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_GRAPH_EXECUTOR_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "nexus"
    / "execution"
    / "graph_executor.py"
)

_FORBIDDEN_AGENT_ENGINE_CALLS = frozenset({"run_agent_with_result", "run_agent"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _collect_agent_engine_bypass_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(_repo_root()).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in _FORBIDDEN_AGENT_ENGINE_CALLS:
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "AgentEngine":
            continue
        violations.append(f"{rel}:{node.lineno} calls AgentEngine.{func.attr}")
    return violations


def test_graph_executor_has_no_direct_agent_engine_execution_bypass() -> None:
    violations = _collect_agent_engine_bypass_calls(_GRAPH_EXECUTOR_PATH)
    assert violations == []
