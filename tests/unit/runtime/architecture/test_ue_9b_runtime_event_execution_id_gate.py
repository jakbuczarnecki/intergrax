# © Artur Czarnecki. All rights reserved.

"""UE-9B — RuntimeEvent execution_id architecture gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEvent

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_RUNTIME_EVENT_PRODUCTION_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "signals.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "spine_consolidation.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "context_skill_recording.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "trace_bridge.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "planner_events.py",
    _REPO_ROOT / "intergrax" / "runtime" / "events" / "ingestion_events.py",
    _REPO_ROOT / "intergrax" / "runtime" / "kernel" / "step_kernel.py",
    _REPO_ROOT / "intergrax" / "runtime" / "middleware" / "trace_middleware.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "agent_router.py",
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "trace_scope.py",
    _REPO_ROOT / "intergrax" / "agents" / "uaep.py",
    _REPO_ROOT / "intergrax" / "contracts" / "runtime_execution_context.py",
)


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


def test_runtime_event_has_canonical_execution_id_field() -> None:
    assert "execution_id" in RuntimeEvent.model_fields


def test_production_runtime_event_emitters_do_not_mint_execution_id() -> None:
    violations: list[str] = []
    for path in _RUNTIME_EVENT_PRODUCTION_PATHS:
        violations.extend(
            _collect_forbidden_calls(path, frozenset({"mint_execution_id"}))
        )
    assert violations == []
